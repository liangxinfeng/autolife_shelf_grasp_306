"""Position-first grasp IK with bounded orientation relaxation and path checks.

No motion publishers. Scene scope is deliberately the user's isolated tabletop
experiment, not arbitrary cluttered shelves. Unknown external objects are NOT
claimed to have been checked.
"""
import time
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from autolife_handeye_calibration_306.core import Kinematics, ARMS, TORSO
from autolife_handeye_calibration_306.verification_core import wrist_frame
from autolife_handeye_calibration_306.auto_plan_worker import CollisionGuard
from autolife_handeye_calibration_306.verification_trajectory import JointPath
from control_center_joint_control.joint_definitions import GROUPS
from .core import stages
from .bottle import horizontal_basis
from .motion import configure_grasp_path,configure_reset_path
from .joint_limits import load_arm_policy,check_pose
from .carry import carry_goal
from .object_extent import directional_yaws,projected_width,bbox


def approach_candidates(yaws=None):
    """Keep small yaw ahead of large yaw, trying arm/path alternatives first."""
    if yaws is None:yaws=[yaw for magnitude in range(0,61,5) for yaw in ([0.] if magnitude==0 else [float(-magnitude),float(magnitude)])]
    for yaw in yaws:
        for flexible in [False,True]:
            for profile in ['direct','joint_direct','joint_elbow_positive','joint_elbow_negative',
                            'elbow_positive','elbow_negative','outward_4','outward_8']:
                yield yaw,flexible,profile


def reference_position(start,end,t,offset):
    return start+t*(end-start)+np.sin(np.pi*t)*offset


def distance_to_polyline(point,points):
    starts=points[:-1];lines=np.diff(points,axis=0)
    t=np.clip(np.sum((point-starts)*lines,axis=1)/np.maximum(1e-12,np.sum(lines*lines,axis=1)),0.,1.)
    return float(np.min(np.linalg.norm(point-(starts+t[:,None]*lines),axis=1)))


def withdrawal_freedom(distance):
    """Keep 6 cm of straight back-out, then release orientation over 4 cm.

    This is an orientation envelope, not commanded rotation or a claim that
    the gripper has cleared an unknown shelf. Every accepted curve is checked.
    """
    return float(np.clip((distance-.06)/.04,0.,1.))


class Planner:
    def __init__(self,cfg):
        self.cfg=cfg;self.fk=Kinematics(cfg['urdf_path'])
        # No marker is attached in this task. Preserve a conservative tool envelope.
        self.guard=CollisionGuard({**cfg,'board_envelope_radius_m':.09})
        self.names=ARMS[cfg['arm']]
        limits={n:(lo,hi) for n,_,lo,hi in GROUPS[cfg['arm']+'_arm']['joints']}
        self.joint_limit_policy=load_arm_policy(cfg,self.names,limits)
        usable=self.joint_limit_policy['bounds_deg']
        self.bounds=np.deg2rad(np.array([(max(usable[n][0],np.rad2deg(self.guard.kin.lower[self.guard.kin.joint_q[n]])),
                            min(usable[n][1],np.rad2deg(self.guard.kin.upper[self.guard.kin.joint_q[n]]))) for n in self.names]).T)
        # Same configured URDF and torso-relative TCP, evaluated in compiled FK.
        # The previous Python FK reconstructed the chain for every numerical
        # Jacobian sample (thousands of times per candidate).
        self.pose_model=pin.buildModelFromUrdf(cfg['urdf_path'])
        self.pose_data=self.pose_model.createData()
        self.pose_q=pin.neutral(self.pose_model)
        self.pose_indices=[];self.pose_velocity_indices=[]
        for name in self.names:
            if not self.pose_model.existJointName(name):raise ValueError('URDF缺少关节 '+name)
            joint=self.pose_model.joints[self.pose_model.getJointId(name)]
            if joint.nq!=1:raise ValueError('抓取FK需要单自由度旋转关节 '+name)
            self.pose_indices.append(joint.idx_q)
            self.pose_velocity_indices.append(joint.idx_v)
        self.torso_frame=self.pose_model.getFrameId(TORSO)
        self.wrist_frame=self.pose_model.getFrameId(wrist_frame(cfg))
        if max(self.torso_frame,self.wrist_frame)>=self.pose_model.nframes:raise ValueError('URDF缺少抓取坐标系')

    def reset_plan(self,source):
        from autolife_handeye_calibration_306.core import NECK
        from .reset_recovery import recovery_target,check_recovery_geometry
        names=ARMS['left']+ARMS['right']+NECK;goal=dict(source)
        for group,ns in [('left_arm',ARMS['left']),('right_arm',ARMS['right']),('neck',NECK)]:
            goal.update(zip(ns,self.cfg['initial_pose_deg'][group]))
        # User empties jaws before reset. Opening runs concurrently; the tool
        # envelope covers the full jaw sweep rather than just the closed model.
        reset_source,changes=recovery_target(source,goal,self.guard.kin)
        checked=0;segments=[]
        if changes:
            recovery=JointPath(names,[source,reset_source])
            checked+=check_recovery_geometry(self.guard,source,recovery,changes)
            segments.append({'name':'颈部轻微越界，先向限位内回退','waypoints':[source,reset_source],
                             'neck_recovery':changes,'speed_scale':.1})
        path=configure_reset_path(JointPath(names,[reset_source,goal]),self.cfg)
        self.guard.cache.clear()
        for values in path.checks():
            self.guard.check({**source,**values});checked+=1
        segments.append({'name':'复位至初始姿态','waypoints':[reset_source,goal],'max_speed_deg_s':path.max_speed})
        return {'kind':'prepare','source_q':source,'end_q':goal,'checked_states':checked,
                'segments':segments,
                'open_both_grippers_deg':self.cfg['gripper_open_deg'],'urdf_sha256':self.fk.sha,
                'self_collision_checked':True,'table_height_checked':False,'config':dict(self.cfg),
                'created_monotonic':time.monotonic()}

    def pose(self,q,tcp):
        values=np.deg2rad([q[n] for n in self.names])
        if not np.isfinite(values).all():raise ValueError('关节反馈包含无效值')
        self.pose_q[self.pose_indices]=values
        pin.forwardKinematics(self.pose_model,self.pose_data,self.pose_q)
        root=pin.updateFramePlacement(self.pose_model,self.pose_data,self.torso_frame)
        tip=pin.updateFramePlacement(self.pose_model,self.pose_data,self.wrist_frame)
        T=root.inverse()*tip
        return T.translation+T.rotation@tcp,T.rotation.copy()

    def pose_jacobian(self,q,tcp):
        position,rotation=self.pose(q,tcp)
        pin.computeJointJacobians(self.pose_model,self.pose_data,self.pose_q)
        world=pin.getFrameJacobian(self.pose_model,self.pose_data,self.wrist_frame,
                                  pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:,self.pose_velocity_indices]
        root=self.pose_data.oMf[self.torso_frame];tip=self.pose_data.oMf[self.wrist_frame]
        offset=tip.rotation@tcp
        linear=root.rotation.T@(world[:3]-pin.skew(offset)@world[3:])
        angular=root.rotation.T@world[3:]
        return position,rotation,linear,angular

    def solve(self,source,tcp,point,rotation,flexible=False,limit_deg=None,posture_target=None,
              global_goal=False,retreat_freedom=0.,free_heading=False):
        seed=np.deg2rad([source[n] for n in self.names]);lo,hi=self.bounds
        if np.any(seed<lo-1e-9) or np.any(seed>hi+1e-9):
            raise ValueError('当前手臂已在厂商限位保护区附近，请先“复位并准备”再规划')
        seed=np.clip(seed,lo,hi)
        # Bound each local solve, rather than finding a remote redundant branch
        # then discarding it as a jump. Cartesian subdivision handles curvature.
        if not global_goal:
            step=np.deg2rad(8.);lo=np.maximum(lo,seed-step);hi=np.minimum(hi,seed+step)
        angle_limit=np.deg2rad(limit_deg if limit_deg is not None else self.cfg.get('orientation_relax_deg',30.) if flexible else 3.)
        # Aim inside the hard boundary, not exactly on it: numerical residuals
        # at 3.0001/5.0001 degrees otherwise discard an almost feasible route.
        preferred_limit=max(0.,angle_limit-np.deg2rad(.25))
        def qval(x):return {**source,**dict(zip(self.names,np.rad2deg(x)))}
        analytic=hasattr(self,'pose_model');cached={}
        def evaluate(x):
            if 'x' in cached and np.array_equal(x,cached['x']):return cached['residual']
            if analytic:p,R,Jp,Jw=self.pose_jacobian(qval(x),tcp)
            else:p,R=self.pose(qval(x),tcp)
            error=Rotation.from_matrix(rotation.T@R).as_rotvec()
            # Soft orientation preference, hard acceptance envelope. Position
            # accuracy is never relaxed to manufacture a successful grasp.
            posture=[] if posture_target is None else .004*(x-posture_target)
            residual=np.r_[p-point,(.01 if flexible else .10)*error,
                           max(0.,np.linalg.norm(error)-preferred_limit),.001*(x-seed),posture]
            if analytic:
                Jr=pin.Jlog3(rotation.T@R)@R.T@Jw
                norm=np.linalg.norm(error)
                barrier=error@Jr/norm if norm>preferred_limit and norm>1e-12 else np.zeros(len(seed))
                jac=np.vstack([Jp,(.01 if flexible else .10)*Jr,barrier,.001*np.eye(len(seed))])
                if posture_target is not None:jac=np.vstack([jac,.004*np.eye(len(seed))])
                cached['jac']=jac
            if retreat_freedom>0 or free_heading:
                # Placement frees heading only (5-degree upright envelope);
                # grasp/withdrawal retain their existing acceptance limits.
                # Keep position primary. Prefer upright carrying but let the
                # solver choose heading, rather than replaying 12 guessed yaws.
                up=np.asarray(self.cfg['up_torso']);target_up=rotation.T@up;actual_up=R.T@up
                tilt_error=actual_up-target_up;tilt_norm=np.linalg.norm(tilt_error)
                tilt_limit=2*np.sin(np.deg2rad(4.75 if free_heading else 2.75+retreat_freedom*12.)/2)
                heading_limit=np.pi if free_heading else np.deg2rad(2.75+retreat_freedom*87.)
                residual=np.r_[p-point,.01*tilt_error,max(0.,tilt_norm-tilt_limit),
                               .001*error,max(0.,np.linalg.norm(error)-heading_limit),.001*(x-seed)]
                if analytic:
                    Jtilt=pin.skew(actual_up)@R.T@Jw
                    tilt_bar=tilt_error@Jtilt/tilt_norm if tilt_norm>tilt_limit else np.zeros(len(seed))
                    heading_bar=error@Jr/norm if norm>heading_limit else np.zeros(len(seed))
                    cached['jac']=np.vstack([Jp,.01*Jtilt,tilt_bar,.001*Jr,heading_bar,.001*np.eye(len(seed))])
            cached.update(x=x.copy(),residual=residual)
            return residual
        def jacobian(x):evaluate(x);return cached['jac']
        result=least_squares(evaluate,seed,jac=jacobian if analytic else '2-point',
                            bounds=(lo,hi),max_nfev=70,ftol=1e-7,xtol=1e-7,gtol=1e-7)
        q=qval(result.x);p,R=self.pose(q,tcp)
        error=np.linalg.norm(p-point);angle=Rotation.from_matrix(rotation.T@R).magnitude()
        if retreat_freedom>0 or free_heading:
            up=np.asarray(self.cfg['up_torso'])
            tilt=np.arccos(np.clip(np.dot(R.T@up,rotation.T@up),-1.,1.))
            if (error>.004 or tilt>np.deg2rad(5. if free_heading else 3.+retreat_freedom*12.) or
                    angle>(np.pi if free_heading else np.deg2rad(3.+retreat_freedom*87.))+1e-6):
                raise IKFailure(('放置' if free_heading else '收回')+f'位置误差{error*1000:.1f}mm，倾斜{np.rad2deg(tilt):.1f}°')
            return q
        if error>.004 or angle>angle_limit+1e-6:raise IKFailure(f'位置误差{error*1000:.1f}mm，方向偏差{np.rad2deg(angle):.1f}°')
        return q

    def check(self,q,tcp,carrying,product):
        try:check_pose(self.joint_limit_policy,q)
        except ValueError as exc:raise IKFailure(str(exc)) from exc
        try:self.guard.check(q)
        except ValueError as exc:
            if '碰撞' in str(exc) or '包络' in str(exc):raise CollisionFailure(str(exc)) from exc
            raise
        if carrying:
            # Conservative held-bottle envelope against the body/other arm.
            # Use the collision-model frame (not torso-frame coordinates).
            import pinocchio as pin
            import hppfcl as fcl
            from autolife_handeye_calibration_306.core import BODY,NECK
            groups={'leg_waist':[q[n] for n in BODY],'neck':[q[n] for n in NECK],
                    'left_arm':[q[n] for n in ARMS['left']],'right_arm':[q[n] for n in ARMS['right']]}
            kin=self.guard.kin;raw=kin.q_from_feedback(groups,clip=False)
            pin.forwardKinematics(kin.model,kin.data,raw);pin.updateFramePlacements(kin.model,kin.data)
            pin.updateGeometryPlacements(kin.model,kin.data,kin.geom_model,kin.geom_data,raw)
            wrist=kin.data.oMf[self.guard.frame];center=wrist.translation+wrist.rotation@tcp
            collision_height=product.get('collision_height_m',product['height_m'])
            radius=(float(np.hypot(collision_height*.5,product['diameter_m']*.5))+.005
                    if product.get('geometry','cylinder')=='cylinder' else
                    float(np.linalg.norm([collision_height,product['diameter_m'],product['depth_m']]))*.5+.005)
            if 'bbox_axes_torso' in product:radius=float(np.linalg.norm(bbox(product)[1]))*.5+.005
            if product.get('geometry')=='bag_edge':
                # User selects the upper rim; bag hangs below it. Shift the
                # carried envelope with gravity, not camera/torso +Z.
                torso_rotation=kin.data.oMf[kin.model.getFrameId(TORSO)].rotation
                center=center-(torso_rotation@np.asarray(self.cfg['up_torso']))*product['height_m']*.5
                radius+=.015 # allowance for a soft bag; not a swinging-body model
            bottle=fcl.Sphere(radius);transform=fcl.Transform3f(np.eye(3),center)
            for i,obj in enumerate(kin.geom_model.geometryObjects):
                if any(obj.name.startswith(prefix) for prefix in self.guard.skip):continue
                t=kin.geom_data.oMg[i];lo,hi=self.guard.boxes[i];local=t.rotation.T@(center-t.translation)
                if np.linalg.norm(local-np.clip(local,lo,hi))>radius+.005:continue
                distance=fcl.distance(bottle,transform,obj.geometry,fcl.Transform3f(t.rotation,t.translation),fcl.DistanceRequest(),fcl.DistanceResult())
                if distance<.005:raise CollisionFailure('携带饮料可能碰到机器人：'+obj.name)

    def plan(self,source,tcp,center,product):
        started=time.monotonic();failures=[];late_failures=set()
        self.candidate_failures=failures
        yaws=None
        if 'bbox_axes_torso' in product:
            yaws=list(directional_yaws(product,self.cfg))
            if not yaws:return self._local_contact_plan(source,tcp,center,product,started)
        if self.cfg.get('fixed_carry_enabled',False):
            # A bad common endpoint cannot be repaired by searching grasp yaws.
            self.check(carry_goal(self.cfg,source,self.names),tcp,True,product)
        for yaw_offset,flexible,profile in approach_candidates(yaws):
            # These alternatives only change the pregrasp. If the direct route
            # already reached it and failed later, prioritize another direction
            # over five near-identical replays of the same carry endpoint.
            if (yaw_offset,flexible) in late_failures and not profile.startswith('joint_'):continue
            self._candidate_stage=0
            try:
                plan=self._route(source,tcp,center,product,flexible,started,yaw_offset,profile)
                plan['candidate_failures']=failures
                return plan
            except (IKFailure,CollisionFailure) as exc:
                failures.append({'yaw_offset_deg':yaw_offset,'flexible':flexible,
                                 'pregrasp_profile':profile,'stage_index':self._candidate_stage,'reason':str(exc)})
                if self._candidate_stage>0 and not profile.startswith('joint_'):late_failures.add((yaw_offset,flexible))
        if 'measured_footprint_torso' in product:
            # A feasible whole-width heading does NOT imply a feasible whole
            # centre path. Retain full held geometry and try measured local
            # contacts after those centre routes have actually failed.
            plan=self._local_contact_plan(source,tcp,center,product,started)
            plan['candidate_failures']=failures+plan.get('candidate_failures',[])
            plan['contact_fallback_reason']='center_paths_exhausted'
            return plan
        raise ValueError('多方向路径均未通过：'+failure_summary(failures)+'；请调整物品与机器人的相对位置后重新预览')

    def _local_contact_plan(self,source,tcp,center,product,started):
        from .local_contact import candidates
        choices=candidates(product,self.cfg)
        if not choices:raise ValueError('整体及前端局部夹取区域都超过夹爪开度，或前端有效插入深度不足')
        toolR=Rotation.from_quat(self.cfg['T_wrist_tool_quat_xyzw']).as_matrix()
        _,R0=self.pose(source,tcp);failures=[]
        prior_failures=list(getattr(self,'candidate_failures',[]))
        self.candidate_failures=prior_failures
        for choice in choices:
            f,s,u=horizontal_basis(self.cfg['up_torso'],np.deg2rad(self.cfg['approach_yaw_deg']+choice['yaw']))
            base=np.column_stack([f,s,u]);rotations=[base@toolR.T,base@np.diag([1.,-1.,-1.])@toolR.T]
            desired=min(rotations,key=lambda r:Rotation.from_matrix(R0.T@r).magnitude())
            offset=desired.T@(-choice['contact']);virtual=np.asarray(tcp)+offset
            local={**product,'object_center_offset_wrist':offset.tolist(),
                   'local_contact_torso':choice['contact'].tolist(),
                   'local_required_opening_m':choice['required_opening_m']}
            try:
                if self.cfg.get('fixed_carry_enabled',False):self.check(carry_goal(self.cfg,source,self.names),virtual,True,local)
            except CollisionFailure as exc:
                failures.append(dict(stage_index=4,reason=str(exc)));continue
            # Local geometry is verified with actual IK orientation at every
            # inserted point; use strict orientation rather than 30deg flex.
            for yaw,flexible,profile in approach_candidates([choice['yaw']]):
                if flexible:continue
                self._candidate_stage=0
                try:
                    plan=self._route(source,virtual,center,local,False,started,yaw,profile)
                    plan.update(physical_tcp=np.asarray(tcp).tolist(),grasp_contact_mode='front_local',
                                contact_center=(np.asarray(center)+choice['contact']).tolist(),candidate_failures=failures)
                    return plan
                except (IKFailure,CollisionFailure) as exc:
                    failure=dict(yaw_offset_deg=yaw,pregrasp_profile=profile,
                                 stage_index=self._candidate_stage,reason=str(exc),contact_mode='front_local')
                    failures.append(failure);self.candidate_failures.append(failure)
        raise ValueError('前端局部开度已通过，但完整路径未通过：'+failure_summary(failures))

    def _carry_segment(self,source,tcp,product,grasp_rotation,started,clearance_waypoints=None):
        if clearance_waypoints is not None:
            try:
                return self._checked_carry_segment(source,tcp,product,grasp_rotation,started,[source])
            except (IKFailure,CollisionFailure):
                # Only a checked clearance curve may replace a blocked direct
                # return; both execute continuously with one final arrival.
                pass
        return self._checked_carry_segment(source,tcp,product,grasp_rotation,started,clearance_waypoints)

    def _checked_carry_segment(self,source,tcp,product,grasp_rotation,started,clearance_waypoints=None):
        """Return ONLY the holding arm, optionally blending the lift into it.

        No reset API, jaw opening, new IK endpoint or unchecked shortcut. The
        interpolation is exactly the JointPath used by the executor.
        """
        goal=carry_goal(self.cfg,source,self.names)
        # Clearance is a bend in ONE checked curve, not a stop or a separate
        # retreat. Recheck the new PCHIP: appending the goal changes curvature.
        waypoints=[source,goal] if clearance_waypoints is None else [*clearance_waypoints,goal]
        path=configure_grasp_path(JointPath(self.names,waypoints),self.cfg)
        up=np.asarray(self.cfg['up_torso'],float)
        start_position,_=self.pose(source,tcp);end_position,_=self.pose(goal,tcp)
        floor=min(float(start_position@up),float(end_position@up))-.005
        def bottom(position,rotation):
            if 'bbox_axes_torso' in product:
                from .object_extent import bbox
                axes,extent=bbox(product)
                half=float(np.abs(up@(rotation@grasp_rotation.T@axes))@extent/2)
            elif 'height_m' in product:
                axis=rotation@grasp_rotation.T@up;cosine=float(np.clip(abs(axis@up),0,1))
                half=float(product['height_m'])*cosine/2+float(product['diameter_m'])/2*np.sqrt(max(0.,1-cosine*cosine))
            else:half=0.
            return float(position@up)-half
        initial_bottom=bottom(start_position,self.pose(source,tcp)[1])
        max_tilt=0.;checked=0
        for values in path.checks():
            if time.monotonic()-started>45:raise ValueError('固定持物路径检查超时，请重新规划')
            q={**source,**values};self.check(q,tcp,True,product);checked+=1
            position,rotation=self.pose(q,tcp)
            if float(position@up)<floor:raise IKFailure('持物路径向下绕行，拒绝低位收回')
            if clearance_waypoints is not None:
                delta=position-start_position
                horizontal=delta-up*float(delta@up)
                # Without a sensed support surface, do not descend through
                # the original object footprint while taking it off the table.
                if np.linalg.norm(horizontal)<.10 and float(delta@up)<-.002:
                    raise IKFailure('起瓶区域内路径下降，拒绝压向支撑面')
                if np.linalg.norm(horizontal)<.10 and bottom(position,rotation)<initial_bottom-.002:
                    raise IKFailure('起瓶区域内物品底部下降，改用连续抬离路径')
            tilt=float(np.rad2deg(np.arccos(np.clip(np.dot(rotation.T@up,grasp_rotation.T@up),-1.,1.))))
            if tilt>float(self.cfg.get('carry_max_tilt_deg',45.)):
                raise IKFailure('持物路径倾斜过大，请调整抓取方向')
            max_tilt=max(max_tilt,tilt)
        return {'name':'直接收回复位姿态' if len(waypoints)==2 else '连续起瓶并收回复位姿态',
                'phase':'carry','waypoints':waypoints,
                'return_strategy':'direct' if len(waypoints)==2 else 'continuous_clearance',
                'tcp_goal':end_position.tolist(),'max_speed_deg_s':path.max_speed,
                'max_tilt_deg':round(max_tilt,2)},checked

    def _route(self,source,tcp,center,product,flexible,started,yaw_offset=0.,profile='direct'):
        cfg=self.cfg;self.guard.cache.clear()
        yaw=np.deg2rad(cfg['approach_yaw_deg']+yaw_offset)
        width=product['depth_m'] if product.get('geometry')=='bag_edge' else product['diameter_m']
        local_contact='object_center_offset_wrist' in product
        if local_contact:width=product['local_required_opening_m']-.010
        elif 'bbox_axes_torso' in product:width=projected_width(product,horizontal_basis(cfg['up_torso'],yaw)[1])
        if width+.010>cfg['gripper_max_width_m']:raise ValueError('夹取部位过宽，没有足够夹取余量')
        route_cfg=cfg
        if 'bbox_axes_torso' in product:
            forward=horizontal_basis(cfg['up_torso'],yaw)[0]
            route_cfg={**cfg,'pregrasp_m':max(cfg['pregrasp_m'],projected_width(product,forward)/2+.05)}
        st=stages(center,yaw,route_cfg)
        continuous=cfg.get('fixed_carry_enabled',False) and cfg.get('carry_return_mode')=='continuous_reset'
        if continuous:st=st[:3]  # No mandatory 20 cm withdrawal detour.
        toolR=Rotation.from_quat(cfg['T_wrist_tool_quat_xyzw']).as_matrix()
        def check_aperture(q):
            if 'bbox_axes_torso' not in product:return
            position,rotation=self.pose(q,tcp)
            if local_contact:
                from .local_contact import required_opening
                physical=position-rotation@np.asarray(product['object_center_offset_wrist'])
                # Before reaching the object, no target material intersects
                # the jaws; once entering, validate the entire swept prefix.
                R=rotation@toolR;contact=physical-center
                required=required_opening(product,contact,R)
                if required>cfg['gripper_max_width_m']:raise IKFailure(f'局部伸入路径需要{required*1000:.0f}mm开度')
                return
            if np.linalg.norm(position-center)>projected_width(product,horizontal_basis(cfg['up_torso'],yaw)[0])/2+.03:return
            required=projected_width(product,(rotation@toolR)[:,1])
            if required+.010>cfg['gripper_max_width_m']:
                raise IKFailure(f'实测IK夹爪方向需要{required*1000:.0f}mm开度，改试其他方向')
        previous=dict(source);p0,R0=self.pose(source,tcp)
        forward,side,up=horizontal_basis(cfg['up_torso'],yaw)
        base=np.column_stack([forward,side,up])
        # Parallel jaws permit a half-turn about the approach axis. Choose the
        # nearest equivalent orientation, not an unnecessary wrist half-turn.
        rotations=[base@toolR.T,base@np.diag([1.,-1.,-1.])@toolR.T]
        desired=min(rotations,key=lambda r:Rotation.from_matrix(R0.T@r).magnitude())
        # Bias redundancy only, never teleport to a different IK seed. Local
        # 8-degree steps and every collision/interpolation check still apply.
        posture=None
        if profile.startswith('elbow_'):
            posture=np.deg2rad([source[n] for n in self.names])
            posture[1]+=np.deg2rad(20. if profile=='elbow_positive' else -20.)
            posture=np.clip(posture,*self.bounds)
        if np.linalg.norm(st[0][1]-p0)>.65:raise ValueError('预抓取点距所选夹爪超过65cm，请先调整起始姿态')
        route=[];pending_checks=[];checked=0;nominal=desired.copy();max_deviation=0.
        max_grasp_deviation=0.;max_withdraw_tilt=0.
        for index,(label,endpoint) in enumerate(st):
            self._candidate_stage=index
            if time.monotonic()-started>45:raise ValueError('规划超时，请调整物体位置后重试')
            if index==0 and profile.startswith('joint_'):
                # A straight TCP line from folded reset may hit the torso even
                # when the pregrasp itself is reachable. Solve that endpoint,
                # then validate a continuous joint-space transit from measured
                # source; this is not a direct jump to a remote IK seed.
                seed=dict(source)
                if profile!='joint_direct':
                    values=np.deg2rad([source[n] for n in self.names])
                    values[1]+=np.deg2rad(20. if profile=='joint_elbow_positive' else -20.)
                    seed.update(zip(self.names,np.rad2deg(np.clip(values,*self.bounds))))
                goal=self.solve(seed,tcp,endpoint,desired,flexible,global_goal=True)
                count=max(2,int(np.ceil(max(abs(goal[n]-source[n]) for n in self.names)/2.))+1)
                waypoints=[{**source,**{n:source[n]+t*(goal[n]-source[n]) for n in self.names}}
                           for t in np.linspace(0,1,count)]
                for q in waypoints:self.check(q,tcp,False,product)
                checked+=len(waypoints)
                path=configure_grasp_path(JointPath(self.names,waypoints),cfg)
                reference=np.array([self.pose(q,tcp)[0] for q in waypoints])
                pending_checks.append((path,reference,desired.copy()))
                route.append({'name':label,'phase':'pregrasp','waypoints':waypoints,
                              'tcp_goal':endpoint.tolist(),'max_speed_deg_s':path.max_speed})
                previous=waypoints[-1]
                continue
            current,R=self.pose(previous,tcp);distance=np.linalg.norm(endpoint-current)
            offset=np.zeros(3)
            if index==0 and profile.startswith('outward_'):
                outward=np.cross(up,forward)*(1. if cfg['arm']=='left' else -1.)
                offset=outward*(.04 if profile=='outward_4' else .08)+up*.02
            angle=Rotation.from_matrix(R.T@desired).magnitude()
            count=max(2,int(np.ceil((distance+np.pi*np.linalg.norm(offset))/.008))+1,
                      int(np.ceil(angle/np.deg2rad(2)))+1)
            reference=np.array([reference_position(current,endpoint,t,offset) for t in np.linspace(0,1,count)])
            rotations=Slerp([0.,1.],Rotation.from_matrix([R,desired]))
            reference_rotation=lambda t,slerp=rotations:slerp(t).as_matrix()
            waypoints=[dict(previous)]
            # If a small Cartesian interval is ill-conditioned, subdivide it.
            # No unchecked shortcut, approximate goal or joint jump is accepted.
            def advance(qprev,start_t,end_t,depth=0):
                if time.monotonic()-started>45:raise ValueError('规划超时，请缩短接近距离或调整起始姿态')
                freedom=withdrawal_freedom(end_t*distance) if index==3 else 0.
                try:q=self.solve(qprev,tcp,reference_position(current,endpoint,end_t,offset),
                                 reference_rotation(end_t),flexible,
                                 5. if flexible and index>=2 else None,posture if index==0 else None,
                                 retreat_freedom=freedom)
                except IKFailure:
                    if depth>=3:raise
                    mid=(start_t+end_t)/2
                    middle=advance(qprev,start_t,mid,depth+1)
                    return middle+advance(middle[-1],mid,end_t,depth+1)
                return [q]
            previous_t=0.
            for t in np.linspace(0,1,count)[1:]:
                chunk=advance(previous,previous_t,t);previous_t=t
                for q in chunk:
                    if index==1:check_aperture(q)
                    self.check(q,tcp,index>=2,product);checked+=1
                    waypoints.append(q)
                previous=chunk[-1]
            path=configure_grasp_path(JointPath(self.names,waypoints),cfg)
            pending_checks.append((path,reference,desired.copy()))
            route.append({'name':label,'phase':['pregrasp','grasp','lift','withdraw'][index],
                          'waypoints':waypoints,'tcp_goal':endpoint.tolist(),
                          'max_speed_deg_s':min(path.max_speed,cfg['max_joint_speed_deg_s'])})
            # After grasp, anchor orientation there (at most 5deg compliance in
            # flexible mode), never straighten the bottle toward nominal again.
            if index==1:desired=self.pose(previous,tcp)[1]
        # Do dense interpolation checks only after the Cartesian IK stages succeed.
        # Nothing is executable until EVERY check below has passed. Failed
        # carry endpoints no longer waste dense checks of the preceding stages.
        for index,(path,reference,desired) in enumerate(pending_checks):
            self._candidate_stage=index
            for v in path.checks():
                if time.monotonic()-started>45:raise ValueError('规划超时，请调整物体位置后重试')
                q={**source,**v};self.check(q,tcp,index>=2,product);checked+=1
                if index==1:check_aperture(q)
                p,Rcheck=self.pose(q,tcp)
                if distance_to_polyline(p,reference)>.010:raise IKFailure('平滑轨迹偏离预设路径')
                deviation=Rotation.from_matrix(desired.T@Rcheck).magnitude()
                permitted=np.deg2rad(cfg.get('orientation_relax_deg',30.) if flexible and index==1 else 5. if flexible and index>=2 else 4.)
                if index==3:
                    line=reference[-1]-reference[0];length=np.linalg.norm(line)
                    progress=float(np.clip(np.dot(p-reference[0],line)/max(length,1e-9),0.,length))
                    freedom=withdrawal_freedom(progress)
                    permitted=np.deg2rad(4.+87.*freedom)
                    tilt=float(np.arccos(np.clip(np.dot(Rcheck.T@up,desired.T@up),-1.,1.)))
                    if tilt>np.deg2rad(3.2+freedom*12.):raise IKFailure('平滑收回倾斜超出允许范围')
                    max_withdraw_tilt=max(max_withdraw_tilt,float(np.rad2deg(tilt)))
                if index>0 and deviation>permitted:raise IKFailure('平滑轨迹方向超出允许范围')
                if index>0:max_deviation=max(max_deviation,float(np.rad2deg(Rotation.from_matrix(nominal.T@Rcheck).magnitude())))
                if index==1:max_grasp_deviation=max(max_grasp_deviation,float(np.rad2deg(deviation)))
        carry_metadata={}
        if cfg.get('fixed_carry_enabled',False):
            self._candidate_stage=4
            grasp_rotation=self.pose(route[1]['waypoints'][-1],tcp)[1]
            clearance=route.pop()['waypoints'] if continuous else None
            segment,count=self._carry_segment(clearance[0] if continuous else previous,
                tcp,product,grasp_rotation,started,clearance)
            route.append(segment);checked+=count
            target=carry_goal(cfg,source,self.names)
            carry_metadata={'carry_protocol_version':2 if continuous else 1,
                            'carry_pose_deg':[target[n] for n in self.names]}
        grasp_q=next(s for s in route if s.get('phase')=='grasp')['waypoints'][-1]
        if local_contact:
            from .local_contact import required_opening
            position,rotation=self.pose(grasp_q,tcp)
            contact=position-rotation@np.asarray(product['object_center_offset_wrist'])-center
            width=max(.001,required_opening(product,contact,rotation@toolR)-.014)
        elif 'bbox_axes_torso' in product:
            width=projected_width(product,(self.pose(grasp_q,tcp)[1]@toolR)[:,1])
        return {'source_q':source,'segments':route,'tcp':np.asarray(tcp).tolist(),'center':np.asarray(center).tolist(),
                'grasp_width_m':float(width),
                'product':{k:v for k,v in product.items() if k not in ('features','examples')},
                'checked_states':checked,'seconds':round(time.monotonic()-started,2),
                'self_collision_checked':True,'table_height_checked':False,
                'support_required':False,'up_torso':up.tolist(),'geometry_version':2,
                'joint_limit_policy':self.joint_limit_policy,
                'orientation_mode':'flexible' if flexible else 'horizontal',
                'approach_yaw_deg':float(cfg['approach_yaw_deg']+yaw_offset),
                'pregrasp_profile':profile,
                'withdraw_orientation_mode':'fixed_carry' if carry_metadata else 'adaptive',
                **carry_metadata,
                'max_withdraw_tilt_deg':round(max_withdraw_tilt,2),
                'max_grasp_orientation_deviation_deg':round(max_grasp_deviation,2),
                'max_orientation_deviation_deg':round(max_deviation,2),
                'external_objects_checked':False,'execution_scope':'isolated_tabletop_only',
                'urdf_sha256':self.fk.sha}


class IKFailure(ValueError):
    """Kinematic failure of one candidate; all alternative paths are checked."""


class CollisionFailure(ValueError):
    """A rejected path, never an instruction to disable collision checking."""


def failure_summary(failures):
    """Report deepest completed stage, not the last unrelated search failure."""
    if not failures:return '未找到有可靠几何依据的接近路径'
    best=max(failures,key=lambda f:f.get('stage_index',0))
    phase={0:'接近物体',1:'伸入夹取',2:'抬起',3:'退离',4:'持物收回'}.get(best.get('stage_index',0),'规划')
    collisions=sum('碰撞' in f.get('reason','') for f in failures)
    return f"{phase}阶段：{best['reason']}（已检查{len(failures)}条候选，{collisions}条因碰撞拒绝）"
