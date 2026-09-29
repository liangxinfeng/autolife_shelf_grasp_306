"""Plan a measured-table placement without commanding hardware."""
import time
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from autolife_handeye_calibration_307.core import TORSO
from autolife_handeye_calibration_307.verification_trajectory import JointPath
from .carry import carry_goal
from .motion import configure_grasp_path
from .planning import IKFailure,CollisionFailure
from .demo_config import require_drink


def box_separation(rotation,position,low,high,box_rotation,box_center,half_extent):
    """Lower bound on shape distance via its enclosing AABB in box axes."""
    transform=box_rotation.T@rotation
    center=box_rotation.T@(position-box_center)+transform@((low+high)*.5)
    extent=np.abs(transform)@((high-low)*.5)
    return float(np.linalg.norm(np.maximum(np.abs(center)-extent-half_extent,0.)))


def remove_robot_points(planner,q,tcp,product,points):
    """Exclude robot/held-bottle returns before environmental checks."""
    import pinocchio as pin
    from autolife_handeye_calibration_307.core import ARMS,BODY,NECK
    kin=planner.guard.kin
    raw=kin.q_from_feedback({'leg_waist':[q[n] for n in BODY],'neck':[q[n] for n in NECK],
        'left_arm':[q[n] for n in ARMS['left']],'right_arm':[q[n] for n in ARMS['right']]},clip=False)
    pin.forwardKinematics(kin.model,kin.data,raw);pin.updateFramePlacements(kin.model,kin.data)
    pin.updateGeometryPlacements(kin.model,kin.data,kin.geom_model,kin.geom_data,raw)
    root=kin.data.oMf[kin.model.getFrameId('Link_Zero_Point')]
    world=points@root.rotation.T+root.translation;keep=np.ones(len(points),bool)
    for i,t in enumerate(kin.geom_data.oMg):
        lo,hi=planner.guard.boxes[i];local=(world-t.translation)@t.rotation
        keep&=np.linalg.norm(local-np.clip(local,lo,hi),axis=1)>.015
    wrist=kin.data.oMf[planner.guard.frame];center=wrist.translation+wrist.rotation@tcp
    radius=np.hypot(product['height_m']/2,product['diameter_m']/2)+.025
    keep&=np.linalg.norm(world-center,axis=1)>radius
    return points[keep]


class TableGuard:
    def __init__(self,planner,table,product,tcp,grasp_rotation):
        import hppfcl as fcl
        self.fcl=fcl;self.p=planner;self.tcp=np.asarray(tcp);self.grasp_rotation=grasp_rotation
        self.product=product;lo=np.array(table['minimum']);hi=np.array(table['maximum'])
        a,b,c=table['coef'];normal=np.array([-a,-b,1.]);normal/=np.linalg.norm(normal)
        x=np.array([1.,0.,a]);x/=np.linalg.norm(x);y=np.cross(normal,x)
        self.rotation=np.column_stack([x,y,normal]);self.center=(lo+hi)/2
        self.center[2]=a*self.center[0]+b*self.center[1]+c
        corners=np.array([[xx,yy,a*xx+b*yy+c] for xx in (lo[0],hi[0]) for yy in (lo[1],hi[1])])
        extent=np.ptp((corners-self.center)@self.rotation,axis=0)
        self.center-=normal*.02
        self.box=fcl.Box(float(extent[0]),float(extent[1]),.04)
        self.half_extent=np.array([extent[0]/2,extent[1]/2,.02])
        self.exact_distance_calls=0;self.broadphase_skips=0
        self.obstacles=[]
        for box in table.get('obstacles',[]):
            low=np.array(box['minimum']);high=np.array(box['maximum'])
            self.obstacles.append((fcl.Box(*map(float,high-low)),(low+high)/2,(high-low)/2))

    def check(self,q,carrying):
        # Planner.check already updates the same collision model placements.
        import pinocchio as pin
        from autolife_handeye_calibration_307.core import ARMS,BODY,NECK
        p=self.p;kin=p.guard.kin
        raw=kin.q_from_feedback({'leg_waist':[q[n] for n in BODY],'neck':[q[n] for n in NECK],
            'left_arm':[q[n] for n in ARMS['left']],'right_arm':[q[n] for n in ARMS['right']]},clip=False)
        pin.forwardKinematics(kin.model,kin.data,raw);pin.updateFramePlacements(kin.model,kin.data)
        pin.updateGeometryPlacements(kin.model,kin.data,kin.geom_model,kin.geom_data,raw)
        root=kin.data.oMf[kin.model.getFrameId('Link_Zero_Point')]
        table_rotation=root.rotation@self.rotation;table_center=root.translation+root.rotation@self.center
        targets=[(self.box,table_rotation,table_center,self.half_extent,.001,'放置轨迹与已识别桌面相交')]
        targets.extend((shape,root.rotation,root.translation+root.rotation@center,half,.01,
                        f'放置轨迹接近桌面已有物品#{i+1}') for i,(shape,center,half) in enumerate(self.obstacles))
        def test(shape,rotation,position,bounds=None,label='夹爪/饮料'):
            if bounds is None:
                shape.computeLocalAABB();bounds=(np.array(shape.aabb_local.min_),np.array(shape.aabb_local.max_))
            for obstacle,rot,center,half,margin,message in targets:
                if box_separation(rotation,position,*bounds,rot,center,half)>margin+1e-9:
                    self.broadphase_skips+=1;continue
                self.exact_distance_calls+=1
                distance=self.fcl.distance(shape,self.fcl.Transform3f(rotation,position),obstacle,
                    self.fcl.Transform3f(rot,center),self.fcl.DistanceRequest(),self.fcl.DistanceResult())
                if distance<margin:raise CollisionFailure(f'{message}（{label}，间距{distance*1000:.1f}mm）')
        for i,obj in enumerate(kin.geom_model.geometryObjects):
            if obj.name.startswith('Link_Left'):
                t=kin.geom_data.oMg[i];test(obj.geometry,t.rotation,t.translation,p.guard.boxes[i],obj.name)
        wrist=kin.data.oMf[p.guard.frame];center=wrist.translation+wrist.rotation@self.tcp
        test(self.fcl.Sphere(.045),np.eye(3),center)  # jaw/tool clearance even while opening
        if carrying:
            torso=kin.data.oMf[kin.model.getFrameId(TORSO)]
            up=np.asarray(p.cfg['up_torso']);_,R=p.pose(q,self.tcp)
            axis=torso.rotation@(R@self.grasp_rotation.T@up);axis/=np.linalg.norm(axis)
            other=np.array([1.,0.,0.]) if abs(axis[0])<.9 else np.array([0.,1.,0.])
            x=np.cross(other,axis);x/=np.linalg.norm(x);y=np.cross(axis,x)
            test(self.fcl.Cylinder(self.product['diameter_m']/2,self.product['height_m']),np.column_stack([x,y,axis]),center)


def plan_place(planner,source,tcp,grasp,points,table,settings):
    product=grasp['product'];require_drink(product)
    if grasp.get('geometry',{}).get('image_cropped'):
        raise ValueError('本轮瓶子未完整入镜，瓶底到夹取点未知；不能自动放置')
    cfg=planner.cfg;up=np.asarray(cfg['up_torso']);tcp=np.asarray(tcp)
    ground=planner.fk.fk('Link_Zero_Point',TORSO,source);inv=np.linalg.inv(ground)
    centers=np.asarray(points)@inv[:3,:3].T+inv[:3,3]
    # A vertical cylinder's lower rim reaches a sloping plane before its centre.
    # Compensate that geometry, rather than relaxing the table collision check.
    rim_clearance=product['diameter_m']/2*(np.linalg.norm(table['coef'][:2])+np.sin(np.deg2rad(5)))
    centers+=up*(product['height_m']/2+settings['place_clearance_m']+rim_clearance)
    # Keep calibrated grasp-point correction; never move the detected table.
    centers+=np.asarray(cfg.get('target_offset_torso_mm',[0,0,0]))/1000
    # Do not reject near-centre candidates with a hand-written side limit.
    # Actual reachability is determined by bounded IK and full path checks.
    positions=centers
    if not len(positions):raise ValueError('未找到连续可见的桌面支撑区域，请调整观察位置')
    grasp_R=planner.pose(grasp['segments'][1]['waypoints'][-1],tcp)[1]
    table_guard=TableGuard(planner,table,product,tcp,grasp_R)
    # Prefer the left-hand front, then spatially diverse alternatives, at most 6.
    order=np.argsort(np.linalg.norm(positions[:,:2]-np.array([.53,-.18]),axis=1));candidates=[]
    for i in order:
        if all(np.linalg.norm(positions[i]-v)>.025 for v in candidates):candidates.append(positions[i])
        if len(candidates)>=6:break
    # Narrow valid strips can have a useful 5--10 mm neighbouring placement.
    # Keep bounded nearby alternatives instead of discarding the whole strip.
    for i in order:
        if all(np.linalg.norm(positions[i]-v)>.006 for v in candidates):candidates.append(positions[i])
        if len(candidates)>=18:break
    started=time.monotonic();last='没有可用放置路径';planner.place_diagnostics=[];best_progress=-1
    checked_cache=set()
    # Bottle heading is irrelevant on a clear table; keep it upright but allow
    # small heading changes before declaring a reachable centre unreachable.
    for target,yaw in ((t,y) for y in (0,-15,15,-25,25,-35,35,-45,45) for t in candidates):
        try:
            phase='place_approach';progress=0
            if time.monotonic()-started>15:raise TimeoutError('放置规划超时，已保持饮料')
            place_R=Rotation.from_rotvec(up*np.deg2rad(yaw)).as_matrix()@grasp_R
            route=[];previous=dict(source);checked=0
            def check(q,carrying):
                nonlocal checked
                if time.monotonic()-started>15:raise TimeoutError('放置规划超时，已保持饮料')
                key=(carrying,tuple(sorted(q.items())))
                if key not in checked_cache:
                    planner.check(q,tcp,carrying,product);table_guard.check(q,carrying)
                    checked_cache.add(key)
                checked+=1
            pre=target+settings['approach_height_m']*up
            goal=planner.solve(source,tcp,pre,place_R,True,global_goal=True,limit_deg=5.)
            check(goal,True)
            progress=1
            stages=[('移至桌面上方','place_approach',[source,goal],True)]
            previous=goal
            for name,phase,end,carrying in [
                ('平稳放置饮料','place_lower',target,True),
                ('张爪后退离','place_release_retreat',target-np.array([settings['release_retreat_m'],0,0])+up*.03,False)]:
                begin,R=planner.pose(previous,tcp);count=max(2,int(np.ceil(np.linalg.norm(end-begin)/.008))+1)
                waypoints=[previous]
                for t in np.linspace(0,1,count)[1:]:
                    previous=planner.solve(previous,tcp,begin+(end-begin)*t,place_R,True,limit_deg=5.)
                    # Reject a blocked descent before spending time checking
                    # the long approach/return curves. Successful paths still
                    # receive EVERY original 0.1-degree smoothed-curve check.
                    check(previous,carrying)
                    waypoints.append(previous)
                stages.append((name,phase,waypoints,carrying))
                progress+=1
            end=carry_goal(cfg,previous,planner.names)
            stages.append(('空手收回','place_return',[previous,end],False))
            for name,phase,waypoints,carrying in stages:
                path=configure_grasp_path(JointPath(planner.names,waypoints),cfg)
                previous_sample=None;meters_per_degree=0.
                for values in path.checks():
                    q={**source,**values};check(q,carrying)
                    pos,R=planner.pose(q,tcp);angles=np.array([q[n] for n in planner.names])
                    if previous_sample is not None:
                        d=float(np.max(np.abs(angles-previous_sample[0])))
                        if d>1e-9:meters_per_degree=max(meters_per_degree,float(np.linalg.norm(pos-previous_sample[1]))/d)
                    previous_sample=(angles,pos)
                    if carrying:
                        tilt=np.rad2deg(np.arccos(np.clip(np.dot(R.T@up,grasp_R.T@up),-1,1)))
                        if tilt>45:raise IKFailure('放置过程中饮料倾斜过大')
                cartesian_speed=settings['place_speed_m_s'] if phase=='place_lower' else .06 if phase=='place_release_retreat' else .12
                speed_cap=min(path.max_speed,cartesian_speed/max(meters_per_degree*1.25,1e-6))
                route.append({'name':name,'phase':phase,'waypoints':waypoints,'path_speed_cap':speed_cap,
                              'tcp_goal':planner.pose(waypoints[-1],tcp)[0].tolist()})
            return dict(kind='place',place_protocol_version=1,source_q=source,segments=route,tcp=tcp.tolist(),
                product=product,center=target.tolist(),up_torso=up.tolist(),geometry_version=2,
                self_collision_checked=True,table_height_checked=True,checked_states=checked,
                joint_limit_policy=planner.joint_limit_policy,urdf_sha256=planner.fk.sha,
                table=table,seconds=round(time.monotonic()-started,2))
        except (IKFailure,CollisionFailure,ValueError) as exc:
            planner.place_diagnostics.append(dict(target=target.tolist(),yaw=yaw,phase=phase,error=str(exc)))
            if progress>best_progress:best_progress=progress;last=str(exc)
        except TimeoutError:
            planner.place_diagnostics.append(dict(target=target.tolist(),yaw=yaw,phase=phase,
                error='规划预算用尽',elapsed_s=round(time.monotonic()-started,3),checked_states=checked))
            raise
    raise ValueError('已找到桌面空位，但当前姿态没有可行放置路径：'+last)
