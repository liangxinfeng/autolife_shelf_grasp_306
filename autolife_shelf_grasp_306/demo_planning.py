"""Plan a measured-table placement without commanding hardware."""
import time
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from autolife_handeye_calibration_306.core import TORSO
from autolife_handeye_calibration_306.verification_trajectory import JointPath
from .carry import carry_goal
from .motion import configure_grasp_path,cartesian_speed_profile
from .planning import IKFailure,CollisionFailure
from .demo_config import require_placeable,footprint_radius,carried_height
from .object_extent import bbox
from .placement_interior import InteriorPreference,interior_rank

# Observed obstacles already include 10mm depth padding. Keep 5mm additional
# exact-shape clearance, instead of stacking 25mm padding plus another 10mm.
CLUTTER_MARGIN_M=.005


def held_box_rotation(wrist_rotation,grasp_rotation,product):
    """Transport the complete measured solid envelope, not a fitted cylinder."""
    axes,extent=bbox(product)
    return wrist_rotation@grasp_rotation.T@axes,extent


def jaw_envelope_bounds(low,high,physical_tcp,max_width):
    """Audited wrist axes: X approach, Z opening, Y finger thickness.

    Preserve the complete model separately; this additional front envelope
    covers opening throughout release, including material beyond the gap.
    """
    low=np.array(low,float,copy=True);high=np.array(high,float,copy=True)
    physical_tcp=np.asarray(physical_tcp,float)
    if (low.shape!=(3,) or high.shape!=(3,) or physical_tcp.shape!=(3,) or
        not np.isfinite(np.r_[low,high,physical_tcp,max_width]).all() or
        np.any(high<=low) or max_width<=0):raise ValueError('夹爪包络参数无效')
    low[0]=min(max(low[0],physical_tcp[0]-.045),physical_tcp[0]-.001)
    high[0]=max(high[0],physical_tcp[0]+.045)
    half_open=max_width/2+.04
    low[2]=min(low[2],physical_tcp[2]-half_open)
    high[2]=max(high[2],physical_tcp[2]+half_open)
    return low-.001,high+.001


def placement_trials(targets,seed_count):
    """Fair, deterministic position/heading/branch coverage within a deadline."""
    yaws=(0,-5,5,-10,10,-15,15,-20,20,-25,25,-30,30,-35,35,-40,40)
    order=sorted((i+.45*j+.75*k,i,j,k) for i in range(len(targets))
                 for j in range(len(yaws)) for k in range(seed_count))
    return ((targets[i],yaws[j],k) for _,i,j,k in order)


def release_displacement(rotation,tool_quaternion,up,distance,lift):
    """Withdraw along the jaws, not fixed torso X; keep the full exit length."""
    up=np.asarray(up,float);norm=np.linalg.norm(up)
    if not np.isfinite(norm) or norm<1e-8:raise ValueError('放置退出方向无效')
    up=up/norm
    direction=rotation@Rotation.from_quat(tool_quaternion).as_matrix()[:,0]
    direction-=up*np.dot(direction,up);norm=np.linalg.norm(direction)
    if not np.isfinite(norm) or norm<.2:raise ValueError('夹爪方向不适合水平退出，请重新规划')
    if not np.isfinite([distance,lift]).all() or distance<=0 or lift<0:raise ValueError('放置退出距离无效')
    return -direction/norm*distance+up*lift


def check_release_departure(delta,rotation,initial_rotation,tool,up,retreat,height):
    withdrawn=-float(delta@tool[:,0]);risen=float(delta@up)
    if withdrawn<retreat-.005 and risen<height/2+.05:
        turn=Rotation.from_matrix(initial_rotation.T@rotation).magnitude()
        if float(delta@tool[:,0])>.002 or risen<-.002 or abs(float(delta@tool[:,1]))>.01 or turn>np.deg2rad(5):
            raise CollisionFailure(f'松爪后的直接收回尚未退出物品，改用连续退出路径（退出{withdrawn*1000:.0f}mm，侧移{abs(float(delta@tool[:,1]))*1000:.0f}mm，转角{np.rad2deg(turn):.1f}°）')


def box_separation(rotation,position,low,high,box_rotation,box_center,half_extent):
    """Lower bound on shape distance via its enclosing AABB in box axes."""
    transform=box_rotation.T@rotation
    center=box_rotation.T@(position-box_center)+transform@((low+high)*.5)
    extent=np.abs(transform)@((high-low)*.5)
    return float(np.linalg.norm(np.maximum(np.abs(center)-extent-half_extent,0.)))


def box_separations(rotation,position,low,high,rotations,centers,half_extents):
    """Vectorized same conservative bound, before exact geometry distances."""
    inverse=np.swapaxes(rotations,1,2)
    transform=inverse@rotation
    center=np.einsum('nij,nj->ni',inverse,position-centers)+np.einsum('nij,j->ni',transform,(low+high)*.5)
    extent=np.einsum('nij,j->ni',np.abs(transform),(high-low)*.5)
    return np.linalg.norm(np.maximum(np.abs(center)-extent-half_extents,0.),axis=1)


def geometry_box_separations(shape_rotations,shape_positions,lows,highs,rotations,centers,half_extents):
    """一次计算全部连杆×桌面/障碍的同一保守下界，精确距离检查保持不变。"""
    inverse=np.swapaxes(rotations,1,2)
    transform=inverse[None,:,:,:]@shape_rotations[:,None,:,:]
    center=np.einsum('tij,ntj->nti',inverse,shape_positions[:,None,:]-centers[None,:,:])
    center+=np.einsum('ntij,nj->nti',transform,(lows+highs)*.5)
    extent=np.einsum('ntij,nj->nti',np.abs(transform),(highs-lows)*.5)
    return np.linalg.norm(np.maximum(np.abs(center)-extent-half_extents[None,:,:],0.),axis=2)


def remove_robot_points(planner,q,tcp,product,points):
    """Exclude robot/held-bottle returns before environmental checks."""
    import pinocchio as pin
    from autolife_handeye_calibration_306.core import ARMS,BODY,NECK
    kin=planner.guard.kin
    raw=kin.q_from_feedback({'leg_waist':[q[n] for n in BODY],'neck':[q[n] for n in NECK],
        'left_arm':[q[n] for n in ARMS['left']],'right_arm':[q[n] for n in ARMS['right']]},clip=False)
    pin.forwardKinematics(kin.model,kin.data,raw);pin.updateFramePlacements(kin.model,kin.data)
    pin.updateGeometryPlacements(kin.model,kin.data,kin.geom_model,kin.geom_data,raw)
    root=kin.data.oMf[kin.model.getFrameId('Link_Zero_Point')]
    from .point_cloud_filter import outside_boxes
    world=points@root.rotation.T+root.translation
    keep=outside_boxes(world,kin.geom_data.oMg,planner.guard.boxes)
    wrist=kin.data.oMf[planner.guard.frame];center=wrist.translation+wrist.rotation@tcp
    radius=np.hypot(carried_height(product)/2,footprint_radius(product))+.025
    keep&=np.linalg.norm(world-center,axis=1)>radius
    return points[keep]


class TableGuard:
    def __init__(self,planner,table,product,tcp,grasp_rotation):
        import hppfcl as fcl
        self.fcl=fcl;self.p=planner;self.tcp=np.asarray(tcp);self.grasp_rotation=grasp_rotation
        self.product=product;lo=np.array(table['minimum']);hi=np.array(table['maximum'])
        self.support=None
        if 'footprint_support' in table:
            from .placement_support import unpack_support
            self.support=unpack_support(table['footprint_support'])
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
        # The URDF mesh is always checked below. Add a direction-aware envelope
        # for the opening fingers rather than treating jaw thickness as width.
        self.jaw_envelope=None
        for i,obj in enumerate(planner.guard.kin.geom_model.geometryObjects):
            if (obj.name=='Link_'+planner.cfg['arm'].title()+'_Wrist_Lower_to_Gripper_0' and
                obj.parentFrame==planner.guard.frame and
                np.allclose(obj.placement.rotation,np.eye(3)) and
                np.allclose(obj.placement.translation,0.) and
                np.allclose(np.abs(Rotation.from_quat(planner.cfg['T_wrist_tool_quat_xyzw']).as_matrix()),
                            [[1,0,0],[0,0,1],[0,1,0]],atol=1e-3)):
                physical=self.tcp-np.asarray(product.get('object_center_offset_wrist',[0.,0.,0.]))
                low,high=jaw_envelope_bounds(*planner.guard.boxes[i],physical,
                                            planner.cfg.get('gripper_max_width_m',.095))
                self.jaw_envelope=(fcl.Box(*map(float,high-low)),(low+high)/2)
                break
        self.complete_jaw=None
        if planner.cfg.get('geometry_source')=='urdf' and planner.cfg.get('urdf_path'):
            # 与低物体抓取共用完整手指开合几何，替代与实际工具坐标无关的45mm球。
            from .gripper_contact_geometry import gripper_geometry
            from scipy.spatial import ConvexHull
            vertices=gripper_geometry(planner)['vertices'];hull=ConvexHull(vertices)
            shape=fcl.BVHModelOBBRSS();shape.beginModel(len(hull.simplices),len(vertices))
            points=fcl.StdVec_Vec3s();triangles=fcl.StdVec_Triangle()
            for vertex in vertices:points.append(vertex)
            for a,b,c in hull.simplices:triangles.append(fcl.Triangle(int(a),int(b),int(c)))
            shape.addSubModel(points,triangles);shape.endModel()
            self.complete_jaw=(shape,(vertices.min(0),vertices.max(0)))
        self.obstacles=[]
        for box in table.get('obstacles',[]):
            low=np.array(box['minimum']);high=np.array(box['maximum'])
            self.obstacles.append((fcl.Box(*map(float,high-low)),(low+high)/2,(high-low)/2))
        self.geometry_indices=[i for i,obj in enumerate(planner.guard.kin.geom_model.geometryObjects)
            if getattr(planner,'placement_whole_body_check',False) or obj.name.startswith('Link_'+planner.cfg['arm'].title())]
        self.geometry_lows=np.array([planner.guard.boxes[i][0] for i in self.geometry_indices])
        self.geometry_highs=np.array([planner.guard.boxes[i][1] for i in self.geometry_indices])
        self.checked_cache=set()
        self._environment_key=None
        self._sphere_shape=fcl.Sphere(.045)
        self._distance_request=fcl.DistanceRequest()
        self._held_shape=(fcl.Box(*map(float,bbox(product)[1])) if 'bbox_axes_torso' in product
                          else fcl.Cylinder(footprint_radius(product),carried_height(product)))

    def check_support(self,q):
        if self.support is None:return
        position,rotation=self.p.pose(q,self.tcp)
        if not self.support_pose(position,rotation,q):
            raise CollisionFailure('物品实际底面超出已确认桌面支撑，改试其他落点或方向')

    def support_pose(self,position,rotation,q):
        if self.support is None:return True
        from .placement_support import supported,footprint_outline
        ground=(self.p.anchor if getattr(self.p,'anchor',None) is not None else self.p.fk.fk('Link_Zero_Point',TORSO,q))
        outline=footprint_outline(self.product,ground[:3,:3]@rotation@self.grasp_rotation.T)
        return supported(*self.support,ground[:3,:3]@position+ground[:3,3],outline)

    def check(self,q,carrying):
        # Planner.check already updates the same collision model placements.
        import pinocchio as pin
        from autolife_handeye_calibration_306.core import ARMS,BODY,NECK
        key=(carrying,tuple(sorted(q.items())))
        if key in self.checked_cache:return
        p=self.p;kin=p.guard.kin
        raw=kin.q_from_feedback({'leg_waist':[q[n] for n in BODY],'neck':[q[n] for n in NECK],
            'left_arm':[q[n] for n in ARMS['left']],'right_arm':[q[n] for n in ARMS['right']]},clip=False)
        pin.forwardKinematics(kin.model,kin.data,raw);pin.updateFramePlacements(kin.model,kin.data)
        pin.updateGeometryPlacements(kin.model,kin.data,kin.geom_model,kin.geom_data,raw)
        root=kin.data.oMf[kin.model.getFrameId('Link_Zero_Point')]
        environment_key=(root.rotation.tobytes(),root.translation.tobytes())
        if environment_key!=self._environment_key:
            # 同一规划内桌面/障碍不动；只有根坐标变换改变才重新构造环境数据。
            table_rotation=root.rotation@self.rotation;table_center=root.translation+root.rotation@self.center
            targets=[(self.box,table_rotation,table_center,self.half_extent,.001,'放置轨迹与已识别桌面相交')]
            targets.extend((shape,root.rotation,root.translation+root.rotation@center,half,CLUTTER_MARGIN_M,
                            f'放置轨迹接近桌面已有物品#{i+1}') for i,(shape,center,half) in enumerate(self.obstacles))
            rotations=np.array([t[1] for t in targets]);centers=np.array([t[2] for t in targets])
            extents=np.array([t[3] for t in targets]);margins=np.array([t[4] for t in targets])
            transforms=[self.fcl.Transform3f(t[1],t[2]) for t in targets]
            self._environment=(targets,rotations,centers,extents,margins,transforms)
            self._environment_key=environment_key
        targets,rotations,centers,extents,margins,transforms=self._environment
        def test(shape,rotation,position,bounds=None,label='夹爪/饮料',scope='all'):
            if bounds is None:
                shape.computeLocalAABB();bounds=(np.array(shape.aabb_local.min_),np.array(shape.aabb_local.max_))
            lower=box_separations(rotation,position,*bounds,rotations,centers,extents)
            indices=np.flatnonzero(lower<=margins+1e-9);self.broadphase_skips+=len(targets)-len(indices)
            for index in indices:
                if scope=='table' and index!=0:continue
                if scope=='clutter' and index==0:continue
                obstacle,rot,center,half,margin,message=targets[index]
                self.exact_distance_calls+=1
                distance=self.fcl.distance(shape,self.fcl.Transform3f(rotation,position),obstacle,
                    transforms[index],self._distance_request,self.fcl.DistanceResult())
                if distance<margin:raise CollisionFailure(f'{message}（{label}，间距{distance*1000:.1f}mm）')
        placements=[kin.geom_data.oMg[i] for i in self.geometry_indices]
        if placements:
            bounds=geometry_box_separations(np.array([t.rotation for t in placements]),
                np.array([t.translation for t in placements]),self.geometry_lows,self.geometry_highs,
                rotations,centers,extents)
            possible=np.flatnonzero(np.any(bounds<=margins[None,:]+1e-9,axis=1))
            self.broadphase_skips+=(len(placements)-len(possible))*len(targets)
            for k in possible:
                i=self.geometry_indices[k];obj=kin.geom_model.geometryObjects[i];t=placements[k]
                test(obj.geometry,t.rotation,t.translation,p.guard.boxes[i],obj.name)
        wrist=kin.data.oMf[p.guard.frame];center=wrist.translation+wrist.rotation@self.tcp
        physical_center=center-wrist.rotation@np.asarray(self.product.get('object_center_offset_wrist',[0.,0.,0.]))
        if self.complete_jaw is not None:
            shape,bounds=self.complete_jaw
            test(shape,wrist.rotation,wrist.translation,bounds,label='完整手指开合范围',scope='table')
            test(self._sphere_shape,np.eye(3),physical_center,label='夹爪障碍包络',scope='clutter')
        elif self.jaw_envelope is None:
            test(self._sphere_shape,np.eye(3),physical_center,label='夹爪保守包络')
        else:
            shape,offset=self.jaw_envelope
            # Correct thickness only for support-plane contact. Preserve the
            # previous clutter guard, in addition to the complete URDF meshes;
            # do not turn the full opening sweep into occupied solid space.
            test(shape,wrist.rotation,wrist.translation+wrist.rotation@offset,label='夹爪张开包络',scope='table')
            test(self._sphere_shape,np.eye(3),physical_center,label='夹爪障碍包络',scope='clutter')
        if carrying:
            if 'bbox_axes_torso' in self.product:
                rotation,extent=held_box_rotation(wrist.rotation,self.grasp_rotation,self.product)
                test(self._held_shape,rotation,center)
                self.checked_cache.add(key)
                return
            torso=kin.data.oMf[kin.model.getFrameId(TORSO)]
            up=np.asarray(p.cfg['up_torso']);_,R=p.pose(q,self.tcp)
            reference=p.anchor[:3,:3] if getattr(p,'anchor',None) is not None else torso.rotation
            axis=reference@(R@self.grasp_rotation.T@up);axis/=np.linalg.norm(axis)
            other=np.array([1.,0.,0.]) if abs(axis[0])<.9 else np.array([0.,1.,0.])
            x=np.cross(other,axis);x/=np.linalg.norm(x);y=np.cross(axis,x)
            test(self._held_shape,np.column_stack([x,y,axis]),center)
        self.checked_cache.add(key)


def placement_candidates(positions,current,limit=10,spacing=.045,clearance=None,cover_surface=False,edge_margins=None,preferred_margin=.08):
    """Spatially diverse, near-current candidates without a hard-coded XY goal."""
    positions=np.asarray(positions,float).reshape(-1,3)
    if not len(positions):return []
    # Input ordering retains local support clearance as a small preference.
    support_rank=np.arange(len(positions))/max(1,len(positions)-1)
    score=np.linalg.norm(positions-current,axis=1)+.025*support_rank
    if clearance is not None:
        clearance=np.asarray(clearance,float)
        if clearance.shape!=(len(positions),) or not np.isfinite(clearance).all():
            raise ValueError('放置候选空间数据无效')
        # Ranking only: narrow/borderline sites stay available as fallbacks.
        score+=.20*np.clip((.012-clearance)/.012,0.,1.)
    if edge_margins is not None:
        margins=np.asarray(edge_margins,float)
        if margins.shape!=(len(positions),) or not np.isfinite(margins).all():
            raise ValueError('桌边余量数据无效')
        # 仅改变排序：8cm舒适余量达标后仍优先近点；边缘可达点保留为备选。
        score+=4.*np.maximum(0.,preferred_margin-margins)
    order=np.argsort(score,kind='stable');selected=[]
    if cover_surface:
        # Keep the best local point, then cover separate tabletop regions.
        # A crowded near corner must not consume the entire endpoint budget.
        selected=[positions[order[0]]];remaining=list(map(int,order[1:]))
        while remaining and len(selected)<limit:
            distances=np.min(np.linalg.norm(positions[remaining,None,:]-np.asarray(selected)[None,:,:],axis=2),axis=1)
            quality=distances-.25*(score[remaining]-score[order[0]])
            index=int(np.argmax(quality))
            if distances[index]<min(.015,spacing):break
            selected.append(positions[remaining.pop(index)])
        return selected
    for separation in (spacing,min(.015,spacing)):
        for i in order:
            if all(np.linalg.norm(positions[i]-p)>separation for p in selected):selected.append(positions[i])
            if len(selected)>=limit:return selected
    return selected


def landing_candidate_clearance(positions,ground,obstacles):
    positions=np.asarray(positions,float).reshape(-1,3)
    if not obstacles:return np.full(len(positions),1.)
    points=positions@ground[:3,:3].T+ground[:3,3]
    low=np.asarray([b['minimum'] for b in obstacles],float)
    high=np.asarray([b['maximum'] for b in obstacles],float)
    distance=np.linalg.norm(np.maximum(np.maximum(low[None,:,:]-points[:,None,:],
                                points[:,None,:]-high[None,:,:]),0.),axis=2)
    return np.min(distance,axis=1)-(.045+CLUTTER_MARGIN_M)


def landing_candidate_mask(positions,ground,product,obstacles):
    """Necessary jaw clearance before IK, never a replacement for collision.

    The exact guard keeps a 45mm sphere at the physical jaw centre and 5mm
    obstacle clearance. Allow ALL possible object-centre offsets and the 4mm
    IK error here, so this filter only rejects impossible sphere placements.
    """
    allowance=.004+np.linalg.norm(product.get('object_center_offset_wrist',[0.,0.,0.]))
    return landing_candidate_clearance(positions,ground,obstacles)+allowance>=-1e-9


def upright_heading(actual,reference,up):
    """Preserve the held object's vertical axis; extract only solved yaw."""
    up=np.asarray(up,float);up=up/np.linalg.norm(up)
    axis=int(np.argmin(np.abs(reference.T@up)))
    a=reference[:,axis]-up*(reference[:,axis]@up);a/=np.linalg.norm(a)
    b=actual[:,axis]-up*(actual[:,axis]@up);b/=np.linalg.norm(b)
    yaw=float(np.rad2deg(np.arctan2(up@np.cross(a,b),a@b)))
    return Rotation.from_rotvec(up*np.deg2rad(yaw)).as_matrix()@reference,yaw


def rank_place_ik(planner,source,tcp,candidates,grasp_R,up,product,settings,table_guard,deadline,diagnostics,fallback_candidates=(),alternative_seeds=(),allow_detour=False,free_heading=False,interior=None,ground=None):
    """Bounded endpoint screening before any expensive smoothed path checks."""
    from_angles=np.array([source[n] for n in planner.names]);ranked=[];blocked=[]
    attempts=0;started=time.monotonic()
    # Alternative configurations are IK initial guesses, NEVER commanded waypoints.
    seeds=[source]+list(alternative_seeds)
    for batch,targets in enumerate((candidates,fallback_candidates)):
        # Reserve a small part of the unchanged budget for fine support pockets
        # when the primary set is entirely rejected.
        batch_budget=2.8 if batch==0 and len(fallback_candidates) else 3.5
        trials=list(placement_trials(targets,len(seeds)))
        # Keep reachable-near candidates ahead of far table centres.
        # Validate support at the actual solved landing, not the nominal seed.
        for target,yaw,seed_index in trials:
            if time.monotonic()>min(deadline,started+batch_budget):break
            attempts+=1
            desired=Rotation.from_rotvec(up*np.deg2rad(yaw)).as_matrix()@grasp_R
            screening_stage='above_ik'
            try:
                pre=target+settings['approach_height_m']*up
                above=planner.solve(seeds[seed_index],tcp,pre,desired,True,global_goal=True,limit_deg=5.,**({'free_heading':True} if free_heading else {}))
                if free_heading:
                    desired,yaw=upright_heading(planner.pose(above,tcp)[1],grasp_R,up)
                screening_stage='landing_ik'
                landing=planner.solve(above,tcp,target,desired,True,global_goal=True,limit_deg=5.)
                if hasattr(table_guard,'check_support'):table_guard.check_support(landing)
                edge={}
                if interior is not None:
                    margin=float(interior.margins([planner.pose(landing,tcp)[0]],ground)[0])
                    edge=dict(edge_margin_m=margin,edge_shortfall_m=float(interior.shortfalls(margin)))
                for screening_stage,q in (('above_collision',above),('landing_collision',landing)):
                    planner.check(q,tcp,True,product);table_guard.check(q,True)
                # Cheap rejection only; EVERY smoothed-curve state is still
                # checked below before a plan can leave this module.
                for t in np.linspace(0.,1.,13)[1:-1]:
                    screening_stage='approach_collision'
                    q={**source,**{n:source[n]+t*(above[n]-source[n]) for n in planner.names}}
                    planner.check(q,tcp,True,product);table_guard.check(q,True)
                a=np.array([above[n] for n in planner.names]);b=np.array([landing[n] for n in planner.names])
                score=float(np.linalg.norm(a-from_angles)+.5*np.linalg.norm(b-a)+.15*abs(yaw))
                if not any(np.linalg.norm(target-r['target'])<1e-5 and
                           max(abs(above[n]-r['above'][n]) for n in planner.names)<1. for r in ranked):
                    ranked.append(dict(target=target,yaw=yaw,rotation=desired,above=above,score=score,seed_index=seed_index,**edge))
                if len(ranked)>=6:break
            except (IKFailure,CollisionFailure,ValueError) as exc:
                diagnostics.append(dict(phase='endpoint_ik',screening_stage=screening_stage,target=target.tolist(),yaw=yaw,seed_index=seed_index,error=str(exc)))
                if allow_detour and screening_stage=='approach_collision':
                    # Endpoints passed IK/support/collision. Only the direct
                    # connector failed; require a checked detour below.
                    score=float(np.linalg.norm(np.array([above[n] for n in planner.names])-from_angles)+.15*abs(yaw))
                    if not any(np.linalg.norm(target-r['target'])<.03 for r in blocked):
                        blocked.append(dict(target=target,yaw=yaw,rotation=desired,above=above,score=score,
                                            seed_index=seed_index,needs_detour=True,**edge))
        if ranked or time.monotonic()>min(deadline,started+3.5):break
    diagnostics.append(dict(phase='ik_screen_summary',attempts=attempts,feasible=len(ranked),
                            detour_endpoints=len(blocked),
                            seconds=round(time.monotonic()-started,3)))
    return (sorted(ranked,key=interior_rank)+sorted(blocked,key=interior_rank))[:6]


def checked_transfer_approach(planner,source,goal,tcp,up,check):
    """Bounded side/lift connectors; full dense validation is still required."""
    begin,R=planner.pose(source,tcp);end,end_R=planner.pose(goal,tcp)
    failures=[]
    forward=np.array([1.,0.,0.]);forward-=up*(forward@up);forward/=np.linalg.norm(forward)
    side=np.cross(up,forward);side*=1 if (end-begin)@side>=0 else -1
    offsets=[side*.08,-side*.08,up*.08,up*.16,-forward*.06+up*.08,-side*.16,side*.16,up*.22,-side*.08+up*.14,side*.08+up*.14]
    for offset in offsets:
        try:
            raised=planner.solve(source,tcp,begin+offset,R,True,global_goal=True,limit_deg=15.)
            check(raised,True)
        except (IKFailure,CollisionFailure,ValueError) as exc:
            failures.append(str(exc));continue
        for overpass in (False,True):
            try:
                waypoints=[source,raised]
                if overpass:
                    over=planner.solve(raised,tcp,end+up*.08,end_R,True,global_goal=True,limit_deg=10.)
                    waypoints.append(over)
                waypoints.append(goal)
                curve=JointPath(planner.names,waypoints)
                for values in curve.checks():
                    check({**source,**values},True)
                return waypoints
            except (IKFailure,CollisionFailure,ValueError) as exc:failures.append(str(exc))
    raise CollisionFailure('直达及侧向/抬高绕行均未通过：'+(failures[-1] if failures else '无可用过渡位姿'))


def plan_place(planner,source,tcp,grasp,points,table,settings):
    planner.guard.begin_fixed_neck(source)
    product=grasp['product'];require_placeable(product)
    from .demo_config import require_placement_bottom
    require_placement_bottom(grasp)
    cfg=planner.cfg;up=np.asarray(cfg['up_torso']);tcp=np.asarray(tcp)
    ground=planner.fk.fk('Link_Zero_Point',TORSO,source);inv=np.linalg.inv(ground)
    centers=np.asarray(points)@inv[:3,:3].T+inv[:3,3]
    # A vertical cylinder's lower rim reaches a sloping plane before its centre.
    # Compensate that geometry, rather than relaxing the table collision check.
    rim_clearance=footprint_radius(product)*(np.linalg.norm(table['coef'][:2])+np.sin(np.deg2rad(5)))
    centers+=up*(carried_height(product)/2+settings['place_clearance_m']+rim_clearance)
    # Keep calibrated grasp-point correction; never move the detected table.
    centers+=np.asarray(cfg.get('target_offset_torso_mm',[0,0,0]))/1000
    # Do not reject near-centre candidates with a hand-written side limit.
    # Actual reachability is determined by bounded IK and full path checks.
    clearance=landing_candidate_clearance(centers,ground,table.get('obstacles',[]))
    allowance=.004+np.linalg.norm(product.get('object_center_offset_wrist',[0.,0.,0.]))
    possible=clearance+allowance>=-1e-9
    positions=centers[possible];clearance=clearance[possible]
    selection_diagnostic=dict(phase='landing_prefilter',support_candidates=len(centers),
                              jaw_clearance_candidates=len(positions))
    if not len(positions):
        planner.place_diagnostics=[selection_diagnostic]
        raise ValueError('桌面空位可支撑物品，但夹爪伸入空间不足；请调整观察位置或清理周边物品')
    grasp_R=(planner.placement_grasp_rotation(grasp) if hasattr(planner,'placement_grasp_rotation') else planner.pose(grasp['segments'][1]['waypoints'][-1],tcp)[1])
    table_guard=TableGuard(planner,table,product,tcp,grasp_R)
    interior=InteriorPreference(table,footprint_radius(product),settings.get('edge_margin_m',.08))
    margins=interior.margins(positions,ground)
    selection_diagnostic.update(preferred_edge_margin_m=interior.preferred,
                                maximum_available_edge_margin_m=float(np.max(margins)))
    current,_=planner.pose(source,tcp)
    preference=dict(edge_margins=margins,preferred_margin=interior.preferred)
    candidates=placement_candidates(positions,current,clearance=clearance,**preference)
    fine=placement_candidates(positions,current,limit=50,spacing=.015,clearance=clearance,cover_surface=True,**preference)
    fine=[p for p in fine if all(np.linalg.norm(p-c)>.008 for c in candidates)]
    started=time.monotonic();deadline=started+8.;last='没有可用放置路径';planner.place_diagnostics=[selection_diagnostic];best_progress=-1
    carry_waypoints=grasp['segments'][-1]['waypoints']
    alternatives=[]
    if len(carry_waypoints)>=2:
        alternatives=[{**source,**{n:carry_waypoints[-2][n] for n in planner.names}}]
    ranked=rank_place_ik(planner,source,tcp,candidates,grasp_R,up,product,settings,table_guard,deadline,planner.place_diagnostics,fine,alternatives,allow_detour=True,free_heading=True,interior=interior,ground=ground)
    if not ranked:raise ValueError('桌面有空位，但候选放置姿态未通过可达性或障碍间距检查')
    checked_cache=set()
    approach_cache={}
    # Only screened poses get complete interpolation and collision validation.
    # Endpoint success alone never authorizes motion.
    # Try the higher clearance exit first. A lower exit remains available for
    # scenes with overhead constraints; both receive identical full validation.
    for choice,exit_lift in ((c,h) for c in ranked for h in (None,.08,.03)):
        target=choice['target'];yaw=choice['yaw']
        try:
            phase='place_approach';progress=0
            if time.monotonic()>deadline:raise TimeoutError('放置规划预算用尽，已保持饮料')
            place_R=choice['rotation']
            route=[];previous=dict(source);checked=0
            def check(q,carrying):
                nonlocal checked
                if time.monotonic()>deadline:raise TimeoutError('放置规划预算用尽，已保持饮料')
                key=(carrying,tuple(sorted(q.items())))
                if key not in checked_cache:
                    planner.check(q,tcp,carrying,product);table_guard.check(q,carrying)
                    checked_cache.add(key)
                checked+=1
            pre=target+settings['approach_height_m']*up
            goal=choice['above']
            check(goal,True)
            progress=1
            key=id(choice)
            if key not in approach_cache:
                try:
                    approach_cache[key]=(checked_transfer_approach(planner,source,goal,tcp,up,check)
                                         if choice.get('needs_detour') else [source,goal])
                except (IKFailure,CollisionFailure,ValueError) as exc:approach_cache[key]=exc
            if isinstance(approach_cache[key],Exception):raise approach_cache[key]
            stages=[('移至桌面上方','place_approach',approach_cache[key],True)]
            previous=goal
            exit_offset=release_displacement(place_R,cfg['T_wrist_tool_quat_xyzw'],up,
                                               settings['release_retreat_m']+(.03 if exit_lift is not None else 0.),exit_lift or 0.)
            motions=[
                ('平稳放置饮料','place_lower',target,True),
                ('张爪后退离','place_release_retreat',target+exit_offset,False)]
            if exit_lift is None:motions=motions[:1]
            for name,phase,end,carrying in motions:
                begin,R=planner.pose(previous,tcp)
                motion_R=place_R
                if phase=='place_release_retreat':
                    # Preserve the actual solved landing attitude. Returning to
                    # nominal orientation here can twist the open jaws twice.
                    motion_R=R
                    end=begin+release_displacement(R,cfg['T_wrist_tool_quat_xyzw'],up,
                                                  settings['release_retreat_m']+.03,exit_lift)
                count=max(2,int(np.ceil(np.linalg.norm(end-begin)/.008))+1)
                waypoints=[previous]
                for t in np.linspace(0,1,count)[1:]:
                    previous=planner.solve(previous,tcp,begin+(end-begin)*t,motion_R,True,
                                           limit_deg=3. if phase=='place_release_retreat' else 5.)
                    # Reject a blocked descent before spending time checking
                    # the long approach/return curves. Successful paths still
                    # receive EVERY original 0.1-degree smoothed-curve check.
                    check(previous,carrying)
                    waypoints.append(previous)
                stages.append((name,phase,waypoints,carrying))
                if phase=='place_lower':table_guard.check_support(previous)
                progress+=1
            end=planner.carry_target(previous) if hasattr(planner,'placement_grasp_rotation') else carry_goal(cfg,previous,planner.names)
            return_points=[previous,end] if exit_lift is None else [*stages.pop()[2],end]
            stages.append(('直接收回复位姿态' if exit_lift is None else '连续退出并收回复位姿态','place_return',return_points,False))
            release_start,release_R=planner.pose(return_points[0],tcp)
            tool=release_R@Rotation.from_quat(cfg['T_wrist_tool_quat_xyzw']).as_matrix()
            # Reject a bad direct return before dense-checking approach/lower.
            # This is only a fast rejection pass: the accepted curve still
            # receives every original 0.1-degree check below.
            phase='place_return'
            trial=JointPath(planner.names,return_points)
            for s in np.unique(np.r_[trial.s,np.linspace(0,trial.length,max(2,int(np.ceil(trial.length))+1))]):
                q={**source,**dict(zip(planner.names,map(float,trial.at(s))))}
                pos,R=planner.pose(q,tcp)
                check_release_departure(pos-release_start,R,release_R,tool,up,settings['release_retreat_m'],carried_height(product))
                check(q,False)
            for name,phase,waypoints,carrying in stages:
                path=configure_grasp_path(JointPath(planner.names,waypoints),cfg)
                tcp_samples=[]
                for values in path.checks():
                    q={**source,**values}
                    # 原厂执行模式在 finalize_place 对最终线性插值逐0.1度检查，
                    # 不再先对不会直接执行的PCHIP曲线重复做同一组密集碰撞计算。
                    if not hasattr(planner,'finalize_place'):check(q,carrying)
                    pos,R=planner.pose(q,tcp)
                    if phase=='place_return':
                        # Until the open fingers have cleared the placed object,
                        # do not swing sideways, twist or advance into it.
                        delta=pos-release_start
                        check_release_departure(delta,R,release_R,tool,up,settings['release_retreat_m'],carried_height(product))
                    tcp_samples.append(pos)
                    if carrying:
                        tilt=np.rad2deg(np.arccos(np.clip(np.dot(R.T@up,grasp_R.T@up),-1,1)))
                        if tilt>45:raise IKFailure('放置过程中饮料倾斜过大')
                cartesian_speed=settings['place_speed_m_s'] if phase=='place_lower' else .06 if phase=='place_release_retreat' else .18 if phase=='place_return' else .12
                samples=np.unique(np.r_[path.s,np.linspace(0,path.length,max(2,int(np.ceil(path.length/.1))+1))])
                tcp_samples=np.asarray(tcp_samples)
                limits=np.full(len(samples),cartesian_speed)
                transfer_from=path.length
                if phase=='place_lower':
                    # Same checked curve; retain original slow speed for the
                    # final 40mm. Brake in advance, not at an extra waypoint.
                    indices=np.flatnonzero(np.linalg.norm(tcp_samples-tcp_samples[-1],axis=1)<=.04)
                    switch=float(samples[indices[0]])
                    limits[samples<switch]=min(.06,2*cartesian_speed)
                elif phase in ('place_release_retreat','place_return'):
                    # Do not speed up the initial withdrawal beside the object.
                    near=np.flatnonzero(np.linalg.norm(tcp_samples-tcp_samples[0],axis=1)<.04)
                    switch=float(samples[min(int(near[-1])+1,len(samples)-1)])
                    limits[:]=.06
                    limits[samples>=switch]=.18 if phase=='place_return' else .12
                    delta=tcp_samples-release_start
                    clear=((-delta@tool[:,0]>=settings['release_retreat_m']-.005)|
                           (delta@up>=carried_height(product)/2+.05))
                    uncleared=np.flatnonzero(~clear)
                    transfer_from=float(samples[min(int(uncleared[-1])+1,len(samples)-1)]) if len(uncleared) else 0.
                if getattr(planner,'placement_whole_body_check',False):
                    # 原厂整段定时会逐处限制关节速度/曲率加速度，不再叠加旧推进器
                    # 的全段最慢关节上限；末端接近桌面与退离的局部限速仍保留。
                    path.max_speed=float(cfg.get('max_joint_speed_deg_s',120.))
                profile=cartesian_speed_profile(path,samples,tcp_samples,limits)
                speed_cap=max(v for _,v in profile)
                route.append({'name':name,'phase':phase,'waypoints':waypoints,'path_speed_cap':speed_cap,
                              'path_speed_profile':profile,
                              'transfer_from_s':transfer_from,
                              'tcp_goal':planner.pose(waypoints[-1],tcp)[0].tolist()})
            result=dict(kind='place',place_protocol_version=3,source_q=source,segments=route,tcp=tcp.tolist(),
                product=product,center=target.tolist(),up_torso=up.tolist(),geometry_version=2,
                self_collision_checked=True,table_height_checked=True,checked_states=checked,
                joint_limit_policy=planner.joint_limit_policy,urdf_sha256=planner.fk.sha,
                table=table,placement_search='endpoint_screen_then_full_path',
                ik_screened=len(ranked),selected_yaw_deg=yaw,selected_seed_index=choice['seed_index'],
                preferred_edge_margin_m=interior.preferred,
                selected_edge_margin_m=float(interior.margins([release_start],ground)[0]),
                exit_lift_m=exit_lift,seconds=round(time.monotonic()-started,2))
            if hasattr(planner,'finalize_place'):
                phase='native_path_check'
                result=planner.finalize_place(result,source,tcp,grasp,table,settings)
                result['seconds']=round(time.monotonic()-started,2)
            return result
        except (IKFailure,CollisionFailure,ValueError) as exc:
            planner.place_diagnostics.append(dict(target=target.tolist(),yaw=yaw,phase=phase,
                seed_index=choice['seed_index'],exit_lift_m=exit_lift,error=str(exc)))
            if progress>best_progress:best_progress=progress;last=str(exc)
        except TimeoutError:
            planner.place_diagnostics.append(dict(target=target.tolist(),yaw=yaw,phase=phase,
                error='规划预算用尽',elapsed_s=round(time.monotonic()-started,3),checked_states=checked))
            raise
    raise ValueError('已找到桌面空位，但当前姿态没有可行放置路径：'+last)
