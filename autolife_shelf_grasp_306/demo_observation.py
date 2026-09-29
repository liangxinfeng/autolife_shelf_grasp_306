"""Bounded held-bottle observation motion; no gripper commands or hardware I/O."""
import time
import cv2
import numpy as np
from autolife_handeye_calibration_307.core import TORSO
from autolife_handeye_calibration_307.verification_trajectory import JointPath
from .motion import configure_grasp_path
from .demo_planning import TableGuard
from .planning import IKFailure,CollisionFailure


def coverage(center,product,camera,roi):
    """Projected conservative bottle/tool sphere coverage of visible table ROI."""
    point=np.linalg.inv(camera['T'])@np.r_[center,1.]
    radius=np.hypot(product['height_m']/2,product['diameter_m']/2)+.045
    if point[2]<=radius:return float('inf')
    uv=cv2.projectPoints(point[:3].reshape(1,3),np.zeros(3),np.zeros(3),camera['K'],camera['D'])[0][0,0]
    px=max(camera['K'][0,0],camera['K'][1,1])*radius/(point[2]-radius)
    x0,y0,x1,y1=roi
    return max(0.,min(x1,uv[0]+px)-max(x0,uv[0]-px))*max(0.,min(y1,uv[1]+px)-max(y0,uv[1]-px))


def plan_observation(planner,source,tcp,grasp,table,camera,roi):
    """Lift 6--12 cm, optionally outward <=8 cm, only if visibility improves."""
    tcp=np.asarray(tcp);cfg=planner.cfg;product=grasp['product'];up=np.asarray(cfg['up_torso'])
    ground=planner.fk.fk('Link_Zero_Point',TORSO,source)
    start,R=planner.pose(source,tcp);grasp_R=planner.pose(grasp['segments'][1]['waypoints'][-1],tcp)[1]
    guard=TableGuard(planner,table,product,tcp,grasp_R)
    before=coverage(ground[:3,:3]@start+ground[:3,3],product,camera,roi)
    # Torso-side direction is derived from the actual left-hand position.
    lateral=np.array([0.,-1.,0.]);lateral-=up*np.dot(lateral,up);lateral/=np.linalg.norm(lateral)
    choices=[]
    for height in (.08,.12,.06):
        for outward in (0.,.04,.08):
            target=start+height*up+outward*lateral
            after=coverage(ground[:3,:3]@target+ground[:3,3],product,camera,roi)
            if after<before*.90:choices.append((after,height,outward,target))
    if not choices:raise ValueError('抬手不能可靠改善桌面视野，请调整前台位置或头部观察角度')
    began=time.monotonic();last='未找到安全观察姿态'
    for after,height,outward,target in sorted(choices,key=lambda v:(v[1],v[0])):
        try:
            previous=dict(source);waypoints=[previous];checked=0
            for t in np.linspace(0,1,17)[1:]:
                if time.monotonic()-began>8:raise TimeoutError('观察姿态规划超时，保持饮料')
                previous=planner.solve(previous,tcp,start+(target-start)*t,R,True,limit_deg=5.)
                waypoints.append(previous)
            path=configure_grasp_path(JointPath(planner.names,waypoints),cfg)
            for values in path.checks():
                if time.monotonic()-began>8:raise TimeoutError('观察姿态检查超时，保持饮料')
                q={**source,**values};planner.check(q,tcp,True,product);guard.check(q,True);checked+=1
            return dict(kind='observe',place_protocol_version=1,observation_protocol_version=1,
                source_q=source,tcp=tcp.tolist(),product=product,up_torso=up.tolist(),geometry_version=2,
                self_collision_checked=True,table_height_checked=True,table=table,checked_states=checked,
                joint_limit_policy=planner.joint_limit_policy,urdf_sha256=planner.fk.sha,
                held_gripper_deg=float(source['Joint_Left_Gripper']),
                visibility_before=before,visibility_after=after,lift_m=height,outward_m=outward,
                segments=[dict(name='保持夹紧，轻抬饮料避开桌面视野',phase='observe_lift',waypoints=waypoints,
                               path_speed_cap=min(path.max_speed,8.))])
        except (IKFailure,CollisionFailure,ValueError) as exc:last=str(exc)
    raise ValueError('观察姿态未通过路径检查：'+last)
