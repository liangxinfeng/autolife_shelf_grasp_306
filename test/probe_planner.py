"""Synthetic model-space preflight. Does not read or move the physical robot."""
import json
from pathlib import Path
import yaml
import numpy as np
from scipy.spatial.transform import Rotation
from autolife_shelf_grasp_306.planning import Planner
from autolife_handeye_calibration_306.core import ARMS,BODY,NECK

root=Path('/home/ubuntu/ros2_ws/src/autolife_shelf_grasp_306')
cfg=yaml.safe_load((root/'config/grasp.yaml').read_text())
planner=Planner(cfg)
q=dict(zip(BODY,[0,0,28.699,0]))
q.update(zip(NECK,[0,-20,0]));q.update(zip(ARMS['left'],[0,0,0,110,0,13,0]));q.update(zip(ARMS['right'],[0,0,0,-110,0,-13,0]))
q.update(Joint_Left_Gripper=10.,Joint_Right_Gripper=10.)
tcp=np.array([.22947,.00731,-.01594])
p,R=planner.pose(q,tcp)
print('initial_tcp',p.tolist(),'wrist_rpy',Rotation.from_matrix(R).as_euler('xyz',degrees=True).tolist())
planner.guard.check(q)
print('INITIAL_SELF_COLLISION_OK')
# Position+orientation IK is independently validated at an FK-generated point.
target=q.copy();target[ARMS['left'][0]]+=1.
point,rot=planner.pose(target,tcp)
fit=planner.solve(q,tcp,point,rot)
got,r=planner.pose(fit,tcp)
assert np.linalg.norm(got-point)<.004
assert Rotation.from_matrix(rot.T@r).magnitude()<np.deg2rad(3)
print('FIXED_POSE_IK_OK',float(np.linalg.norm(got-point)))
planner.cfg['up_torso']=[0,0,-1]
route=planner.plan(q,tcp,np.array([.6,-.3,-.08]),{'id':'synthetic','name':'仿真瓶','diameter_m':.07,'height_m':.22})
assert len(route['segments'])==4
print('FOUR_STAGE_ROUTE_OK',route['seconds'],route['checked_states'])
for kind,dimensions in [('solid',(.04,.09,.025)),('bag_edge',(.13,.20,.004))]:
    product=dict(id='synthetic',name=kind,geometry=kind,
                 diameter_m=dimensions[0],height_m=dimensions[1],depth_m=dimensions[2])
    route=planner.plan(q,tcp,np.array([.6,-.3,-.08]),product)
    assert len(route['segments'])==4
    print('OBJECT_ROUTE_OK',kind,route['seconds'],route['checked_states'])
reset=planner.reset_plan(q)
assert reset['end_q']['Joint_Neck_Pitch']==-20
assert reset['end_q']['Joint_Knee']==q['Joint_Knee']
assert reset['open_both_grippers_deg']==10
print('RESET_PREFLIGHT_OK',reset['checked_states'])
