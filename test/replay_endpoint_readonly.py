"""Replay the observed shoulder residual against saved model, no ROS."""
import json
from pathlib import Path
import numpy as np
from autolife_handeye_calibration_307.core import Kinematics,TORSO
from autolife_handeye_calibration_307.verification_core import wrist_frame,q_for_fk
from autolife_shelf_grasp_306.motion import transit_pose_ok,Arrival

p=Path('/tmp/shelf306_cu4jjp8w/plan_09ec0b31bb9747a38ddeeae84cd52bf4.json')
plan=json.loads(p.read_text());cfg=plan['config'];fk=Kinematics(cfg['urdf_path'])
goal=plan['segments'][0]['waypoints'][-1];actual=dict(goal)
actual['Joint_Left_Shoulder_Outer']-=.9593
a=fk.fk(TORSO,wrist_frame(cfg),q_for_fk(actual,cfg));b=fk.fk(TORSO,wrist_frame(cfg),q_for_fk(goal,cfg));tcp=np.array(plan['tcp'])
distance=np.linalg.norm(a[:3,3]+a[:3,:3]@tcp-b[:3,3]-b[:3,:3]@tcp)*1000
assert transit_pose_ok(a,b,tcp)
assert not Arrival(goal,.35).update(actual,0)
print('OBSERVED_ENDPOINT_REPLAY_OK',{'tcp_error_mm':distance,'pregrasp_accepted':True,'grasp_joint_tolerance_unchanged':.35})
