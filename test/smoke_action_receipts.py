"""Exercise the existing all-joint interpolation with mocked motor feedback."""
import os
os.environ['ROS_DOMAIN_ID']='217';os.environ['QT_QPA_PLATFORM']='offscreen'
import sys, json, tempfile, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unittest.mock import patch
from copy import deepcopy
from PyQt5 import QtWidgets as Q
from std_msgs.msg import String
from autolife_shelf_grasp_306.app import Window
from control_center_joint_control.joint_definitions import ALL_JOINTS, JOINT_LIMITS, GROUPS, GROUP_ORDER, group_joint_names

app=Q.QApplication([]);root=Path(tempfile.mkdtemp());(root/'data').mkdir();run=root/'run';run.mkdir()
w=Window(root,run,start_worker=False);w.timer.stop();w.studio.timer.stop()
b=w.studio.ensure();w.studio.panel.spin_timer.stop()
q={n:(JOINT_LIMITS[n][0]+JOINT_LIMITS[n][1])/2 for n in ALL_JOINTS}
b._current=q.copy();b._controller_target=q.copy();b._preview=q.copy()
b._details={n:dict(communication_lost=False,speed=0) for n in q}
clock=[100.];b._feedback_monotonic=100.
b._hardware_enabled=b._hardware_ready=b._leg_waist_enabled=True
sent=[]
def motor(pub, payload):
    sent.append(payload)
    for g in ('neck','left_arm','right_arm','leg_waist'):
        key=GROUPS[g]['command_key']
        if key in payload:b._current.update(zip(group_joint_names(g),payload[key]))
    for side in ('Left','Right'):
        key=side.lower()+'_gripper_target_joints_position'
        if key in payload:b._current['Joint_'+side+'_Gripper']=payload[key][0]
b._publish_vendor=motor
target=q.copy();target['Joint_Left_Shoulder_Inner']+=5
record=dict(groups=GROUP_ORDER,loop_count=1,speed_percent=100,keyframes=[
    dict(joints_deg=target,transition_sec=1.,hold_sec=.2,easing='smooth'),
    dict(joints_deg=q,transition_sec=1.,hold_sec=0.,easing='smooth')])
with patch('control_center_joint_control.joint_bridge.time.monotonic',side_effect=lambda:clock[0]):
    b._on_action_command(String(data=json.dumps(dict(command='start',name='token:1',action=record))))
    assert b._action_state
    for i in range(35):
        clock[0]+=.1;b._feedback_monotonic=clock[0];b._advance_action()
    assert b.outcome==('token:1',True,'动作完成') and b._action_state is None
    assert abs(b._current['Joint_Left_Shoulder_Inner']-q['Joint_Left_Shoulder_Inner'])<1.
    b.outcome=None
    b._on_action_command(String(data=json.dumps(dict(command='start',name='token:2',action=record))))
    clock[0]+=1.;b._advance_action()
    assert b.outcome[0:2]==('token:2',False) and not b._hardware_enabled
assert sent
w.studio.shutdown();w.close()
print('ACTION_RECEIPTS_PASS: full-body interpolation, exact completion, feedback loss stops; motors mocked')
