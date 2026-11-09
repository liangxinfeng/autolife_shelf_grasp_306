"""Real controller code with synthetic feedback and all motor writes replaced."""
import os,json,tempfile
os.environ['ROS_DOMAIN_ID']='87';os.environ['QT_QPA_PLATFORM']='offscreen'
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PyQt5 import QtCore as C,QtWidgets as Q
from std_msgs.msg import String
from test_teaching import pose,draft
from control_center_joint_control.joint_definitions import *
from autolife_shelf_grasp_306.action_studio import ActionExecutor
from autolife_shelf_grasp_306.teaching_motion import compile_action,stage_action
from autolife_shelf_grasp_306.teaching_trajectory import build

app=Q.QApplication([]);root=Path(tempfile.mkdtemp());(root/'run').mkdir()
def panel(node):
    w=Q.QWidget();w.stop_action_preview=lambda **kw:None;w.preview_timer=C.QTimer();w.spin_timer=C.QTimer()
    w._pending_live_groups=set();w._pending_live_joint_names=set();w._pending_preview_groups=set();w.live_toggle=Q.QCheckBox();w._set_checked_silently=lambda *a:None
    return w
with patch('autolife_shelf_grasp_306.action_studio.make_panel',panel):
    e=ActionExecutor(SimpleNamespace(root=root,run=root/'run'));e.timer.stop();b=e.ensure()
assert any(t.callback==b._advance_motion and t.timer_period_ns==20_000_000 for t in b.timers)
q=pose();clock=[100.];sent=[]
b.task_pause.clock=lambda:clock[0]
def motor(pub,payload):
    before=dict(b._current)
    for g in ('neck','left_arm','right_arm','leg_waist'):
        if GROUPS[g]['command_key'] in payload:b._current.update(zip(group_joint_names(g),payload[GROUPS[g]['command_key']]))
    for side in ('Left','Right'):
        key=side.lower()+'_gripper_target_joints_position'
        if key in payload:b._current['Joint_'+side+'_Gripper']=payload[key][0]
    for n in b._current:b._details[n]['speed']=(b._current[n]-before[n])/.02
    for n in b._current:
        lead=float(b.get_parameter('max_gripper_command_lead_deg' if n in GRIPPER_JOINTS else 'max_single_step_deg').value)
        assert abs(b._current[n]-before[n])<=lead+1e-6,(n,b._current[n],before[n],lead)
    b._controller_target.update(b._current);sent.append((clock[0],dict(b._current),payload))
b._publish_vendor=motor
def execute(poses,track_record=None,track_directory=None):
    global sent
    sent=[];b._current=q.copy();b._controller_target=q.copy();b._preview=q.copy();b._details={n:dict(communication_lost=False,speed=0.) for n in q}
    b._feedback_monotonic=clock[0];b._hardware_enabled=b._hardware_ready=b._leg_waist_enabled=True;b.outcome=None
    if track_record:
        from autolife_shelf_grasp_306.teaching_trajectory import plan
        record,b.teaching_segments,_=plan(track_record,q,track_directory or root/'data/fixed_actions')
    else:
        staged,_=stage_action(compile_action(draft(poses)),q);record,b.teaching_segments=build(staged,q)
    with patch('time.monotonic',lambda:clock[0]):
        b._on_action_command(String(data=json.dumps(dict(command='start',name='isolated-test',action=record))))
        assert b._action_state,b._last_event
        for _ in range(4000):
            clock[0]+=.02;b._feedback_monotonic=clock[0]
            for detail in b._details.values():detail['speed']=0.
            b._advance_action()
            if b._action_state is None:break
    assert b.outcome and b.outcome[1],b.outcome
    return record

a=pose(Joint_Right_Shoulder_Inner=10);bpose=pose(Joint_Right_Shoulder_Inner=20);c=pose(Joint_Right_Shoulder_Inner=30)
execute([a,bpose,c])
samples=[(t,p['Joint_Right_Shoulder_Inner']) for t,p,payload in sent if 'right_arm_target_joints_position' in payload]
middle=[(samples[i+1][0]-samples[i][0]) for i in range(len(samples)-1) if 7<samples[i][1]<23]
assert max(middle)<.041,max(middle)
close=pose(Joint_Right_Shoulder_Inner=20,Joint_Right_Gripper=180)
away=pose(Joint_Right_Shoulder_Inner=-10,Joint_Right_Gripper=180)
openpose=pose(Joint_Right_Shoulder_Inner=-10,Joint_Right_Gripper=10)
execute([close,away,openpose])
previous=q
for t,p,_ in sent:
    if abs(p['Joint_Right_Gripper']-previous['Joint_Right_Gripper'])>1e-8:
        expected=20 if p['Joint_Right_Gripper']>previous['Joint_Right_Gripper'] else -10
        assert abs(p['Joint_Right_Shoulder_Inner']-expected)<1e-6
    previous=p
print('EXECUTOR_OK through_points_no_dwell; body_then_jaws; guarded_receipts; 50Hz timer',flush=True)
import numpy as np
from autolife_shelf_grasp_306.teaching_track import make_draft,NAMES
q=pose(Joint_Right_Elbow=-70,Joint_Left_Elbow=-70,Joint_Right_Gripper=100)
t=np.arange(0.,6.001,.02);measurements=np.array([[q[n] for n in NAMES]]*len(t))
measurements[:,NAMES.index('Joint_Right_Elbow')]+=12*np.sin(t/6.*2*np.pi)
measurements[:,NAMES.index('Joint_Right_Gripper')]+=40*np.clip((t-3.)/.5,0.,1.)
legacy=make_draft(root/'data/fixed_actions','连续抓取',t,measurements);legacy['timing']='interaction'
record=compile_action(legacy)
execute([],record)
elbow=[p['Joint_Right_Elbow'] for _,p,payload in sent if 'right_arm_target_joints_position' in payload]
assert max(elbow)-min(elbow)>22
previous=q;last_body_move=None
for now,p,payload in sent:
    if any(abs(p[n]-previous[n])>1e-6 for n in BODY_JOINTS):last_body_move=now
    if any(abs(p[n]-previous[n])>1e-6 for n in GRIPPER_JOINTS):
        assert all(abs(p[n]-previous[n])<1e-6 for n in BODY_JOINTS)
        assert last_body_move is None or now-last_body_move>=.24
    previous=p
print('TRACK_EXECUTOR_OK: recorded reversal retained, actual controller receipts, body-settle before jaw action',flush=True)
original=make_draft(root/'data/fixed_actions','原时序抓取',t,measurements)
execute([],compile_action(original))
motor_times=[stamp for stamp,_,_ in sent]
assert abs(motor_times[-1]-motor_times[0]-t[-1])<.025,(motor_times[-1]-motor_times[0],t[-1])
assert any(3.05<stamp-motor_times[0]<3.45 and p['Joint_Right_Gripper']>100 and abs(p['Joint_Right_Elbow']+70)>.1 for stamp,p,_ in sent)
print('ORIGINAL_EXECUTOR_OK: exact 6-second clock, synchronous arm/jaw movement, no block waits',flush=True)
if os.environ.get('CHECK_NAMED_TRACK')=='1':
    from autolife_shelf_grasp_306.teaching_library import TeachingLibrary
    directory=Path('/home/ubuntu/ros2_ws/src/autolife_shelf_grasp_306/data/fixed_actions')
    customer=TeachingLibrary(directory).load('轨迹_1');q=dict(customer['frames'][0]['joints_deg'])
    execute([],compile_action(customer),directory)
    elapsed=sent[-1][0]-sent[0][0]
    assert abs(elapsed-customer['recorded_sec'])<.025
    print('NAMED_TRACK_MOCK_CONTROLLER_OK',dict(recorded=customer['recorded_sec'],controller=elapsed),flush=True)
e.shutdown()
