"""失败记录回归；所有节点/发布器为替身，不建立运动接口。"""
from types import SimpleNamespace as NS
from copy import deepcopy
import numpy as np
import pytest
from PyQt5 import QtWidgets as Q,QtCore as C
from test_teaching import pose,draft
from autolife_shelf_grasp_306.teaching_motion import compile_action
from autolife_shelf_grasp_306.teaching_trajectory import plan
from autolife_shelf_grasp_306.pose_gripper_intent import stable_held_grippers
from autolife_shelf_grasp_306.pose_synchronized_output import before_tick
from autolife_shelf_grasp_306.pose_start_ready import ready


def held_fixture(tmp_path):
    q=pose(Joint_Right_Gripper=108.9777778)
    r,segs,phases=plan(compile_action(draft([q])),q,tmp_path)
    state=dict(record=r,stage='transition',frame_index=0,loop_index=0,progress=1.,commanded_values=dict(q))
    b=NS(_current=dict(q),_details={n:dict(speed=0.,communication_lost=False) for n in q},
         _is_feedback_fresh=lambda:True,_feedback_monotonic=10.,teaching_segments=segs,
         get_parameter=lambda n:NS(value=.2))
    b._current['Joint_Right_Gripper']=111.95349
    return b,state,q


@pytest.mark.parametrize('fault',['none','offline','body_error','large_jaw_error','new_opening','command_pending','frozen','unfinished','moving_jaw','nan'])
def test_recorded_held_jaw_residual_and_faults(tmp_path,fault):
    b,s,q=held_fixture(tmp_path);jaw='Joint_Right_Gripper'
    if fault=='offline':b._details[jaw]['communication_lost']=True
    if fault=='body_error':b._current['Joint_Right_Elbow']-=1.
    if fault=='large_jaw_error':b._current[jaw]+=4.
    if fault=='new_opening':b.teaching_segments[0]['start'][jaw]+=20.
    if fault=='command_pending':s['commanded_values'][jaw]-=1.
    if fault=='unfinished':s['progress']=.999
    if fault=='nan':b._current[jaw]=float('nan')
    result=False
    for i in range(12):
        if fault!='frozen':b._feedback_monotonic=10.+i*.1
        if fault=='moving_jaw':b._current[jaw]=108.+(i%2)*3.
        result=stable_held_grippers(b,s,q)
    assert result is (fault=='none')


def test_hold_tag_only_on_body_phase(tmp_path):
    q=pose();r,segs,p=plan(compile_action(draft([pose(Joint_Right_Gripper=180.)])),q,tmp_path)
    assert [s['hold_grippers'] for s in segs]==[True,False]


def test_actual_bridge_leaves_body_stage_with_recorded_residual(tmp_path,monkeypatch):
    from control_center_joint_control.joint_bridge import JointBridge
    from autolife_shelf_grasp_306.pose_execution_bridge import make_bridge
    from test_pose_arrival import fixture
    original,goal=fixture();goal['Joint_Right_Gripper']=108.9777778
    clock=[100.];monkeypatch.setattr('time.monotonic',lambda:clock[0])
    r,segs,phases=plan(compile_action(draft([goal])),goal,tmp_path)
    cls=make_bridge(JointBridge,NS(owner=NS(run=tmp_path),gripper_targets={}))
    b=object.__new__(cls);b.__dict__.update(original.__dict__)
    b.teaching_segments=segs;b._current=dict(goal);b._current['Joint_Right_Gripper']=111.95349
    b._preview=dict(goal);b._controller_target=dict(goal);b._gripper_pub=None;b._whole_body_pub=None
    b._communication_lost_joints=lambda _:[];b._publish_vendor=lambda *args:None;b._event=lambda *args:None
    b.get_parameter=lambda n:NS(value=dict(goal_tolerance_deg=.2,gripper_goal_tolerance_deg=2.,max_single_step_deg=2.,action_progress_epsilon_deg=.2,action_stall_timeout_sec=30.)[n])
    s=b._action_state;s.update(record=r,start_values=dict(goal),commanded_values=dict(goal),
        frame_started=99.,frame_deadline=110.,last_progress_at=100.,synchronized_clock=dict(key=(0,0),progress=1.,last=100.))
    for i in range(15):
        clock[0]=100.+i*.05;b._feedback_monotonic=clock[0]
        b._advance_action()
        if b._action_state is None or b._action_state['stage']=='hold':break
    assert b._action_state is None or b._action_state['stage']=='hold'


@pytest.mark.parametrize('now',[1234.12345,999999.9213,123456789.25])
@pytest.mark.parametrize('seconds',[.6,1.64,4.19])
def test_endpoint_rounding_reaches_arrival(monkeypatch,now,seconds):
    monkeypatch.setattr('time.monotonic',lambda:now)
    s=dict(stage='transition',frame_index=0,loop_index=0,frame_deadline=now+10,
           synchronized_clock=dict(key=(0,0),progress=1.,last=now-.02),record=dict(speed_percent=100))
    b=NS(_action_state=s,teaching_segments=[dict(seconds=seconds)],_cancel_action=lambda _:pytest.fail('deadline'))
    before_tick(b)
    assert (now-s['frame_started'])/seconds>=1.


@pytest.mark.parametrize('fault',['none','delayed_lost','permanent_lost','stale','movement'])
def test_enable_gate_waits_without_sending_goals(monkeypatch,fault):
    clock=[100.];monkeypatch.setattr('time.monotonic',lambda:clock[0]);q=pose()
    b=NS(_current=dict(q),_details={n:dict(speed=0.) for n in q},_feedback_monotonic=100.,
         _is_feedback_fresh=lambda:fault!='stale',_communication_lost_joints=lambda _:[])
    s=NS(lower_ready_gate=dict(started=100.,anchor=q,stable=None));passed=None
    if fault=='movement':b._current['Joint_Knee']+=2.
    if fault in ('permanent_lost','stale','movement'):
        with pytest.raises(ValueError):
            for i in range(70):
                clock[0]=100.+i*.1;b._feedback_monotonic=clock[0]
                b._communication_lost_joints=lambda _:['Joint_Left_Elbow'] if fault=='permanent_lost' else []
                assert not ready(b,s)
        return
    for i in range(40):
        clock[0]=100.+i*.1;b._feedback_monotonic=clock[0]
        b._communication_lost_joints=lambda _:['Joint_Left_Elbow'] if fault=='delayed_lost' and .6<clock[0]-100.<1. else []
        if ready(b,s):passed=clock[0]-100.;break
    assert passed is not None and passed>=1.
    if fault=='delayed_lost':assert passed>=2.


def ui():
    from autolife_shelf_grasp_306.pose_workspace import PoseWorkspace
    app=Q.QApplication.instance() or Q.QApplication([])
    w=PoseWorkspace.__new__(PoseWorkspace);Q.QWidget.__init__(w)
    w.mode=Q.QTabWidget(w);layout=Q.QVBoxLayout(w);layout.addWidget(w.mode)
    w.ik_session=None;w.ik_result=None;w.ik_pending=None;w.ik_proc=None;w.edits={};w.tcp_axes=None
    w.mode_changed=lambda *a:None;w.available=lambda:None
    w.make_ik();w.resize(520,600);w.show();app.processEvents()
    return app,w


def test_ik_buttons_fixed_after_long_status():
    app,w=ui();buttons=w.findChildren(Q.QPushButton);before=[b.mapTo(w,C.QPoint()).y() for b in buttons]
    w.ik_text.setText('基座系末端 X 123 / Y 456 / Z 789 mm\n固定部件模型提示：'+'提示文字'*40)
    app.processEvents();assert before==[b.mapTo(w,C.QPoint()).y() for b in buttons]
    w.close();w.deleteLater()


def test_first_direction_auto_captures_and_subsequent_direction_accumulates():
    app,w=ui();q=pose();calls=[];w.capture=lambda:dict(q)
    w.request_ik=lambda op,source,**kw:calls.append((op,source,kw))
    w.nudge_ik(0,1,False);assert calls[-1][0]=='capture' and w.ik_after_capture==(0,1,False)
    w.ik_session=dict(source=dict(q),target=np.eye(4).tolist())
    w.nudge_ik(0,1,False);assert calls[-1][0]=='solve' and calls[-1][2]['target'][0][3]==.005
    w.ik_result=dict(joints={},target=calls[-1][2]['target'])
    w.nudge_ik(0,1,False);assert calls[-1][2]['target'][0][3]==.01
    q['Joint_Right_Elbow']-=3.;w.nudge_ik(0,1,False);assert calls[-1][0]=='capture'
    w.close();w.deleteLater()


@pytest.mark.parametrize('use_object',[True,False])
def test_single_pose_optional_object_correction(use_object):
    from autolife_shelf_grasp_306.pose_workspace import PoseWorkspace
    q=pose();records=[]
    w=NS(capture=lambda:q,pose_list=NS(currentText=lambda:'p'),manual_speed=NS(value=lambda:70),
         store=NS(read=lambda:{'p':q},binding=lambda _:dict(reference_id='object')),
         pose_use_object=NS(isChecked=lambda:use_object),execute_record=records.append)
    PoseWorkspace.execute_pose(w)
    assert bool(records[0].get('visual_bindings')) is use_object
