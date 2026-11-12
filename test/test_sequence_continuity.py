"""连续姿态与可逆纠偏：真实Qt/控制桥方法，替身反馈，无ROS连接。"""
from copy import deepcopy
from types import SimpleNamespace as NS
import numpy as np
import pytest
from PyQt5 import QtWidgets as Q,QtCore as C
from test_teaching import pose,draft
from test_visual_actions import bound
from autolife_shelf_grasp_306.teaching_motion import compile_action
from autolife_shelf_grasp_306.teaching_trajectory import plan,evaluate


def test_list_checkbox_save_reload_reenable_and_reorder(tmp_path):
    from autolife_shelf_grasp_306.app import Window
    app=Q.QApplication.instance() or Q.QApplication([])
    root=tmp_path/'root';root.mkdir();(root/'data').mkdir();run=tmp_path/'run';run.mkdir()
    w=Window(root,run,start_worker=False);w.timer.stop();w.busy=False;p=w.pose_workspace
    try:
        p.store.save('交互位',pose(),visual_binding=bound());p.reload_poses();p.pose_list.setCurrentText('交互位')
        p.add_pose();p.add_pose();p.sequence_name.setText('连续动作');p.rows[0]['hold_sec']=.4;p.render_rows()
        p.sequence.item(0).setCheckState(C.Qt.Unchecked)
        assert p.rows[0]['visual_binding']==bound() and not p.rows[0]['visual_enabled']
        assert p.rows[1].get('visual_enabled',True) and p.store.binding('交互位')==bound()
        p.make_continuous();assert all(r['hold_sec']==0 for r in p.rows)
        p.save_action();p.dirty=False;p.load_action()
        assert p.sequence.item(0).checkState()==C.Qt.Unchecked
        assert w.studio.records()['连续动作']['visual_bindings']=={'1':bound()}
        p.sequence.setCurrentRow(0);p.move_row(1)
        assert p.sequence.item(1).checkState()==C.Qt.Unchecked
        p.sequence.item(1).setCheckState(C.Qt.Checked);p.save_action()
        assert set(w.studio.records()['连续动作']['visual_bindings'])=={'0','1'}
        p.row_hold.setValue(.5);assert p.rows[1]['hold_sec']==.5
        p.make_continuous();assert p.row_hold.value()==0
    finally:
        p.dirty=False;w.studio.editor.dirty=False;w.close();app.processEvents()


@pytest.mark.parametrize('turn',[False,True])
def test_boundary_flags_tangent_and_no_extra_dwell(tmp_path,turn):
    qs=[pose(Joint_Right_Shoulder_Inner=v) for v in ([10,20,10] if turn else [10,20,30])]
    record,segs,_=plan(compile_action(draft(qs)),pose(),tmp_path)
    assert [s['continuous_boundary'] for s in segs]==[True,True,False]
    assert all(f['hold_sec']==0 for f in record['keyframes'])
    for a,b in zip(segs,segs[1:]):
        for order in (0,1,2):
            assert max(abs(evaluate(a,1,order)[n]-evaluate(b,0,order)[n]) for n in pose())<1e-7


def test_explicit_dwell_and_gripper_events_still_stop(tmp_path):
    qs=[pose(Joint_Right_Shoulder_Inner=10),pose(Joint_Right_Shoulder_Inner=20,Joint_Right_Gripper=100),pose(Joint_Right_Shoulder_Inner=30,Joint_Right_Gripper=100)]
    d=draft(qs);d['frames'][0]['hold_sec']=.4
    r,s,_=plan(compile_action(d),pose(),tmp_path)
    assert not any(v['continuous_boundary'] for v in s)
    assert r['keyframes'][1]['hold_sec']==.25
    assert s[1]['target']['Joint_Right_Gripper']==pose()['Joint_Right_Gripper']
    assert s[2]['target']['Joint_Right_Gripper']==100


@pytest.mark.parametrize('turn',[False,True])
def test_real_bridge_boundary_has_no_settle_and_next_tick_moves(tmp_path,monkeypatch,turn):
    from control_center_joint_control.joint_bridge import JointBridge
    from autolife_shelf_grasp_306.pose_execution_bridge import make_bridge
    from autolife_shelf_grasp_306.pose_synchronized_output import before_tick
    from test_pose_arrival import fixture
    start=pose();qs=[pose(Joint_Right_Shoulder_Inner=v) for v in ([10,20,10] if turn else [10,20,30])]
    record,segs,_=plan(compile_action(draft(qs)),start,tmp_path)
    original,_=fixture();cls=make_bridge(JointBridge,NS(owner=NS(run=tmp_path),gripper_targets={}))
    b=object.__new__(cls);b.__dict__.update(original.__dict__)
    b.teaching_segments=segs;b._current=dict(qs[1]);b._preview=dict(qs[1]);b._controller_target=dict(qs[1])
    b._details={n:dict(speed=4.,communication_lost=False) for n in start}
    params=dict(max_single_step_deg=2.,goal_tolerance_deg=.2,gripper_goal_tolerance_deg=2.,action_progress_epsilon_deg=.2,action_stall_timeout_sec=30.)
    b.get_parameter=lambda n:NS(value=params.get(n,2.));b._event=lambda *a:None;b._record_action_event=lambda *a:None
    b._communication_lost_joints=lambda _:[];b._is_feedback_fresh=lambda:True
    state=b._action_state;state.update(record=record,frame_index=1,commanded_values=dict(qs[1]),start_values=dict(qs[0]),progress=1.,stage='transition',
        synchronized_clock=dict(key=(0,1),progress=1.,last=99.98),frame_started=100-segs[1]['seconds'])
    # Even at a direction reversal no stationary-arrival wait is imposed.
    assert b._through_point() and b._action_target_reached(qs[1])
    clock=[100.];monkeypatch.setattr('time.monotonic',lambda:clock[0]);b._advance_action_frame(100.)
    assert b._action_state['frame_index']==2 and b._action_state['stage']=='transition'
    clock[0]+=.02;before_tick(b)
    assert clock[0]-b._action_state['frame_started']>.019
    assert b._action_state['synchronized_clock']['key']==(0,2)


def test_tracking_error_not_waived_at_continuous_point(tmp_path):
    from control_center_joint_control.joint_bridge import JointBridge
    from autolife_shelf_grasp_306.pose_execution_bridge import make_bridge
    from test_pose_arrival import fixture
    q=pose();record,segs,_=plan(compile_action(draft([pose(Joint_Right_Shoulder_Inner=10),pose(Joint_Right_Shoulder_Inner=20)])),q,tmp_path)
    original,_=fixture();b=object.__new__(make_bridge(JointBridge,NS(gripper_targets={})));b.__dict__.update(original.__dict__)
    b.teaching_segments=segs;target=record['keyframes'][0]['joints_deg'];b._current=dict(q)
    b._action_state.update(record=record,frame_index=0,progress=1.,commanded_values=dict(target))
    b.get_parameter=lambda n:NS(value=2.);assert not b._action_target_reached(target)


def test_complete_real_bridge_playback_with_zero_dwell(tmp_path,monkeypatch):
    from control_center_joint_control.joint_bridge import JointBridge
    from autolife_shelf_grasp_306.pose_execution_bridge import make_bridge
    from test_pose_arrival import fixture
    start=pose();targets=[pose(Joint_Right_Shoulder_Inner=v) for v in (5,10,5)]
    record,segs,_=plan(compile_action(draft(targets)),start,tmp_path)
    original,_=fixture();b=object.__new__(make_bridge(JointBridge,NS(owner=NS(run=tmp_path),gripper_targets={})))
    b.__dict__.update(original.__dict__);b.teaching_segments=segs;b._current=dict(start);b._preview=dict(start);b._controller_target=dict(start)
    b._details={n:dict(speed=0.,communication_lost=False) for n in start};b._whole_body_pub='body';b._gripper_pub='jaw'
    b._publish_vendor=lambda *a:None;b._event=lambda *a:None;b._record_action_event=lambda *a:None
    b._communication_lost_joints=lambda _:[];b._is_feedback_fresh=lambda:True
    params=dict(max_single_step_deg=2.,goal_tolerance_deg=.2,gripper_goal_tolerance_deg=2.,action_progress_epsilon_deg=.01,action_stall_timeout_sec=30.,max_gripper_command_lead_deg=100.)
    b.get_parameter=lambda n:NS(value=params.get(n,2.))
    state=b._action_state;state.update(record=record,frame_index=0,commanded_values=dict(start),start_values=dict(start),progress=0.,stage='transition',frame_started=100.,frame_deadline=140.,last_progress_at=100.,best_error_deg=5.)
    clock=[100.];monkeypatch.setattr('time.monotonic',lambda:clock[0]);changes=[];samples=[]
    for tick in range(3000):
        clock[0]=100.+tick*.02;b._feedback_monotonic=clock[0]
        before=b._action_state['frame_index'];b._advance_action()
        if b._action_state is None:break
        s=b._action_state;b._current.update(s['commanded_values']);samples.append((tick,s['frame_index'],s['commanded_values']['Joint_Right_Shoulder_Inner']))
        if s['frame_index']!=before:changes.append(tick)
    assert b._action_state is None and b.outcome[1] and len(changes)==2
    by_tick={s[0]:s[2] for s in samples}
    for tick in changes:
        assert abs(by_tick[tick+1]-by_tick[tick])>1e-8
    assert clock[0]-100<sum(s['seconds'] for s in segs)+.15
