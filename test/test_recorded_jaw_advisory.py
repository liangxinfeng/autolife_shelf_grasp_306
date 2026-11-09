from types import SimpleNamespace as NS
import pytest
from autolife_shelf_grasp_306.teaching_gripper_follow import check
from control_center_joint_control.joint_definitions import GRIPPER_JOINTS

@pytest.mark.parametrize('actual,target',[(237.794626,222.64),(10.,330.),(330.,10.)])
def test_position_mismatch_only_logs_once_and_never_blocks(actual,target):
    events=[];state={}
    node=NS(_current={n:actual for n in GRIPPER_JOINTS},_record_action_event=lambda *a:events.append(a))
    for i in range(500):
        assert not check(node,state,{n:target for n in GRIPPER_JOINTS},i*.02,tracking_required=False)
    assert len(events)==len(GRIPPER_JOINTS)
    assert all(e[0]=='recorded_gripper_tracking_advisory' for e in events)
    assert all(len(j['samples'])<22 for j in state['recorded_gripper_history'].values())
    node._current={n:target for n in GRIPPER_JOINTS}
    assert not check(node,state,{n:target for n in GRIPPER_JOINTS},10.,tracking_required=False)
    assert all(not j['tracking_warning'] for j in state['recorded_gripper_history'].values())

@pytest.mark.parametrize('actual,target',[(float('nan'),10.),(10.,float('inf')),(370.,10.),(10.,400.)])
def test_invalid_feedback_or_limits_still_block(actual,target):
    # A preceding jaw's advisory must not hide the other jaw's hard failure.
    first,last=GRIPPER_JOINTS
    node=NS(_current={first:10.,last:actual},_record_action_event=lambda *a:None)
    state={}
    assert check(node,state,{first:330.,last:target},0.,tracking_required=False)
    assert check(node,state,{first:330.,last:target},.2,tracking_required=False)

@pytest.mark.parametrize('jaw_mode',['frozen_open','frozen_closed','wrong_direction'])
def test_continuous_playback_completes_on_recorded_time_with_jaw_lag(tmp_path,monkeypatch,jaw_mode):
    from test_recorded_timing import fixture,Controller
    from autolife_shelf_grasp_306.teaching_motion import compile_action
    from autolife_shelf_grasp_306.teaching_replay import plan,advance
    d,t,q,entry=fixture(tmp_path,seconds=6.)
    r,segs,_=plan(compile_action(d),entry,tmp_path)
    clock=[0.];monkeypatch.setattr('time.monotonic',lambda:clock[0])
    c=Controller(r,segs,entry,clock);sent=[];events=[]
    c._record_action_event=lambda *a:events.append(a)
    def publish(state,desired,frame):
        sent.append((clock[0],dict(desired)))
        c._current=dict(desired)
        for n in GRIPPER_JOINTS:
            c._current[n]=10. if jaw_mode=='frozen_open' else 330. if jaw_mode=='frozen_closed' else 370.-desired[n]
        state['commanded_values']=dict(desired)
    c._publish_action_targets=publish
    while c._action_state and clock[0]<7.:
        advance(c);clock[0]+=.02
    assert c.outcome and c.outcome[1] and not c.reason
    assert 6.<=sent[-1][0]<=6.021
    assert '夹爪到位未核验' in c.outcome[2]
    assert any(e[0]=='recorded_gripper_arrival_unverified' for e in events)

@pytest.mark.parametrize('fault',['lost','nan','range','body'])
def test_hard_faults_still_stop_before_output(tmp_path,monkeypatch,fault):
    from test_recorded_timing import fixture,Controller
    from autolife_shelf_grasp_306.teaching_motion import compile_action
    from autolife_shelf_grasp_306.teaching_replay import plan,advance
    d,t,q,entry=fixture(tmp_path,seconds=6.)
    r,segs,_=plan(compile_action(d),entry,tmp_path)
    clock=[0.];monkeypatch.setattr('time.monotonic',lambda:clock[0])
    c=Controller(r,segs,entry,clock)
    if fault=='lost':c._communication_lost_joints=lambda names:[GRIPPER_JOINTS[0]]
    if fault=='nan':c._current[GRIPPER_JOINTS[0]]=float('nan')
    if fault=='range':c._current[GRIPPER_JOINTS[0]]=500.
    if fault=='body':c._current['Joint_Right_Elbow']+=30.
    advance(c)
    assert c.reason and not c.outcome and not c.published
