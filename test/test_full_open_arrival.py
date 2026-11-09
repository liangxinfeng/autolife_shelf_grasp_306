"""Reproduce the 345 track_1 preparation failure with no robot connection."""
from copy import deepcopy
from types import SimpleNamespace as NS
import pytest
from autolife_shelf_grasp_306.action_arrival import settled_full_open_arrival,settled_opening_arrival
from control_center_joint_control.joint_bridge import JointBridge


def fixture():
    goal={'Joint_Left_Gripper':10.,'Joint_Right_Gripper':10.,'Joint_Right_Elbow':-110.,'Joint_Knee':1.347}
    measured={**goal,'Joint_Left_Gripper':11.4330534935,'Joint_Right_Gripper':12.8868951797}
    b=NS(_current=measured,_feedback_monotonic=10.,_is_feedback_fresh=lambda:True,
        _details={n:dict(speed=0.,communication_lost=False) for n in goal},
        get_parameter=lambda n:NS(value={'goal_tolerance_deg':1.,'gripper_goal_tolerance_deg':2.}[n]))
    b._details['Joint_Left_Gripper']['speed']=4.407439
    b._details['Joint_Right_Gripper']['speed']=1.469146
    b._action_state=dict(loop_index=0,frame_index=1,start_values=dict(measured),commanded_values=dict(goal))
    return b,goal


def stable(b,goal):
    result=False
    for i in range(7):
        b._feedback_monotonic=10.+i*.1
        result=settled_full_open_arrival(b,goal)
    return result


def test_recorded_failure_now_reaches_without_changing_commands():
    b,goal=fixture();commands=deepcopy(b._action_state['commanded_values'])
    assert not JointBridge._action_target_reached(b,goal)
    assert not settled_opening_arrival(b,goal)
    assert not settled_full_open_arrival(b,goal)
    assert stable(b,goal)
    assert b._action_state['commanded_values']==commands


@pytest.mark.parametrize('fault',['large','partial','closing','body','command','moving_speed','lost','stale','nan','start_below_open'])
def test_endpoint_exception_cannot_mask_other_failures(fault):
    b,goal=fixture();n='Joint_Right_Gripper'
    if fault=='large':b._current[n]=15.01
    elif fault=='partial':goal[n]=30.;b._current[n]=32.9;b._action_state['start_values'][n]=32.9;b._action_state['commanded_values'][n]=30.
    elif fault=='closing':goal[n]=200.;b._current[n]=197.1;b._action_state['commanded_values'][n]=200.
    elif fault=='body':b._current['Joint_Knee']+=1.01
    elif fault=='command':b._action_state['commanded_values'][n]=10.1
    elif fault=='moving_speed':b._details[n]['speed']=5.01
    elif fault=='lost':b._details[n]['communication_lost']=True
    elif fault=='stale':b._is_feedback_fresh=lambda:False
    elif fault=='nan':b._current[n]=float('nan')
    elif fault=='start_below_open':b._action_state['start_values'][n]=9.
    assert not stable(b,goal)


def test_position_drift_is_rejected_even_with_low_reported_speed():
    b,goal=fixture()
    for i in range(20):
        b._feedback_monotonic=10.+i*.1
        b._current['Joint_Right_Gripper']=12.5+(i%2)*.6
        assert not settled_full_open_arrival(b,goal)


def test_stationary_encoder_noise_is_bounded():
    b,goal=fixture()
    for i in range(7):
        b._feedback_monotonic=10.+i*.1
        b._current['Joint_Right_Gripper']=12.8869+(i%2)*.0765
        result=settled_full_open_arrival(b,goal)
    assert result


def test_cached_feedback_new_goal_and_frame_reset_window():
    b,goal=fixture()
    for _ in range(100):assert not settled_full_open_arrival(b,goal)
    assert stable(b,goal)
    b._action_state['frame_index']+=1;b._feedback_monotonic=10.7
    assert not settled_full_open_arrival(b,goal)


def test_real_editor_bridge_uses_full_open_receipt(tmp_path):
    import ast,json,time
    from pathlib import Path
    import autolife_shelf_grasp_306.action_studio as module
    cls=next(n for n in ast.walk(ast.parse(Path(module.__file__).read_text()))
             if isinstance(n,ast.ClassDef) and n.name=='ReceiptBridge')
    class Base:
        _action_target_reached=JointBridge._action_target_reached
        def _event(self,*args):pass
    scope=dict(JointBridge=Base,self=NS(owner=NS(run=tmp_path)),time=time,json=json,
               __name__=module.__name__,__package__=module.__package__)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(module.__file__),'exec'),scope)
    b,goal=fixture();bridge=object.__new__(scope['ReceiptBridge'])
    bridge.__dict__.update(b.__dict__)
    bridge._hardware_enabled=bridge._hardware_ready=True
    bridge._action_state.update(stage='transition',progress=1.,name='test',
        record=dict(groups=[],keyframes=[dict(joints_deg=goal)]*2))
    for i in range(7):
        bridge._feedback_monotonic=10.+i*.1
        result=bridge._action_target_reached(goal)
    assert result
    events=[json.loads(s) for s in (tmp_path/'action_events.jsonl').read_text().splitlines()]
    assert any(e['event']=='settled_full_open' for e in events)


def test_recorded_final_pose_can_settle_inside_original_time(tmp_path,monkeypatch):
    from test_recorded_timing import fixture as recorded,Controller
    from autolife_shelf_grasp_306.teaching_motion import compile_action
    from autolife_shelf_grasp_306.teaching_trajectory import plan
    from autolife_shelf_grasp_306.teaching_replay import advance
    from autolife_shelf_grasp_306.teaching_track import make_draft,NAMES
    d,t,q,entry=recorded(tmp_path,seconds=3.)
    # A finished recording with a stationary final pose and an open-end offset.
    q[:]=q[0];q[:,NAMES.index('Joint_Right_Gripper')]=10.
    d=make_draft(tmp_path,'full-open-final',t,q)
    r,segs,_=plan(compile_action(d),entry,tmp_path);clock=[0.]
    monkeypatch.setattr('time.monotonic',lambda:clock[0])
    c=Controller(r,segs,entry,clock)
    c._action_state.update(loop_index=0,start_values=dict(entry))
    c._details={n:dict(speed=0.,communication_lost=False) for n in entry}
    def publish(state,desired,frame):
        c.published+=1;state['commanded_values'].update(desired)
        c._current={**desired,'Joint_Right_Gripper':12.8869}
        c._details['Joint_Right_Gripper']['speed']=4.4074
        c._feedback_monotonic=clock[0]
    c._publish_action_targets=publish
    while c._action_state and clock[0]<3.1:
        advance(c);clock[0]+=.02
    assert c.outcome and c.outcome[1] and not c.reason
    assert 3.<=clock[0]<=3.04
