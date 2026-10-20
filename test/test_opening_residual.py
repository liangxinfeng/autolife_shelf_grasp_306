"""Replay the 2026-10-10 frame-2 residual; never send hardware commands."""
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest
from autolife_shelf_grasp_306.action_arrival import settled_opening_arrival
from control_center_joint_control.joint_bridge import JointBridge


def fixture(error=3.14710385851513):
    goal={'Joint_Left_Gripper':118.26789614148487,'Joint_Right_Gripper':10.,
          'Joint_Left_Elbow':73.39861343372955,'Joint_Knee':-55.40101106501808}
    start=dict(goal);start['Joint_Left_Gripper']=261.8
    record=dict(groups=['left_arm','leg_waist','grippers'],speed_percent=100,loop_count=1,keyframes=[
        dict(joints_deg=goal,transition_sec=1.5,hold_sec=.2,easing='smooth'),
        dict(joints_deg={**goal,'Joint_Left_Elbow':129.6478},transition_sec=1.5,hold_sec=.2,easing='smooth')])
    state=dict(record=record,name='recorded-residual',frame_index=0,loop_index=0,start_values=start,
        commanded_values=dict(goal),frame_started=0.,progress=1.,stage='transition',
        best_error_deg=63.,last_progress_at=0.,frame_deadline=16.,joint_names=list(goal))
    current=dict(goal);current['Joint_Left_Gripper']+=error;current['Joint_Left_Elbow']-=.022
    params=dict(goal_tolerance_deg=1.,gripper_goal_tolerance_deg=2.,action_progress_epsilon_deg=.2,action_stall_timeout_sec=30.)
    b=NS(_action_state=state,_current=current,_details={n:dict(speed=0.,communication_lost=False) for n in goal},
        _feedback_monotonic=10.,_hardware_enabled=True,_hardware_ready=True,_leg_waist_enabled=True,_preview={})
    b.get_parameter=lambda n:NS(value=params[n]);b._is_feedback_fresh=lambda:True
    return b,goal


def stable(b,goal,start=10.):
    result=False
    for i in range(7):
        b._feedback_monotonic=start+i*.1;result=settled_opening_arrival(b,goal)
    return result


@pytest.mark.parametrize('error',[2.01,2.9,3.14710385851513,4.,5.,-3.147])
def test_small_settled_opening_continues(error):
    b,goal=fixture(error)
    assert not JointBridge._action_target_reached(b,goal)
    assert not settled_opening_arrival(b,goal)
    assert stable(b,goal)


@pytest.mark.parametrize('case',['large','closing','holding','short_travel','arm','body','command_pending','moving','lost','nan','stale'])
def test_does_not_mask_real_failure(case):
    b,goal=fixture();n='Joint_Left_Gripper'
    if case=='large':b._current[n]=goal[n]+5.01
    if case=='closing':b._action_state['start_values'][n]=100.
    if case=='holding':b._action_state['start_values'][n]=goal[n]
    if case=='short_travel':b._action_state['start_values'][n]=goal[n]+10.
    if case=='arm':b._current['Joint_Left_Elbow']=goal['Joint_Left_Elbow']+1.01
    if case=='body':b._current['Joint_Knee']=goal['Joint_Knee']+1.01
    if case=='command_pending':b._action_state['commanded_values'][n]=goal[n]+1.
    if case=='moving':b._details[n]['speed']=2.01
    if case=='lost':b._details[n]['communication_lost']=True
    if case=='nan':b._current[n]=float('nan')
    if case=='stale':b._is_feedback_fresh=lambda:False
    assert not stable(b,goal)


def test_repeated_feedback_does_not_count():
    b,goal=fixture()
    for _ in range(100):assert not settled_opening_arrival(b,goal)


def test_drift_resets_stability():
    b,goal=fixture();assert stable(b,goal)
    b._current['Joint_Left_Gripper']+=.5;b._feedback_monotonic=10.7
    assert not settled_opening_arrival(b,goal)
    assert stable(b,goal,10.7)


def test_feedback_gap_and_new_frame_reset():
    b,goal=fixture();assert stable(b,goal)
    b._feedback_monotonic=11.
    assert not settled_opening_arrival(b,goal)
    assert stable(b,goal,11.)
    b._action_state['frame_index']=1;b._feedback_monotonic=11.7
    assert not settled_opening_arrival(b,goal)


def test_real_bridge_advances_before_timeout_without_changing_goal():
    b,goal=fixture();state=b._action_state;advanced=[];canceled=[]
    b._communication_lost_joints=lambda _:[];b._publish_action_targets=lambda *a:None
    b._action_frame_max_error=lambda frame:JointBridge._action_frame_max_error(b,frame)
    b._action_target_reached=lambda targets:JointBridge._action_target_reached(b,targets) or settled_opening_arrival(b,targets)
    b._advance_action_frame=lambda now:advanced.append(now)
    b._cancel_action=lambda reason:canceled.append(reason)
    b._action_error_summary=lambda frame:'recorded 3.147 degree residual'
    for i in range(20):
        now=1.5+i*.1;b._feedback_monotonic=now
        with patch('control_center_joint_control.joint_bridge.time.monotonic',return_value=now):JointBridge._advance_action(b)
        if advanced:break
    assert advanced and advanced[0]<3. and not canceled
    assert state['frame_deadline']==16. and state['record']['keyframes'][0]['joints_deg']==goal
