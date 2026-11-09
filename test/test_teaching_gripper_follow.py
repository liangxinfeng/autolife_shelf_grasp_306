from types import SimpleNamespace as NS
import pytest
from autolife_shelf_grasp_306.teaching_gripper_follow import check

JAW='Joint_Right_Gripper'


def test_fast_gripper_motion_with_bounded_transport_delay():
    node=NS(_current={JAW:10.});state={};goals=[]
    for i in range(100):
        now=i*.02;target=10.+280.*min(1.,max(0.,(now-.2)/.14))
        goals.append(target);node._current[JAW]=goals[max(0,i-4)]
        assert not check(node,state,{JAW:target},now)


def test_blocked_gripper_is_not_accepted_as_completed_grasp():
    node=NS(_current={JAW:10.});state={};failure=''
    for i in range(70):
        now=i*.02;target=10.+280.*min(1.,max(0.,(now-.2)/.14))
        failure=check(node,state,{JAW:target},now)
        if failure:break
    assert failure and .2<now<.9


def test_wrong_direction_stops_and_full_open_residual_is_bounded():
    node=NS(_current={JAW:12.9});state={}
    for i in range(40):assert not check(node,state,{JAW:10.},i*.02)
    node._current[JAW]=40.
    for i in range(40,55):failure=check(node,state,{JAW:10.},i*.02)
    assert failure


@pytest.mark.parametrize('actual,target',[(float('nan'),10.),(10.,float('nan')),(370.,10.),(10.,400.)])
def test_invalid_values_fail_before_publication(actual,target):
    assert check(NS(_current={JAW:actual}),{},{JAW:target},1.)


def test_duplicate_ticks_do_not_accumulate_or_fake_elapsed_time():
    node=NS(_current={JAW:10.});state={}
    for _ in range(1000):assert not check(node,state,{JAW:200.},1.)
    assert len(state['recorded_gripper_history'][JAW]['samples'])==1
    assert check(node,state,{JAW:200.},1.2)


def test_moving_window_expires_old_opening_and_is_bounded():
    node=NS(_current={JAW:10.});state={}
    for i in range(500):check(node,state,{JAW:10.},i*.02)
    assert len(state['recorded_gripper_history'][JAW]['samples'])<22
    assert check(node,state,{JAW:10.},0.)
