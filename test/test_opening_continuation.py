import ast
import json
import time
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from test_opening_residual import fixture,stable
from autolife_shelf_grasp_306.action_arrival import remember_opening_arrival


def test_current_shorter_release_not_scaled_to_three_degrees():
    b,goal=fixture();n='Joint_Left_Gripper'
    goal[n]=120.;b._action_state['start_values'][n]=180.9
    b._action_state['commanded_values'][n]=120.;b._current[n]=124.2348404487525
    assert stable(b,goal)


def release_then_next():
    b,goal=fixture();n='Joint_Left_Gripper'
    goal[n]=120.;b._action_state['start_values'][n]=180.9
    b._action_state['commanded_values'][n]=120.;b._current[n]=124.23484
    assert stable(b,goal)
    b._action_state['frame_index']+=1
    b._action_state['start_values']=dict(b._current)
    return b,goal


def test_following_recorded_frame_keeps_release_tolerance():
    b,goal=release_then_next();n='Joint_Left_Gripper'
    goal[n]=121.3;b._action_state['commanded_values'][n]=goal[n]
    assert stable(b,goal,11.)
    assert b._action_state['opening_references'][n]['goal']==120.


@pytest.mark.parametrize('case',['closing','drifting_target','new_loop','arm_error','large_residual','feedback_lost'])
def test_continuation_does_not_hide_errors(case):
    b,goal=release_then_next();n='Joint_Left_Gripper'
    if case=='closing':goal[n]=180.
    if case=='drifting_target':goal[n]=123.
    if case=='new_loop':b._action_state['loop_index']+=1
    if case=='arm_error':b._current['Joint_Left_Elbow']+=4.
    if case=='large_residual':b._current[n]=126.
    if case=='feedback_lost':b._is_feedback_fresh=lambda:False
    b._action_state['commanded_values'][n]=goal[n]
    assert not stable(b,goal,11.)


def test_ordinary_precise_opening_also_supports_next_frame():
    b,goal=fixture();n='Joint_Left_Gripper'
    b._current[n]=goal[n];remember_opening_arrival(b,goal)
    assert n in b._action_state['opening_references']


def test_editor_error_saved_before_action_state_cleared(tmp_path):
    import autolife_shelf_grasp_306.action_studio as module
    tree=ast.parse(Path(module.__file__).read_text())
    cls=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='ReceiptBridge')
    class Base:
        def _cancel_action(self,reason):self._action_state=None
    scope=dict(JointBridge=Base,self=NS(owner=NS(run=tmp_path)),time=time,json=json,__name__=module.__name__,__package__=module.__package__)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(module.__file__),'exec'),scope)
    b,goal=fixture();bridge=scope['ReceiptBridge']()
    bridge.__dict__.update(b.__dict__)
    bridge._cancel_action('simulated residual timeout')
    saved=json.loads((tmp_path/'last_action_error.json').read_text())
    assert saved['reason']=='simulated residual timeout' and saved['targets']==goal
    assert saved['frame_index']==1 and saved['measured']==b._current
    assert bridge._action_state is None and bridge.outcome[1] is False
    # Later status/idle changes cannot overwrite this saved failure.
    bridge._record_action_event('locked','idle')
    assert json.loads((tmp_path/'last_action_error.json').read_text())==saved
