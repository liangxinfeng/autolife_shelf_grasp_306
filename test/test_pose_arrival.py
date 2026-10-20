"""Bounded ordinary arrivals; no ROS publishers or real motor commands."""
import ast
import json
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest
from control_center_joint_control.joint_definitions import ALL_JOINTS, JOINT_LIMITS, GROUP_ORDER, GROUPS
from control_center_joint_control.joint_bridge import JointBridge
from autolife_shelf_grasp_306.action_pose_arrival import settled_arm_arrival, PoseEnvelope


def fixture():
    goal={n:(JOINT_LIMITS[n][0]+JOINT_LIMITS[n][1])/2. for n in ALL_JOINTS}
    current=dict(goal);current['Joint_Left_Wrist_Upper']+=1.5
    start=dict(goal);start['Joint_Left_Wrist_Upper']+=10.
    frame=dict(joints_deg=goal,transition_sec=1.,hold_sec=.2,easing='smooth')
    state=dict(record=dict(groups=list(GROUP_ORDER),loop_count=1,speed_percent=100,
                           keyframes=[frame,deepcopy(frame)]),name='test',frame_index=0,loop_index=0,
               start_values=start,commanded_values=dict(goal),stage='transition',progress=1.,
               frame_started=0.,best_error_deg=10.,last_progress_at=0.,frame_deadline=16.,joint_names=list(goal))
    b=NS(_action_state=state,_current=current,_controller_target=dict(goal),
         _details={n:dict(speed=0.,communication_lost=False) for n in goal},
         _hardware_enabled=True,_hardware_ready=True,_leg_waist_enabled=True,_feedback_monotonic=10.,_preview={})
    params=dict(goal_tolerance_deg=1.,gripper_goal_tolerance_deg=2.,action_progress_epsilon_deg=.2,action_stall_timeout_sec=30.)
    b.get_parameter=lambda n:NS(value=params[n]);b._is_feedback_fresh=lambda:True
    return b,goal


def geometry(actual,target):return dict(displacement_m=.003,tool_rotation_deg=1.5)


def stable(b,goal,check=geometry,start=10.):
    result=None
    for i in range(7):
        b._feedback_monotonic=start+i*.1
        result=settled_arm_arrival(b,goal,check)
    return result


@pytest.mark.parametrize('error',[1.01,1.5,2.,-1.5])
def test_bounded_stationary_arms_accepted(error):
    b,goal=fixture();b._current['Joint_Left_Wrist_Upper']=goal['Joint_Left_Wrist_Upper']+error
    before=deepcopy(b._action_state['record'])
    receipt=stable(b,goal)
    assert receipt and receipt['kind']=='settled_arm_tolerance'
    assert b._action_state['record']==before


@pytest.mark.parametrize('case',['large','historical_contact','leg','neck','jaw','opening_now','closing_now',
    'opening_next','closing_next','loop_opening','moving','lost','stale','disabled','not_ready','waist_locked',
    'command_pending','unfinished','nan','out_of_limit','large_geometry','large_rotation','invalid_geometry'])
def test_unsafe_or_unverified_cases_keep_original_arrival(case):
    b,goal=fixture();check=geometry;state=b._action_state
    if case=='large':b._current['Joint_Left_Wrist_Upper']=goal['Joint_Left_Wrist_Upper']+2.001
    if case=='historical_contact':b._current['Joint_Left_Shoulder_Outer']=goal['Joint_Left_Shoulder_Outer']+2.214967
    if case=='leg':b._current['Joint_Knee']+=1.01
    if case=='neck':b._current['Joint_Neck_Pitch']+=1.01
    if case=='jaw':b._current['Joint_Left_Gripper']+=2.01
    if case=='opening_now':state['start_values']['Joint_Left_Gripper']+=30.
    if case=='closing_now':state['start_values']['Joint_Left_Gripper']-=30.
    if case=='opening_next':state['record']['keyframes'][1]['joints_deg']['Joint_Left_Gripper']-=30.
    if case=='closing_next':state['record']['keyframes'][1]['joints_deg']['Joint_Left_Gripper']+=30.
    if case=='loop_opening':
        state['frame_index']=1;state['record']['loop_count']=2
        state['record']['keyframes'][0]['joints_deg']=dict(goal)
        state['record']['keyframes'][0]['joints_deg']['Joint_Left_Gripper']-=30.
    if case=='moving':b._details['Joint_Left_Elbow']['speed']=2.01
    if case=='lost':b._details['Joint_Left_Elbow']['communication_lost']=True
    if case=='stale':b._is_feedback_fresh=lambda:False
    if case=='disabled':b._hardware_enabled=False
    if case=='not_ready':b._hardware_ready=False
    if case=='waist_locked':b._leg_waist_enabled=False
    if case=='command_pending':state['commanded_values']['Joint_Left_Wrist_Upper']+=.1
    if case=='unfinished':state['progress']=.99
    if case=='nan':b._current['Joint_Left_Elbow']=float('nan')
    if case=='out_of_limit':
        goal['Joint_Left_Wrist_Upper']=JOINT_LIMITS['Joint_Left_Wrist_Upper'][1]
        state['commanded_values']['Joint_Left_Wrist_Upper']=goal['Joint_Left_Wrist_Upper']
        b._current['Joint_Left_Wrist_Upper']=goal['Joint_Left_Wrist_Upper']+1.5
    if case=='large_geometry':check=lambda *_:dict(displacement_m=.00501,tool_rotation_deg=1.)
    if case=='large_rotation':check=lambda *_:dict(displacement_m=.003,tool_rotation_deg=2.01)
    if case=='invalid_geometry':check=lambda *_:dict(displacement_m=float('nan'),tool_rotation_deg=1.)
    assert stable(b,goal,check) is None


def test_geometry_exception_and_repeated_feedback_fail_closed():
    b,goal=fixture()
    def broken(*_):raise ValueError('URDF unavailable')
    assert stable(b,goal,broken) is None
    b,goal=fixture()
    for _ in range(100):assert settled_arm_arrival(b,goal,geometry) is None


def test_right_arm_and_original_precise_result():
    b,goal=fixture();b._current=dict(goal)
    assert stable(b,goal) is None  # Original predicate owns exact arrivals.
    b._current['Joint_Right_Wrist_Upper']+=1.5
    assert stable(b,goal)


def test_missing_geometry_does_not_authorize_continuation():
    b,goal=fixture()
    assert stable(b,goal,lambda *_: {}) is None


def test_stability_drift_gap_and_goal_change_reset():
    for mode in ('drift','gap','goal'):
        b,goal=fixture();assert stable(b,goal)
        b._feedback_monotonic=10.7
        if mode=='drift':b._current['Joint_Left_Wrist_Upper']+=.21
        if mode=='gap':b._feedback_monotonic=11.
        if mode=='goal':goal['Joint_Left_Elbow']+=.01
        assert settled_arm_arrival(b,goal,geometry) is None


def receipt_class(tmp_path):
    import autolife_shelf_grasp_306.action_studio as module
    tree=ast.parse(Path(module.__file__).read_text())
    cls=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='ReceiptBridge')
    scope=dict(JointBridge=JointBridge,self=NS(owner=NS(run=tmp_path)),time=time,json=json,
               __name__=module.__name__,__package__=module.__package__)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(module.__file__),'exec'),scope)
    return scope['ReceiptBridge']


def test_receipt_holds_only_affected_arm_and_next_frame_starts_measured(tmp_path):
    b,goal=fixture();cls=receipt_class(tmp_path)
    # Avoid constructing a ROS node. All methods used below operate on this fake.
    b._arrival_envelope=NS(check=geometry)
    events=[];sent=[];b._whole_body_pub='mock';b._event=lambda *a:events.append(a)
    b._record_action_event=lambda *a:cls._record_action_event(b,*a)
    b._publish_vendor=lambda pub,payload:sent.append((pub,payload))
    with patch.object(JointBridge,'_action_target_reached',return_value=False):
        # The method uses zero-arg super(), so create an uninitialized subclass
        # instance while bypassing all ROS Node initialization.
        bridge=object.__new__(cls);bridge.__dict__.update(b.__dict__)
        for i in range(7):
            bridge._feedback_monotonic=10.+i*.1
            accepted=bridge._action_target_reached(goal)
            if accepted:break
        assert accepted and len(sent)==1
        payload=sent[0][1]
        assert not any('gripper' in k for k in payload)
        assert payload[GROUPS['left_arm']['command_key']][5]==bridge._current['Joint_Left_Wrist_Upper']
        bridge._prepare_action_frame=lambda *a:None
        before=deepcopy(bridge._action_state['record'])
        bridge._advance_action_frame(11.)
        assert bridge._action_state['start_values']==bridge._current
        assert bridge._action_state['record']==before
        assert bridge._action_state['frame_index']==1


def test_real_urdf_envelope_if_available():
    path=Path('/home/ubuntu/ros2_ws/src/autolife_shelf_grasp_306/config/grasp.yaml')
    if not path.exists():pytest.skip('robot URDF only')
    import yaml
    env=PoseEnvelope(yaml.safe_load(path.read_text())['urdf_path'])
    b,goal=fixture()
    zero=env.check(goal,goal)
    assert zero['displacement_m']<1e-8 and zero['tool_rotation_deg']<1e-5
    q=dict(goal);q['Joint_Left_Shoulder_Outer']+=10.
    assert env.check(q,goal)['displacement_m']>.005
