from types import SimpleNamespace as NS
import pytest
from autolife_shelf_grasp_306.teaching_output import publish,command_lead
from control_center_joint_control.joint_definitions import GROUP_ORDER,JOINT_LIMITS
from test_teaching import pose


def fixture():
    q=pose(Joint_Right_Shoulder_Inner=20,Joint_Right_Elbow=-90)
    sent=[];n=NS(_current=q,_controller_target=q.copy(),teaching_sync_verified=True,
        _whole_body_pub='body',_gripper_pub='jaw',_publish_vendor=lambda p,d:sent.append((p,d)),
        get_parameter=lambda k:NS(value=30. if 'gripper' in k else 5.))
    state=dict(record=dict(groups=list(GROUP_ORDER)),commanded_values=q.copy())
    return n,state,sent


def test_consistent_per_joint_streaming_and_reverse():
    n,state,sent=fixture();target=n._current.copy()
    target['Joint_Right_Shoulder_Inner']=-20.;target['Joint_Right_Gripper']=100.
    publish(n,state,target)
    assert state['commanded_values']['Joint_Right_Shoulder_Inner']==8.
    assert state['commanded_values']['Joint_Right_Gripper']==100.
    assert command_lead(n,'Joint_Neck_Pitch')==5.
    target['Joint_Right_Shoulder_Inner']=30.;publish(n,state,target)
    assert state['commanded_values']['Joint_Right_Shoulder_Inner']==30.
    assert len(sent)==4


@pytest.mark.parametrize('fault',['no_sync','invalid_target','invalid_feedback','limit'])
def test_all_validation_precedes_publication(fault):
    n,state,sent=fixture();target=n._current.copy();joint='Joint_Right_Elbow'
    if fault=='no_sync':n.teaching_sync_verified=False
    elif fault=='invalid_target':target[joint]=float('nan')
    elif fault=='invalid_feedback':n._current[joint]=float('nan')
    else:target[joint]=JOINT_LIMITS[joint][1]+.01
    with pytest.raises(ValueError):publish(n,state,target)
    assert not sent


@pytest.mark.parametrize('limit,success',[(5.,False),(12.,True)])
def test_recorded_loop_with_motor_lag_not_instantaneous_feedback(tmp_path,monkeypatch,limit,success):
    import numpy as np
    from autolife_shelf_grasp_306.teaching_track import NAMES,make_draft
    from autolife_shelf_grasp_306.teaching_motion import compile_action
    from autolife_shelf_grasp_306.teaching_trajectory import plan
    from autolife_shelf_grasp_306.teaching_replay import advance
    from control_center_joint_control.joint_definitions import GROUPS,group_joint_names
    from test_recorded_timing import Controller
    monkeypatch.setattr('autolife_shelf_grasp_306.teaching_output.ARM_LEAD_DEG',limit)
    t=np.arange(0.,8.0001,.02);entry=pose(Joint_Right_Shoulder_Inner=20)
    q=np.tile([entry[n] for n in NAMES],(len(t),1))
    q[:,NAMES.index('Joint_Right_Shoulder_Inner')]-=22.5*(1-np.cos(2*np.pi*np.minimum(t,6.)/6.))
    d=make_draft(tmp_path,'lag-regression',t,q);record,segments,_=plan(compile_action(d),entry,tmp_path)
    clock=[0.];monkeypatch.setattr('time.monotonic',lambda:clock[0])
    c=Controller(record,segments,entry,clock);c._controller_target=dict(entry)
    c._action_state.update(loop_index=0,start_values=dict(entry));c._details={n:dict(speed=0.,communication_lost=False) for n in entry}
    c._whole_body_pub='body';c._gripper_pub='jaws';command=dict(entry)
    def vendor(pub,payload):
        for group in ('neck','left_arm','right_arm','leg_waist'):
            if GROUPS[group]['command_key'] in payload:command.update(zip(group_joint_names(group),payload[GROUPS[group]['command_key']]))
        for side in ('Left','Right'):
            key=side.lower()+'_gripper_target_joints_position'
            if key in payload:command['Joint_'+side+'_Gripper']=payload[key][0]
    c._publish_vendor=vendor;c._publish_action_targets=lambda state,desired,frame:publish(c,state,desired)
    while c._action_state and clock[0]<8.05:
        c._feedback_monotonic=clock[0];advance(c)
        for n in entry:
            step=3.*(command[n]-c._current[n])*.02;c._current[n]+=step;c._details[n]['speed']=step/.02
        clock[0]+=.02
    if success:assert c.outcome and c.outcome[1] and not c.reason
    else:assert '跟随不足' in c.reason and clock[0]<8.
