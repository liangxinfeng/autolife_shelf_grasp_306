"""真实任务与失败数据的无硬件回放：使用实际桥接器，但禁止创建ROS节点。"""
import json,math
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace as NS
import pytest
from control_center_joint_control.joint_bridge import JointBridge
from control_center_joint_control.joint_definitions import GROUPS,GRIPPER_JOINTS,group_joint_names
from autolife_shelf_grasp_306.pose_execution_bridge import make_bridge
from autolife_shelf_grasp_306.pose_jaw_timing import issued
from autolife_shelf_grasp_306.teaching_trajectory import plan
from test_teaching import pose


@pytest.mark.parametrize('fault',['none','large_residual','quantized_speed','offline','stale','disabled','not_sent','not_finished','invalid_goal'])
def test_jaw_command_receipt_without_opening_feedback(monkeypatch,fault):
    clock=[10.];monkeypatch.setattr('time.monotonic',lambda:clock[0])
    n='Joint_Right_Gripper';target={n:108.9777778}
    b=NS(_hardware_enabled=True,_hardware_ready=True,_is_feedback_fresh=lambda:True,
         _current={n:111.95349},_details={n:dict(communication_lost=False,speed=1.469)})
    s=dict(progress=1.,frame_index=1,loop_index=0,commanded_values=dict(target))
    if fault=='large_residual':b._current[n]=280.
    if fault=='quantized_speed':b._details[n]['speed']=7.346
    if fault=='offline':b._details[n]['communication_lost']=True
    if fault=='stale':b._is_feedback_fresh=lambda:False
    if fault=='disabled':b._hardware_enabled=False
    if fault=='not_sent':s['commanded_values'][n]=180.
    if fault=='not_finished':s['progress']=.9
    if fault=='invalid_goal':target[n]=float('nan')
    assert not issued(b,s,{},target)
    clock[0]+=.51
    assert issued(b,s,{},target) is (fault in ('none','large_residual','quantized_speed'))


def controller(record,entry,tmp_path,clock):
    r,segments,phases=plan(record,entry,tmp_path)
    context=NS(owner=NS(run=tmp_path),gripper_targets={},teaching_phases=phases)
    cls=make_bridge(JointBridge,context);b=object.__new__(cls)
    b._current=dict(entry);b._controller_target=dict(entry);b._preview=dict(entry)
    b._details={n:dict(speed=0.,communication_lost=False) for n in entry}
    b._feedback_monotonic=clock[0];b._hardware_enabled=b._hardware_ready=b._leg_waist_enabled=True
    b._whole_body_pub='body';b._gripper_pub='jaw';b._active_goals={};b._goal_sent_groups=set()
    b._is_feedback_fresh=lambda:True;b._communication_lost_joints=lambda names:[n for n in names if b._details[n]['communication_lost']]
    b._event=lambda *args:None
    params=dict(goal_tolerance_deg=.2,gripper_goal_tolerance_deg=2.,max_single_step_deg=2.,max_gripper_command_lead_deg=15.,command_update_rate_hz=50.,action_progress_epsilon_deg=.2,action_stall_timeout_sec=30.,action_timeout_factor=2.,action_timeout_margin_sec=12.)
    b.get_parameter=lambda n:NS(value=params[n]);b.teaching_segments=segments;b.outcome=None
    b._action_state=dict(record=r,name='whole-task',joint_names=list(entry),frame_index=0,loop_index=0,
        stage='transition',frame_started=clock[0],start_values=dict(entry),commanded_values=dict(entry),progress=0.)
    b._prepare_action_frame(b._action_state,clock[0]);b.sent=dict(entry)
    def publish(pub,payload):
        if pub=='body':
            for group in ('neck','left_arm','right_arm','leg_waist'):
                b.sent.update(zip(group_joint_names(group),payload[GROUPS[group]['command_key']]))
        else:
            b.sent['Joint_Left_Gripper']=payload['left_gripper_target_joints_position'][0]
            b.sent['Joint_Right_Gripper']=payload['right_gripper_target_joints_position'][0]
    b._publish_vendor=publish
    return b,phases


@pytest.mark.parametrize('mode',['observed','jaw_unfollowed','body_unfollowed','lost_jaw'])
def test_complete_saved_garbage_task_and_real_faults(tmp_path,monkeypatch,mode):
    data=json.loads((Path(__file__).parent/'fixtures/garbage_jaw_20261111.json').read_text())
    clock=[100.];monkeypatch.setattr('time.monotonic',lambda:clock[0])
    record=data['record'];entry=dict(record['keyframes'][0]['joints_deg']);entry['Joint_Right_Gripper']=112.0299879
    b,phases=controller(record,entry,tmp_path,clock)
    visited=set()
    for tick in range(14000):
        clock[0]=100.+tick*.02;b._feedback_monotonic=clock[0]
        if not b._action_state:break
        index=b._action_state['frame_index'];visited.add(index)
        for n,goal in b.sent.items():
            if n not in GRIPPER_JOINTS:
                if mode=='body_unfollowed' and index>=2 and n=='Joint_Right_Elbow':continue
                b._current[n]+=(goal-b._current[n])*.35
            elif mode!='jaw_unfollowed':
                offset=2.97571224 if n=='Joint_Right_Gripper' and goal<150 else -4.3 if n=='Joint_Right_Gripper' else 0.
                b._current[n]+=(max(10.,goal+offset)-b._current[n])*.35
            b._details[n]['speed']=[1.469,4.407,-1.47,7.346][tick%4] if n in GRIPPER_JOINTS else 0.
        if mode=='lost_jaw' and index>=2:
            b._details['Joint_Right_Gripper']['communication_lost']=True
            # 使用真实的基类锁定需要ROS事件依赖；此替身保留停止语义，不发布任何指令。
            b._lock=lambda reason:(setattr(b,'outcome',('whole-task',False,reason)),setattr(b,'_action_state',None))
        b._advance_action()
    assert b._action_state is None
    if mode in ('observed','jaw_unfollowed'):
        assert b.outcome and b.outcome[1]
        assert visited==set(range(len(phases)))
        assert any(x[1]=='夹爪开合 → 停顿' for x in phases)
    else:assert b.outcome and not b.outcome[1]


def test_actual_failure_snapshot_is_not_a_body_error(tmp_path,monkeypatch):
    data=json.loads((Path(__file__).parent/'fixtures/garbage_jaw_20261111.json').read_text());r=data['failure']
    assert max(abs(r['targets'][n]-r['measured'][n]) for n in r['targets'] if n not in GRIPPER_JOINTS)<.02
    clock=[100.];monkeypatch.setattr('time.monotonic',lambda:clock[0])
    b,phases=controller(data['record'],r['measured'],tmp_path,clock)
    index=next(i for i,p in enumerate(phases) if p[0]==3 and p[1]=='夹爪开合 → 停顿')
    s=b._action_state;s.update(frame_index=index,progress=1.,commanded_values=dict(r['commanded']))
    b._current=dict(r['measured']);b._details=deepcopy(r['details'])
    assert not b._action_target_reached(r['targets'])
    clock[0]+=.51
    assert b._action_target_reached(r['targets'])


def test_arm_envelope_does_not_reintroduce_held_jaw_feedback_gate():
    from test_pose_arrival import fixture,geometry
    from autolife_shelf_grasp_306.action_pose_arrival import settled_arm_arrival
    b,target=fixture();b.teaching_segments=[dict(start=dict(target))]
    n='Joint_Right_Gripper';b._current[n]=target[n]-80.;b._action_state['start_values'][n]=b._current[n]
    b._details[n]['speed']=7.346
    result=None
    for i in range(8):
        b._feedback_monotonic=10.+i*.1
        result=settled_arm_arrival(b,target,geometry,timed_grippers=True)
    assert result and result['kind']=='settled_arm_tolerance'
