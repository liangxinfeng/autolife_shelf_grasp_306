"""Recorded position streaming within V5's established arm command envelope.

The ordinary 5-degree incremental move profile is for point-to-point motion.
Continuous playback uses the V5 nominal 12-degree arm lead, not its optional
14/20-degree expansion. Body limits remain unchanged. Grippers use V5's
bounded target interface and a separate, timed response guard.
"""
import math
from control_center_joint_control.joint_definitions import GROUPS,GRIPPER_JOINTS,JOINT_LIMITS,group_joint_names

ARMS=frozenset(group_joint_names('left_arm')+group_joint_names('right_arm'))
ARM_LEAD_DEG=12.


def command_lead(node,name):
    if name in ARMS:return ARM_LEAD_DEG
    key='max_gripper_command_lead_deg' if name in GRIPPER_JOINTS else 'max_single_step_deg'
    return max(.1,float(node.get_parameter(key).value))


def publish(node,state,desired):
    """Validate and bound the entire packet before any publication."""
    if not getattr(node,'teaching_sync_verified',False):
        raise ValueError('连续轨迹未取得位置SYNC会话，拒绝下发')
    command={}
    for n,target in desired.items():
        actual=float(node._current[n]);target=float(target)
        if not math.isfinite(actual) or not math.isfinite(target):raise ValueError('轨迹指令或反馈非有限值：'+n)
        lo,hi=JOINT_LIMITS[n]
        if not lo<=target<=hi:raise ValueError('轨迹目标超出关节范围：'+n)
        if n in GRIPPER_JOINTS:
            command[n]=target
        else:
            lead=command_lead(node,n)
            command[n]=min(hi,max(lo,min(actual+lead,max(actual-lead,target))))
    groups=state['record']['groups'];payload={}
    for group in ('neck','left_arm','right_arm','leg_waist'):
        names=group_joint_names(group)
        values=[command[n] if group in groups else node._controller_target[n] for n in names]
        if not all(math.isfinite(float(v)) for v in values):raise ValueError('非执行关节保持值无效')
        payload[GROUPS[group]['command_key']]=values
    if any(g in groups for g in ('neck','left_arm','right_arm','leg_waist')):
        node._publish_vendor(node._whole_body_pub,payload)
    if 'grippers' in groups:
        node._publish_vendor(node._gripper_pub,dict(left_gripper_target_joints_position=[command['Joint_Left_Gripper']],
                                                   right_gripper_target_joints_position=[command['Joint_Right_Gripper']]))
    state['commanded_values'].update(command)
