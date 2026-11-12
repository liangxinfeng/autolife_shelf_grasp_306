"""普通姿态串：夹爪按规划时序执行，不以开口残差/速度判定结束。

这只是指令完成回执，不代表物体抓牢或已释放。身体到位检查在桥接器中执行；
通信失效、非法目标、未发到目标均不能通过此入口。
"""
import math,time
from control_center_joint_control.joint_definitions import GRIPPER_JOINTS,JOINT_LIMITS

GRIPPER_FINISH_WAIT_SEC=.5


def issued(node,state,segment,targets):
    if not node._hardware_enabled or not node._hardware_ready or not node._is_feedback_fresh():return False
    if state.get('progress',0.)<1.:return False
    for name,target in targets.items():
        if node._details.get(name,{}).get('communication_lost',True):return False
        if name not in GRIPPER_JOINTS:continue
        command=float(state['commanded_values'].get(name,float('nan')))
        lo,hi=JOINT_LIMITS[name]
        if not math.isfinite(float(target)) or not lo<=target<=hi or not math.isfinite(command):return False
        if abs(command-target)>.05:return False
    if segment.get('hold_grippers'):return True
    # 插值开合阶段已按原有速度/加速度约束发送完；额外等待一次有限的执行间隔。
    # 不读取夹爪开度或速度，不会因接触物品或量化残差无限等待。
    key=(state['loop_index'],state['frame_index'],tuple((n,targets[n]) for n in GRIPPER_JOINTS if n in targets))
    now=time.monotonic();old=state.get('jaw_command_finished')
    if old is None or old['key']!=key:
        state['jaw_command_finished']=dict(key=key,time=now);return False
    return now-old['time']>=GRIPPER_FINISH_WAIT_SEC
