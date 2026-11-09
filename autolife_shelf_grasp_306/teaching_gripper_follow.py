"""Response-window checks for recorded gripper motion, not arm-angle clipping.

V5 sends the bounded gripper target directly. Measured DM opening can change
over 2000 encoder units/s; a 30-unit lead cap prevents that recorded movement.
Allow bounded response latency, but reject persistent non-response or motion
outside the recent requested openings when tracking_required is True.
Recorded playback explicitly uses advisory tracking: it validates feedback and
limits, but a position mismatch alone cannot abort or extend the timeline.
"""
from collections import deque
import math
from control_center_joint_control.joint_definitions import GRIPPER_JOINTS,JOINT_LIMITS

RESPONSE_SEC=.35
OUTSIDE_SEC=.12
MOVING_MARGIN=5.


def check(node,state,desired,now,*,tracking_required=True):
    history=state.setdefault('recorded_gripper_history',{})
    for name in GRIPPER_JOINTS:
        if name not in desired:continue
        actual=float(node._current[name]);target=float(desired[name]);lo,hi=JOINT_LIMITS[name]
        if not math.isfinite(actual) or not math.isfinite(target) or not lo<=target<=hi:
            return '录制夹爪目标或反馈无效：'+name
        if actual<lo-MOVING_MARGIN or actual>hi+MOVING_MARGIN:
            return '录制夹爪实测超出有效范围：'+name
        jaw=history.setdefault(name,dict(samples=deque(),outside_since=None))
        samples=jaw['samples']
        if samples and now<samples[-1][0]:return '录制夹爪响应时钟倒退，已停止'
        if samples and now==samples[-1][0]:samples[-1]=(now,target)
        else:samples.append((now,target))
        # Keep one sample just before the cutoff, so a discretized observation
        # window never becomes shorter than the specified response allowance.
        while len(samples)>1 and samples[1][0]<=now-RESPONSE_SEC:samples.popleft()
        low=min(v for _,v in samples)-MOVING_MARGIN
        high=max(v for _,v in samples)+MOVING_MARGIN
        if low<=actual<=high:
            jaw['outside_since']=None
            jaw['tracking_warning']=''
            continue
        if jaw['outside_since'] is None:jaw['outside_since']=now
        if now-jaw['outside_since']>=OUTSIDE_SEC:
            message=(f'录制夹爪未跟随近期目标：{name}，实测{actual:.2f}，'
                     f'近期目标范围{low+MOVING_MARGIN:.2f}～{high-MOVING_MARGIN:.2f}')
            if tracking_required:return message+'；已停止，请检查是否受阻或反馈异常'
            jaw['tracking_warning']=message
            # At most one event per jaw per action, not a 50 Hz warning stream.
            # Continue validating the other jaw even after this advisory.
            if not jaw.get('warning_logged'):
                node._record_action_event('recorded_gripper_tracking_advisory',
                    message+'；仅记录，继续原时序播放；不代表夹取或释放成功')
                jaw['warning_logged']=True
    return ''
