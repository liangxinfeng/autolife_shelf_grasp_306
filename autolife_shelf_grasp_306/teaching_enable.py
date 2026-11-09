"""Bounded feedback handshake after enabling, before any taught motion.

The vendor can briefly mark joints lost while servicing the enable request.
Never retry enabling, clear errors, or publish motion while waiting. Actual
execution retains immediate communication-loss cancellation.
"""
import math
import time
from control_center_joint_control.joint_definitions import BODY_JOINTS

RECOVERY_TIMEOUT_SEC=3.0
ONLINE_STABLE_SEC=.4


def wait_for_enable_feedback(bridge):
    """True: this preparation tick is handled; False: use normal preparation."""
    prepare=bridge._hardware_prepare
    if (not bridge._hardware_enabled or bridge._hardware_ready or prepare is None
            or prepare.get('stage')!='verify_hold'
            or getattr(bridge,'_hardware_unlock_mode',None)!='without_reset'):
        return False
    if not bridge._is_feedback_fresh():
        bridge._lock('使能确认期间关节反馈已过期；未开始动作，已锁定')
        return True
    lost=bridge._communication_lost_joints(BODY_JOINTS)
    window=prepare.get('enable_feedback_recovery')
    if not lost and window is None:
        # 保持目标不应与实测相差大角度；禁止向旧缓存角度追赶45秒。
        changed=[name for name in BODY_JOINTS
                 if abs(float(bridge._current.get(name,float('inf')))-
                        float(prepare['hold_targets'].get(name,float('-inf'))))>10.]
        if changed:
            bridge._lock('使能前后位置反馈不一致，未开始复位：'+'、'.join(changed)+
                         '；请核对原厂实时反馈，未自动追赶旧角度')
            return True
        return False
    now=time.monotonic()
    if now-prepare['started']>float(bridge.get_parameter('hardware_prepare_timeout_sec').value):
        bridge._lock('厂家电机位置模式/使能准备超时，已自动锁定')
        return True
    if window is None:
        window=dict(start=now,last=None,online_since=None,count=0)
        prepare['enable_feedback_recovery']=window
        bridge._record_action_event('enable_feedback_wait','使能期间短时失联，等待全部关节恢复；不下发动作或重复使能')
        bridge._event(f'使能确认：等待{len(lost)}个关节恢复通信，最长3秒，尚未播放动作','warn')
    if lost:
        window.update(last=None,online_since=None,count=0,recovered=False)
        prepare['stable_since']=None
        if now-window['start']>=RECOVERY_TIMEOUT_SEC:
            bridge._lock('使能后关节通信未在3秒内稳定恢复，未开始动作：'+'、'.join(lost))
        return True
    if window.get('recovered'):return False
    if now-window['start']>=RECOVERY_TIMEOUT_SEC:
        bridge._lock('使能后的在线反馈未在3秒内完成稳定确认；未开始动作，已锁定')
        return True
    stamp=float(bridge._feedback_monotonic)
    if not math.isfinite(stamp):
        bridge._lock('使能确认反馈时间无效；未开始动作，已锁定');return True
    position_limit=max(.1,float(bridge.get_parameter('hardware_reset_position_tolerance_deg').value))
    speed_limit=max(.1,float(bridge.get_parameter('hardware_reset_speed_tolerance_deg_s').value))
    for name in BODY_JOINTS:
        actual=float(bridge._current.get(name,float('nan')))
        goal=float(prepare['hold_targets'].get(name,float('nan')))
        speed=float(bridge._details.get(name,{}).get('speed',float('nan')))
        if not all(math.isfinite(v) for v in (actual,goal,speed)) or abs(actual-goal)>position_limit:
            bridge._lock('使能通信恢复时关节未保持原姿态，未开始动作：'+name)
            return True
        if abs(speed)>speed_limit:
            # Match native hold verification: a fresh speed spike resets the
            # settling window, it does not claim a permanent pose change.
            window.update(last=None,online_since=None,count=0)
            return True
    if window['last'] is None or stamp<window['last'] or stamp-window['last']>.15:
        window.update(last=stamp,online_since=stamp,count=1)
        return True
    if stamp>window['last']:
        window['last']=stamp;window['count']+=1
    if window['count']<4 or stamp-window['online_since']<ONLINE_STABLE_SEC:return True
    window['recovered']=True
    prepare['stable_since']=None
    prepare['next_publish']=now
    bridge._record_action_event('enable_feedback_recovered','全部身体关节在线且原姿态稳定0.4秒，继续原有使能稳定检查')
    return False
