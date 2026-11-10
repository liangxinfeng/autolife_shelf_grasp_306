"""Check a planned start pose without conflating jaw settling with arm motion."""
import math
from collections import deque
from .arm_selection import gripper_name
from .feedback_state import FeedbackNotReady

# Same loaded-jaw bound as the existing placement/observation provenance check.
# Arm, body, neck and empty-jaw tolerances are deliberately unchanged.
HELD_JAW_TOLERANCE_DEG=8.0


def await_start_pose(snapshot,poll,feedback_stamp,source,stage,clock,on_report,
                     timeout=2.,arm='left'):
    """使能/SYNC交接时等待原参考收敛；不改目标、不扩大0.5°通过门限。

    单帧轻微超差只进入有界等待，期间没有轨迹输出。连续新反馈满足
    原起点门限且静止0.15秒才通过；持续偏离、缺反馈、取消仍停止。
    """
    deadline=clock()+timeout;window=deque();last_stamp=None;last_report=-float('inf')
    last_reason='等待新的关节反馈'
    while True:
        poll();now=clock();stamp=feedback_stamp()
        try:current=snapshot()
        except FeedbackNotReady as exc:
            window.clear();last_reason=str(exc)
        else:
            fresh=(stamp is not None and math.isfinite(stamp) and 0<=now-stamp<=.3
                   and stamp==feedback_stamp())
            if fresh and (last_stamp is None or stamp>last_stamp):
                if last_stamp is not None and stamp-last_stamp>.15:window.clear()
                last_stamp=stamp
                report=start_pose_diagnostic(current,source,stage,None,arm=arm)
                # 明显的真实姿态变化不作为使能微调等待。
                if any(abs(c['delta_deg'])>3. for c in report['changes']):
                    on_report(report);raise ValueError(report['message'])
                if report['ok']:
                    window.append((stamp,dict(current)))
                    while len(window)>1 and stamp-window[1][0]>=.15:window.popleft()
                    stable=(len(window)>=3 and stamp-window[0][0]>=.15 and
                        all(max(q[n] for _,q in window)-min(q[n] for _,q in window)<=.15 for n in source))
                    report['stable']=stable
                    if stable:on_report(report);return current
                    last_reason='位置已进入规划起点范围，等待稳定'
                else:
                    window.clear();last_reason=report['message']
                report['waiting_settle']=True
                if now-last_report>=.1:on_report(report);last_report=now
            elif not fresh:
                window.clear();last_reason='关节反馈过期或无效'
        if now>=deadline:
            raise ValueError(stage+'：原规划起点未在准备期限内稳定，未开始抓取；'+last_reason)


def start_pose_diagnostic(current, source, stage, kind, arm='left'):
    if not source:
        raise ValueError(stage+'：缺少规划起始姿态')
    changes=[]
    for name, reference in source.items():
        if name not in current:
            raise ValueError(stage+'：缺少关节反馈 '+name)
        actual,reference=float(current[name]),float(reference)
        if not math.isfinite(actual) or not math.isfinite(reference):
            raise ValueError(stage+'：关节反馈或起始姿态无效 '+name)
        # Loaded contact is validated separately from free-space positioning.
        # The executor additionally requires fresh, stable feedback before motion.
        loaded_jaw=kind in ('place','observe') and name==gripper_name(arm)
        limit=HELD_JAW_TOLERANCE_DEG if loaded_jaw else .5
        changes.append(dict(joint=name,reference_deg=reference,actual_deg=actual,
                            delta_deg=actual-reference,tolerance_deg=limit))
    violations=[c for c in changes if abs(c['delta_deg'])>c['tolerance_deg']]
    violations.sort(key=lambda c:abs(c['delta_deg'])/c['tolerance_deg'],reverse=True)
    message=''
    if violations:
        v=violations[0]
        label='持瓶夹爪开度改变' if v['joint']==gripper_name(arm) and kind in ('place','observe') else '姿态改变'
        message=(f"{stage}{label}：{v['joint']} 偏差{abs(v['delta_deg']):.2f}°"
                 f"（允许{v['tolerance_deg']:.2f}°），请检查并重新规划")
    return dict(stage=stage,kind=kind,ok=not violations,message=message,
                changes=changes,violations=violations)


class LoadedJawSettling:
    """A bounded fresh-feedback window, never rebasing the original reference."""
    def __init__(self):
        self.samples=deque();self.last_stamp=None

    def add(self,stamp,now,angle):
        if stamp is None or not all(math.isfinite(v) for v in (stamp,now,angle)) or now-stamp>.3 or stamp>now:
            raise ValueError('持物确认反馈过期或无效')
        if self.last_stamp is not None and stamp<=self.last_stamp:return False
        self.last_stamp=stamp;self.samples.append((stamp,angle))
        while len(self.samples)>1 and stamp-self.samples[1][0]>=.2:self.samples.popleft()
        return (len(self.samples)>=5 and stamp-self.samples[0][0]>=.2 and
                max(v for _,v in self.samples)-min(v for _,v in self.samples)<=.5)


def await_loaded_pose(snapshot,poll,feedback_stamp,source,stage,kind,clock,on_report,timeout=1.5,arm='left'):
    """先排空新反馈再确认持物；切换瞬间的旧帧不等于持续通信故障。

    参考姿态、0.3秒新鲜度及稳定窗口均不放宽。等待期间不输出运动，
    反馈持续缺失、真实姿态偏离、取消均停止；超时包含可诊断的反馈年龄。
    """
    window=LoadedJawSettling();deadline=clock()+timeout;last_report=-float('inf');age=None
    while True:
        # 使能回调可能刚完成长耗时操作；必须先服务订阅回调，不能先拒绝旧帧。
        poll()
        stamp=feedback_stamp()
        try:current=snapshot()
        except FeedbackNotReady as exc:
            now=clock();window=LoadedJawSettling()
            if now-last_report>=.1:
                on_report(dict(stage=stage,kind=kind,ok=False,waiting_feedback=True,
                    message=str(exc),feedback_reason=exc.reason,unavailable_joints=exc.joints))
                last_report=now
            if now>=deadline:
                raise ValueError(stage+'：持物反馈未在准备期限内恢复，保持停止；'+str(exc)) from exc
            continue
        stamp_after=feedback_stamp();now=clock()
        age=now-stamp if stamp is not None and math.isfinite(stamp) else None
        fresh=(stamp==stamp_after and age is not None and 0<=age<=.3)
        if not fresh:
            window=LoadedJawSettling()
            if now>=deadline:
                detail='缺失或无效' if age is None else f'{age:.3f}秒'
                raise ValueError(stage+'：等待新鲜持物反馈超时，反馈年龄'+detail+'；保持停止，未开始放置')
            continue
        report=start_pose_diagnostic(current,source,stage,kind,arm=arm)
        report['feedback_age_sec']=age
        report['feedback_stamp_monotonic']=stamp
        if window.last_stamp is not None and stamp-window.last_stamp>.3:
            window=LoadedJawSettling()
        stable=window.add(stamp,now,float(current[gripper_name(arm)]))
        # 报告写盘不应在每个5ms轮询内重复进行，更不能先写盘再判断帧时间。
        if stable or not report['ok'] or now-last_report>=.1:
            on_report(report);last_report=now
        if not report['ok']:raise ValueError(report['message'])
        if stable:return current
        if now>=deadline:raise ValueError(stage+'：持物夹爪反馈未稳定，已保持停止，请检查物品是否松动')
