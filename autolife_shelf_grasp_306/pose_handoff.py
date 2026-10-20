"""Check a planned start pose without conflating jaw settling with arm motion."""
import math
from collections import deque
from .arm_selection import gripper_name

# Same loaded-jaw bound as the existing placement/observation provenance check.
# Arm, body, neck and empty-jaw tolerances are deliberately unchanged.
HELD_JAW_TOLERANCE_DEG=8.0


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
    """No command output. Abort drift immediately; wait only for bounded settling."""
    window=LoadedJawSettling();deadline=clock()+timeout
    while True:
        current=snapshot();report=start_pose_diagnostic(current,source,stage,kind,arm=arm)
        on_report(report)
        if not report['ok']:raise ValueError(report['message'])
        if window.add(feedback_stamp(),clock(),float(current[gripper_name(arm)])):return current
        if clock()>=deadline:raise ValueError(stage+'：持物夹爪反馈未稳定，已保持停止，请检查物品是否松动')
        poll()
