"""统一区分暂未就绪的反馈与非法数据；运动中仍立即抛出并停止。

只有明确的准备/模式切换等待器可以捕获 FeedbackNotReady。
此模块不输出目标、不使能、不清错，也不修改原厂离线标志。
"""
import math
import time
from autolife_handeye_calibration_306.core import BODY,NECK,ARMS

NAMES=BODY+NECK+ARMS['left']+ARMS['right']+['Joint_Left_Gripper','Joint_Right_Gripper']


class FeedbackNotReady(ValueError):
    def __init__(self,message,*,reason,joints=()):
        super().__init__(message)
        self.reason=reason;self.joints=list(joints)


def checked_snapshot(node):
    if not node._is_feedback_fresh():
        raise FeedbackNotReady('关节反馈过期，不能记录或调整',reason='stale')
    q=dict(node._current)
    lost=node._communication_lost_joints(NAMES)
    if lost:
        raise FeedbackNotReady('关节通信离线：'+', '.join(lost),reason='offline',joints=lost)
    missing=[n for n in NAMES if n not in q]
    if missing:
        raise FeedbackNotReady('全身关节反馈不完整：'+', '.join(missing),reason='incomplete',joints=missing)
    if not all(math.isfinite(q[n]) for n in NAMES):raise ValueError('关节反馈含非有限数值')
    return q


def await_ready_snapshot(snapshot,poll,stage,clock=time.monotonic,timeout=1.5,on_report=lambda report:None):
    """动作开始前服务新反馈；健康路径不增加等待，不捕获运动/数据异常。"""
    deadline=clock()+timeout;last_report=-float('inf')
    while True:
        poll()
        try:return snapshot()
        except FeedbackNotReady as exc:
            now=clock()
            if now-last_report>=.1:
                on_report(dict(stage=stage,ok=False,waiting_feedback=True,message=str(exc),
                               feedback_reason=exc.reason,unavailable_joints=exc.joints))
                last_report=now
            if now>=deadline:
                raise ValueError(stage+'：反馈未在准备期限内恢复，未开始运动；'+str(exc)) from exc
