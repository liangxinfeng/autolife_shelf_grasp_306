"""Nonblocking position-SYNC handoff for the taught-action executor.

Uses the same guarded hold-only lease as grasp execution and V5 teleoperation.
No reset, gain change, velocity mode or relaxation of motion limits.
"""
import math
import time
from std_srvs.srv import SetBool
from control_center_joint_control.joint_definitions import BODY_JOINTS
from .sync_session import PositionSyncSession


class TeachingSyncSession(PositionSyncSession):
    def begin(self):
        self.started=time.monotonic();self.future=None;self.accepted_at=None
        self.anchor=dict(self.node._current);self.stable=None

    def poll(self):
        now=time.monotonic();node=self.node
        if not node._hardware_ready or not node._is_feedback_fresh():
            raise ValueError('轨迹SYNC准备期间反馈或控制权失效，未播放动作')
        if self.future is None:
            if not self.client.service_is_ready():
                if now-self.started>2.:raise ValueError('轨迹位置SYNC入口不可用；未降级到普通慢速控制')
                return False
            self.requested=True;self.future=self.client.call_async(SetBool.Request(data=True))
            return False
        if not self.future.done():
            if now-self.started>12.:raise ValueError('轨迹位置SYNC获取超时，未播放动作')
            return False
        if self.accepted_at is None:
            response=self.future.result()
            if not response.success:raise ValueError('轨迹位置SYNC被拒绝：'+response.message)
            self.active=True;self.accepted_at=now;self.tick()
        self.tick()
        if now-self.accepted_at>4.:raise ValueError('SYNC切换后未确认原姿态稳定，未播放动作')
        if node._communication_lost_joints(BODY_JOINTS):
            self.stable=None;return False
        for n in BODY_JOINTS:
            actual=float(node._current[n]);speed=float(node._details[n]['speed'])
            if not math.isfinite(actual) or abs(actual-self.anchor[n])>1.:
                raise ValueError('SYNC切换时姿态改变，未播放动作：'+n)
            if not math.isfinite(speed) or abs(speed)>2.:
                self.stable=None;return False
        stamp=float(node._feedback_monotonic)
        if self.stable is None or stamp<self.stable['last'] or stamp-self.stable['last']>.15:
            self.stable=dict(start=stamp,last=stamp,count=1);return False
        if stamp>self.stable['last']:
            self.stable['last']=stamp;self.stable['count']+=1
        return self.stable['count']>=4 and stamp-self.stable['start']>=.3
