"""手动示教沿用300的单次保持会话：原厂负责使能，页面内复用，不重复切模式。"""
import json,time,math
from std_msgs.msg import String
from std_srvs.srv import SetBool
from control_center_joint_control.joint_definitions import BODY_JOINTS
from .sync_session import PositionSyncSession


class ManualHoldSession(PositionSyncSession):
    def __init__(self,node,**kwargs):
        super().__init__(node,**kwargs)
        self.vendor_state=None;self.vendor_stamp=0.;self.verified=False
        self.subscription=node.create_subscription(String,'/robot_state_topic_0_345',self.on_state,10)

    def on_state(self,msg):
        try:
            value=json.loads(msg.data)['state']
            if type(value) is not int:raise ValueError('state')
            self.vendor_state=value;self.vendor_stamp=time.monotonic()
        except (ValueError,TypeError,KeyError):self.vendor_state=None

    def begin(self):
        n=self.node;self.started=time.monotonic();self.future=None;self.accepted_at=None;self.stable=None
        if not n._is_feedback_fresh():raise ValueError('等待本机关节反馈')
        from .pose_feedback_read import PoseFeedbackRead
        self.reader=PoseFeedbackRead(n);self.anchor=None
        # 仅软件标记准备中，不调用旧的逐组电机模式/使能流程，不下发运动。
        n._hardware_enabled=True;n._hardware_ready=False;n._hardware_prepare=None
        n._hardware_unlock_mode='without_reset';n._leg_waist_enabled=False
        n._last_operator_activity=self.started;n._preview=dict(n._current)

    def poll(self):
        n=self.node;now=time.monotonic()
        if not n._hardware_enabled or not n._is_feedback_fresh():raise ValueError('手动保持准备期间反馈或控制权失效')
        n._last_operator_activity=now
        if self.future is None:
            if not self.client.service_is_ready() or self.vendor_state is None or now-self.vendor_stamp>1.:
                if now-self.started>2.:raise ValueError('原厂保持接口或状态未就绪，未下发目标')
                return False
            if self.vendor_state!=0:raise ValueError('原厂关节已有控制会话，请先停止其他遥操或复位')
            if self.anchor is None:
                self.anchor=self.reader.poll()
                if self.anchor is None:return False
                self.reader.close();self.reader=None
                self.started=now
            self.requested=True;self.future=self.client.call_async(SetBool.Request(data=True));return False
        if not self.future.done():
            if now-self.started>12.:raise ValueError('原厂保持使能超时，已停止此次调节')
            return False
        if self.accepted_at is None:
            result=self.future.result()
            if not result.success:raise ValueError('原厂保持使能拒绝：'+result.message)
            # 原厂已确认保持式 ASYNC→HOME→SYNC；此时才认定硬件就绪。
            self.accepted_at=now;self.active=True;n._hardware_ready=True
        super().tick()
        if now-self.accepted_at>4.:raise ValueError('保持使能后关节反馈未稳定')
        if self.vendor_state!=2 or now-self.vendor_stamp>1.:
            self.stable=None;return False
        if n._communication_lost_joints(BODY_JOINTS):self.stable=None;return False
        for joint in BODY_JOINTS:
            value=float(n._current[joint]);speed=float(n._details[joint]['speed'])
            if not math.isfinite(value) or abs(value-self.anchor[joint])>1.:raise ValueError('保持使能期间姿态改变：'+joint)
            if not math.isfinite(speed) or abs(speed)>2.:self.stable=None;return False
        stamp=float(n._feedback_monotonic)
        if self.stable is None or stamp<self.stable['last'] or stamp-self.stable['last']>.15:
            self.stable=dict(start=stamp,last=stamp,count=1);return False
        if stamp>self.stable['last']:self.stable['last']=stamp;self.stable['count']+=1
        self.verified=self.stable['count']>=4 and stamp-self.stable['start']>=.3
        return self.verified

    def tick(self):
        super().tick()
        if self.active and self.verified:
            if self.vendor_state!=2 or time.monotonic()-self.vendor_stamp>1.:
                raise ValueError('原厂保持会话已退出或状态反馈中断，已停止调节')

    def close(self,preserve_grippers=False):
        try:return super().close(preserve_grippers=preserve_grippers)
        finally:
            self.verified=False
            if getattr(self,'reader',None):self.reader.close();self.reader=None
            if self.subscription is not None:self.node.destroy_subscription(self.subscription);self.subscription=None


def keep_idle(engine):
    """无动作时只维护已有保持会话；不发送新的姿态、不开新会话。"""
    if not getattr(engine,'manual_session',False):return
    try:
        engine.sync_session.tick()
        if engine.bridge._communication_lost_joints(BODY_JOINTS):raise ValueError('保持期间关节通信中断')
        engine.bridge._last_operator_activity=time.monotonic()
    except Exception as exc:
        engine.stop()
        try:engine.release()
        except Exception:pass
        engine.completed.emit('manual-pose-idle',False,str(exc))
