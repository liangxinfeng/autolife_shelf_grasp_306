"""保持式接管前的异步只读刷新；不使能、不发布关节目标、不跨线程spin。"""
import json,math,time
from .feedback_refresh import GROUPS


class PoseFeedbackRead:
    def __init__(self,node):
        from std_srvs.srv import Trigger
        self.node=node;self.request=Trigger.Request
        self.clients={g:node.create_client(Trigger,f'/refresh_grasp_feedback_0_345_{g}') for g in GROUPS}
        self.started=time.monotonic();self.futures=None;self.values=None;self.received=None
    def poll(self):
        now=time.monotonic()
        if now-self.started>6:raise ValueError('接管前真实位置刷新超时，未使能或发送目标')
        if self.futures is None:
            if not all(c.service_is_ready() for c in self.clients.values()):return None
            self.futures={g:c.call_async(self.request()) for g,c in self.clients.items()};return None
        if not all(f.done() for f in self.futures.values()):return None
        if self.values is None:
            values={}
            for group,f in self.futures.items():
                r=f.result()
                if r is None or not r.success:raise ValueError('真实位置刷新失败：'+group)
                d=json.loads(r.message);v=d.get('positions',{})
                if d.get('protocol')!=2 or d.get('group')!=group or not v or values.keys()&v.keys():raise ValueError('真实位置刷新格式无效：'+group)
                if not all(math.isfinite(float(x)) for x in v.values()):raise ValueError('真实位置反馈非有限值')
                values.update({n:float(x) for n,x in v.items()})
            from control_center_joint_control.joint_definitions import ALL_JOINTS
            if set(values)!=set(ALL_JOINTS):raise ValueError('真实位置刷新缺少完整23轴')
            self.values=values;self.received=now
            return None
        n=self.node
        if not n._is_feedback_fresh() or n._feedback_monotonic<=self.received:return None
        if any(abs(n._current.get(k,float('inf'))-v)>(2. if 'Gripper' in k else .5) for k,v in self.values.items()):return None
        return dict(n._current)
    def close(self):
        for c in self.clients.values():self.node.destroy_client(c)
        self.clients={}
