"""全身动作在补充踝膝使能后等待真实反馈，再开始插值，不反复使能。"""
import math,time
from control_center_joint_control.joint_definitions import ALL_JOINTS,BODY_JOINTS


def ready(node,session):
    now=time.monotonic();node._last_operator_activity=now
    gate=getattr(session,'lower_ready_gate',None)
    if gate is None:return True
    lost=node._communication_lost_joints(ALL_JOINTS)
    if now-gate['started']>6.:
        raise ValueError('补充腰腿使能后反馈未稳定，未开始动作：'+('、'.join(lost) if lost else '位置/速度反馈未稳定'))
    stamp=float(node._feedback_monotonic)
    if not node._is_feedback_fresh() or stamp<=gate['started'] or lost:
        gate['stable']=None;return False
    for n in BODY_JOINTS:
        actual=float(node._current[n]);speed=float(node._details[n].get('speed',float('nan')))
        if not math.isfinite(actual) or abs(actual-gate['anchor'][n])>1.:
            raise ValueError('补充腰腿使能期间姿态改变，未开始动作：'+n)
        if not math.isfinite(speed) or abs(speed)>2.:
            gate['stable']=None;return False
    stable=gate.get('stable')
    if stable is None or stamp<stable['last'] or stamp-stable['last']>.2:
        gate['stable']=dict(start=stamp,last=stamp,count=1);return False
    if stamp>stable['last']:stable['last']=stamp;stable['count']+=1
    if stable['count']>=5 and stamp-stable['start']>=1.:
        session.lower_ready_gate=None;return True
    return False
