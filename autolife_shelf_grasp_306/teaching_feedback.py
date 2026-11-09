"""Passive subscriber, deliberately no robot command publishers or services."""
import json,time,math
from collections import deque
from control_center_joint_control.joint_definitions import ALL_JOINTS,BODY_JOINTS,FEEDBACK_KEYS,GRIPPER_FEEDBACK_KEYS,JOINT_LIMITS,group_joint_names


def parse_feedback(data):
    values={};speeds={};lost=[]
    fields=[(key,group_joint_names(group)) for group,key in FEEDBACK_KEYS.items()]
    fields += [(key,[name]) for name,key in GRIPPER_FEEDBACK_KEYS.items()]
    for key,names in fields:
        state=data[key];positions=state['position'];velocity=state.get('speed',[])
        if len(positions)!=len(names):raise ValueError('关节反馈长度不匹配')
        for i,n in enumerate(names):
            value=float(positions[i])
            if not math.isfinite(value):raise ValueError('关节反馈含无效数值')
            values[n]=value;speeds[n]=float(velocity[i]) if i<len(velocity) else None
            comm=state.get('communication_lost',[])
            if i<len(comm) and comm[i]:lost.append(n)
    return values,speeds,lost


class FeedbackState:
    def __init__(self):self.values={};self.speeds={};self.lost=[];self.stamp=0.;self.history=deque(maxlen=120);self.error='等待关节反馈'
    def accept(self,payload,now=None):
        now=time.monotonic() if now is None else now
        try:values,speeds,lost=parse_feedback(payload)
        except (ValueError,TypeError,KeyError,IndexError) as exc:self.error=str(exc);return False
        self.values,self.speeds,self.lost,self.stamp=values,speeds,lost,now;self.error=''
        self.history.append((now,dict(values)));return True
    def fresh(self,now=None):return bool(self.values and not self.error and 0<=(time.monotonic() if now is None else now)-self.stamp<=.4 and not self.lost)
    def capture(self,now=None):
        now=time.monotonic() if now is None else now
        if not self.fresh(now):raise ValueError('关节反馈断开或不新鲜，未记录姿态')
        recent=[(t,v) for t,v in self.history if 0<=now-t<=.65]
        if len(recent)<3 or recent[-1][0]-recent[0][0]<.35:raise ValueError('请在当前姿态停稳约半秒再记录')
        for n in ALL_JOINTS:
            lo,hi=JOINT_LIMITS[n];v=self.values[n]
            if not lo<=v<=hi:raise ValueError('实测姿态超出动作执行范围：'+n+'；请稍微离开关节极限')
            threshold=2. if n in BODY_JOINTS else 5.
            if max(x[n] for _,x in recent)-min(x[n] for _,x in recent)>threshold*.5:raise ValueError('机器人仍在移动，请停稳约半秒后记录')
            speed=self.speeds[n]
            if speed is not None and (not math.isfinite(speed) or abs(speed)>threshold):raise ValueError('关节尚未停稳，未记录姿态')
        return dict(self.values)


def ensure_ros():
    import rclpy
    if not rclpy.ok():
        args=['--ros-args']
        for n in ('preview_command','apply_command','action_command','status','set_hardware_enabled','set_hardware_enabled_without_reset','set_leg_waist_enabled'):
            args+=['-r',f'/control_center_joint_control/{n}:=/shelf_grasp_joint_control/{n}']
        rclpy.init(args=args)


def create_monitor(robot_id,state):
    ensure_ros()
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,ReliabilityPolicy,HistoryPolicy
    from std_msgs.msg import String
    node=Node('shelf_teaching_readonly_monitor')
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST)
    def receive(message):
        try:state.accept(json.loads(message.data))
        except (ValueError,TypeError):pass
    node.create_subscription(String,f'/topic_arm_whole_body_and_gripper_current_joints_status_0_{int(robot_id)}',receive,qos)
    return node
