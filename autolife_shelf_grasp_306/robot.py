"""Read-only feedback for planning. No publisher or automatic hardware takeover."""
import threading
import time
from collections import deque
import json
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from autolife_handeye_calibration_306.core import ARMS,BODY,NECK


class Robot(Node):
    def __init__(self):
        super().__init__('shelf_grasp_feedback_306')
        self.history=deque(maxlen=120);self.mutex=threading.Lock();self.last_error='等待关节反馈'
        self.base=None;self.base_stamp=None
        self.create_subscription(String,'/topic_arm_whole_body_and_gripper_current_joints_status_0_306',self.receive,qos_profile_sensor_data)
        self.create_subscription(Odometry,'/topic_gv_wheel_odom_0_306',self.receive_base,qos_profile_sensor_data)
        self.stop_event=threading.Event();self.thread=threading.Thread(target=self.spin,daemon=True);self.thread.start()

    def spin(self):
        while not self.stop_event.is_set() and rclpy.ok():rclpy.spin_once(self,timeout_sec=.03)

    def receive(self,msg):
        try:
            raw=json.loads(msg.data);q={}
            for key,names in [('leg_waist_joint_state',BODY),('left_arm_joint_state',ARMS['left']),
                              ('right_arm_joint_state',ARMS['right']),('neck_joint_state',NECK)]:
                values=raw[key]['position']
                if len(values)!=len(names):raise ValueError('关节数量不一致')
                if any(raw[key].get('communication_lost',[])):raise ValueError('关节通信离线')
                q.update(zip(names,map(float,values)))
            for side in ('left','right'):
                group=raw.get(f'{side}_gripper_state',{})
                v=group.get('position',[])
                if len(v)!=1:raise ValueError('等待夹爪真实反馈')
                if any(group.get('communication_lost',[])):raise ValueError('夹爪通信离线')
                q[f'Joint_{side.title()}_Gripper']=float(v[0])
            if not np.isfinite(list(q.values())).all():raise ValueError('关节反馈无效')
            with self.mutex:self.history.append((time.monotonic(),q));self.last_error=''
        except Exception as exc:self.last_error=str(exc)

    def sample(self,stamp=None,stationary=False):
        with self.mutex:h=list(self.history)
        now=time.monotonic()
        if not h or now-h[-1][0]>.3:raise ValueError(self.last_error or '关节反馈过期')
        nearest=h[-1] if stamp is None else min(h,key=lambda v:abs(v[0]-stamp))
        if stamp is not None and abs(nearest[0]-stamp)>.12:raise ValueError('图像与关节时间未配对')
        if stationary:
            recent=[q for t,q in h if now-t<.4]
            if len(recent)<5 or h[-1][0]-h[0][0]<.3:raise ValueError('等待机器人停稳')
            if np.max(np.ptp([[q[n] for n in BODY+NECK+ARMS['left']+ARMS['right']] for q in recent],axis=0))>.35:
                raise ValueError('请保持身体、头部和双臂不动')
        return dict(nearest[1])

    def receive_base(self,msg):
        stamp=(msg.header.stamp.sec,msg.header.stamp.nanosec)
        if stamp==self.base_stamp:return
        p=msg.pose.pose.position;r=msg.pose.pose.orientation;v=msg.twist.twist
        values=[p.x,p.y,math.atan2(2*(r.w*r.z+r.x*r.y),1-2*(r.y*r.y+r.z*r.z)),math.hypot(v.linear.x,v.linear.y),abs(v.angular.z)]
        with self.mutex:self.base=(time.monotonic(),values);self.base_stamp=stamp

    def base_sample(self):
        with self.mutex:base=self.base
        if base is None or time.monotonic()-base[0]>.3:raise ValueError('等待新鲜底盘停稳反馈')
        values=base[1]
        if not np.isfinite(values).all() or values[3]>.015 or values[4]>.025:raise ValueError('观察桌面时底盘必须停稳')
        return list(values[:3])

    def close(self):self.stop_event.set();self.thread.join(1);self.destroy_node()
