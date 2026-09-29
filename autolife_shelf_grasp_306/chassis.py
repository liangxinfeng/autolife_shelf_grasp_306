"""Isolated manual-base helper. No navigation startup, arm command or auto motion."""
import json
import math
import sys
import threading
import time


class InputWatchdog:
    def __init__(self):
        self.values=(0.,0.,0.);self.stamp=-1.;self.sequence=-1;self.blocked=False

    def update(self,packet,now):
        values=tuple(float(v) for v in packet['velocity']);stamp=float(packet['stamp']);seq=int(packet['sequence'])
        if (len(values)!=3 or not all(math.isfinite(v) for v in (*values,stamp)) or
                math.hypot(*values[:2])>.151 or abs(values[2])>.401 or
                not 0<=now-stamp<.25 or seq<=self.sequence):
            raise ValueError('底盘输入过期或超出低速范围')
        if not any(values):self.blocked=False
        self.values=values if not self.blocked else (0.,0.,0.);self.stamp=stamp;self.sequence=seq

    def inhibit(self):self.blocked=True;self.values=(0.,0.,0.)

    def current(self,now):
        return self.values if 0<=now-self.stamp<.25 else (0.,0.,0.)


def main():
    import rclpy
    import redis
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Bool
    from action_msgs.msg import GoalStatusArray
    from control_center_chassis_guard import ChassisControlLease
    from autolife_handeye_calibration_307.bridge import competing_controller

    rclpy.init();node=Node('shelf_grasp_chassis_306')
    direct=node.create_publisher(Twist,'/topic_gv_target_cmd_vel_0_307',1)
    manual=node.create_publisher(Twist,'/manual_cmd_vel',1)
    override=node.create_publisher(Bool,'/manual_cmd_vel_active',1)
    lease=ChassisControlLease(domain_id='0',robot_id='307')
    client=redis.Redis(host='127.0.0.1',socket_timeout=.15,socket_connect_timeout=.15)
    state={'odom':0.,'stationary_since':None,'vendor_nav':False,'actions':{}}
    def odom(msg):
        now=time.monotonic();state['odom']=now;v=msg.twist.twist
        still=math.hypot(v.linear.x,v.linear.y)<.015 and abs(v.angular.z)<.025
        state['stationary_since']=(state['stationary_since'] or now) if still else None
    node.create_subscription(Odometry,'/topic_gv_wheel_odom_0_307',odom,qos_profile_sensor_data)
    node.create_subscription(Bool,'/robot_navigation_0_307/is_navigating',lambda m:state.update(vendor_nav=m.data),10)
    for topic in ('/navigate_to_pose/_action/status','/navigate_through_poses/_action/status','/dock_robot/_action/status'):
        node.create_subscription(GoalStatusArray,topic,
            lambda m,t=topic:state['actions'].update({t:any(s.status in (1,2,3) for s in m.status_list)}),
            QoSProfile(depth=10,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    lock=threading.Lock();watch=InputWatchdog();done=threading.Event();input_error=['']
    def read():
        try:
            for line in sys.stdin:
                with lock:
                    try:watch.update(json.loads(line),time.monotonic())
                    except Exception:watch.inhibit();input_error[0]='底盘输入无效，已停止'
        finally:done.set()
    threading.Thread(target=read,daemon=True).start()
    active=False;route=None;current=[0.,0.,0.];error='';error_at=0.;next_health=0.;next_report=0.;conflict=''
    def publish(values):
        msg=Twist();msg.linear.x,msg.linear.y,msg.angular.z=values
        if route is manual:override.publish(Bool(data=True))
        if route is not None:route.publish(msg)
    def stop():
        nonlocal active,route,current
        if active:
            publish((0.,0.,0.))
            if route is manual:override.publish(Bool(data=False))
        active=False;route=None;current=[0.,0.,0.];lease.release()
    try:
        while rclpy.ok() and not done.is_set():
            rclpy.spin_once(node,timeout_sec=.01);now=time.monotonic()
            with lock:
                target=watch.current(now);bad=input_error[0];input_error[0]=''
                if active and now-watch.stamp>=.25:watch.inhibit();bad='操作心跳中断，已停止，请松开后重试'
            if now>=next_health:conflict=competing_controller();next_health=now+.5
            reason=(bad or conflict or ('关节或导航操作进行中，请先停止导航' if state['vendor_nav'] or any(state['actions'].values()) else '')
                    or ('底盘反馈未连接' if now-state['odom']>.5 else ''))
            if active and not lease.held:reason=lease.last_error or '底盘控制权已丢失'
            if active and ((route is manual and not manual.get_subscription_count()) or
                           (route is direct and (not direct.get_subscription_count() or manual.get_subscription_count()))):
                reason='底盘指令通道变化，请松开后重试'
            if reason or not any(target):
                stop()
                if reason:
                    error=reason;error_at=now
                    with lock:watch.inhibit()
                elif now-error_at>1.:error=''
            else:
                if not active:
                    try:
                        if client.get('shared_params:0:300:is_navigating') in (b'true',b'True'):
                            raise ValueError('底盘正由其他程序控制，请先退出对应控制')
                        if not lease.acquire(timeout_sec=0):raise ValueError(lease.last_error or '无法接管底盘')
                        route=manual if manual.get_subscription_count() else direct
                        if not route.get_subscription_count():raise ValueError('没有底盘指令接收端')
                        active=True;error=''
                    except Exception as exc:
                        stop();error=str(exc);error_at=now
                        with lock:watch.inhibit()
                if active:
                    # Gentle acceleration; key-up/watchdog bypass the ramp.
                    for i,(v,limit) in enumerate(zip(target,(.012,.012,.032))):
                        current[i]+=max(-limit,min(limit,v-current[i]))
                    publish(current)
            if now>=next_report:
                since=state['stationary_since']
                print(json.dumps({'active':active,'stationary':bool(since and now-since>.3 and now-state['odom']<.5),
                    'error':error,'route':'导航控制入口' if route is manual else '底盘控制入口'},ensure_ascii=False),flush=True)
                next_report=now+.15
            time.sleep(.01)
    finally:
        stop();lease.shutdown();client.close();node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
