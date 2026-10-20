"""Prevent release from stale, unrelated or incomplete demo requests."""
import json
from .arm_selection import require_same_arm,gripper_name,plan_arm


def check_base_anchor(current,reference):
    import math
    from .demo_config import wrap
    if len(current)!=3 or len(reference)!=3 or not all(math.isfinite(v) for v in [*current,*reference]):
        raise ValueError('桌面观察的底盘位置无效')
    if math.hypot(current[0]-reference[0],current[1]-reference[1])>.01 or abs(wrap(current[2]-reference[2]))>.02:
        raise ValueError('观察桌面后底盘位置改变，请重新识别')


def validate_place(plan,run):
    arm=plan_arm(plan);jaw=gripper_name(arm)
    if plan.get('placement_mode')=='standalone':
        from .placement_request import standalone_source,check_held_gripper
        context,grasp,record=standalone_source(run)
        if plan.get('placement_request_id')!=context['run_id']:
            raise ValueError('放置请求已失效，请重新确认放置')
        if plan_arm(grasp)!=arm:raise ValueError('放置与抓取手臂不一致')
        check_held_gripper(plan['source_q'],record)
    else:
        record=json.loads((run/'demo_grasp_result.json').read_text())
        context=json.loads((run/'demo_run.json').read_text())
        base=json.loads((run/'demo_base_result.json').read_text())
        if base.get('status')!='completed' or (context.get('run_id') and (base.get('run_id')!=context['run_id'] or
                base.get('leg')!='to_front' or base.get('name')!=context['front'] or base.get('map')!=context['map'])):
            raise ValueError('尚未确认本轮导航回到所选前台航点')
    require_same_arm(plan,record)
    if (plan.get('place_protocol_version') not in (1,2,3) or not plan.get('table_height_checked') or
            record.get('status')!='completed' or
            record.get('plan_id')!=context['grasp_plan_id'] or plan.get('parent_plan_id')!=context['grasp_plan_id'] or
            plan['product']['id']!=context['product_id'] or plan.get('demo_settings')!=context['settings']):
        raise ValueError('放置来源与本轮演示不一致')
    if plan.get('kind')=='observe':
        import math
        attempt=json.loads((run/'observation_attempt.json').read_text())
        if attempt.get('plan_id')!=plan.get('plan_id') or attempt.get('run_id')!=context.get('run_id') or (run/'demo_observe_result.json').exists():
            raise ValueError('观察动作不属于当前请求或已经完成，禁止重复抬手')
        if plan.get('observation_protocol_version')!=1 or [s.get('phase') for s in plan['segments']]!=['observe_lift']:
            raise ValueError('观察路径无效，禁止夹爪操作')
        held=float(plan.get('held_gripper_deg',float('nan')))
        previous=float(record.get('q',{}).get(jaw,float('nan')))
        if not math.isfinite(held) or not math.isfinite(previous) or abs(held-previous)>8:
            raise ValueError('观察路径的持瓶状态不一致')
        if abs(float(plan['source_q'][jaw])-held)>1e-6:
            raise ValueError('观察路径夹爪起始状态不一致')
        return
    expected=(['place_approach','place_lower','place_return'] if plan.get('place_protocol_version')==3
              else ['place_approach','place_lower','place_release_retreat','place_return'])
    if [s.get('phase') for s in plan['segments']]!=expected:
        raise ValueError('放置阶段不完整，不能松爪')
    if plan.get('place_protocol_version') in (2,3):
        # Reject malformed/new-format speeds BEFORE unlocking or entering SYNC.
        from autolife_handeye_calibration_306.verification_trajectory import JointPath
        from autolife_handeye_calibration_306.core import ARMS
        from .motion import configure_grasp_path,PathSpeedProfile
        import math
        for segment in plan['segments']:
            path=configure_grasp_path(JointPath(ARMS[arm],segment['waypoints']),plan['config'])
            cap=float(segment.get('path_speed_cap',float('nan')))
            if not math.isfinite(cap) or not 0<cap<=path.max_speed+1e-6:
                raise ValueError('放置速度上限无效，请重新规划')
            path.max_speed=min(path.max_speed,cap)
            PathSpeedProfile(path,segment.get('path_speed_profile',[]))
            switch=float(segment.get('transfer_from_s',path.length))
            if not math.isfinite(switch) or not 0<=switch<=path.length:
                raise ValueError('退出阶段速度切换位置无效')


class StationaryBaseGuard:
    """Acquire the same chassis lease during arm placement; never sends velocity."""
    def __init__(self,node):
        import time,math
        from nav_msgs.msg import Odometry
        from rclpy.qos import qos_profile_sensor_data
        from control_center_chassis_guard import ChassisControlLease
        self.time=time;self.math=math;self.last=None;self.origin=None;self.stamp=None
        self.lease=ChassisControlLease(domain_id='0',robot_id='300')
        if not self.lease.acquire(timeout_sec=0):
            self.lease.shutdown();raise ValueError('放置时底盘控制权不可用')
        def callback(m):
            stamp=(m.header.stamp.sec,m.header.stamp.nanosec)
            if stamp==self.stamp:return
            self.stamp=stamp
            p=m.pose.pose.position;r=m.pose.pose.orientation;v=m.twist.twist
            self.last=(time.monotonic(),p.x,p.y,math.atan2(2*(r.w*r.z+r.x*r.y),1-2*(r.y*r.y+r.z*r.z)),
                       math.hypot(v.linear.x,v.linear.y),abs(v.angular.z))
        self.sub=node.create_subscription(Odometry,'/topic_gv_wheel_odom_0_300',callback,qos_profile_sensor_data)

    def check(self):
        from .demo_config import wrap
        if not self.lease.held:raise ValueError('放置时底盘控制权丢失')
        if self.last is None or self.time.monotonic()-self.last[0]>.3:raise ValueError('放置时底盘反馈过期')
        if not all(self.math.isfinite(v) for v in self.last):raise ValueError('底盘反馈无效')
        if self.last[4]>.015 or self.last[5]>.025:raise ValueError('放置时底盘未停稳')
        if self.origin is None:self.origin=self.last
        if self.math.hypot(self.last[1]-self.origin[1],self.last[2]-self.origin[2])>.01 or abs(wrap(self.last[3]-self.origin[3]))>.02:
            raise ValueError('放置期间底盘位置改变')

    def close(self):self.lease.release();self.lease.shutdown()
