"""Use the existing guarded VR position-SYNC lease, only during user execution.

No motor/CAN access, driver clearing, velocity mode, reset or global tuning.
Vendor's 1.5 s lease watchdog is the crash fallback; normal exit holds then ASYNC.
"""
import json
import time
import rclpy
from std_msgs.msg import String
from std_srvs.srv import SetBool


class PositionSyncSession:
    def __init__(self,node,*,preserve_gripper_targets=False):
        self.node=node;self.requested=False;self.active=False;self.last_heartbeat=-float('inf')
        self.requires_grasp_finish=False
        self.preserve_gripper_targets=bool(preserve_gripper_targets)
        self.finish_client=node.create_client(SetBool,'/control_independent_sync_grasp_finish_0_345')
        self.client=node.create_client(SetBool,('/control_independent_sync_loaded_session_0_345'
            if self.preserve_gripper_targets else '/control_independent_sync_hold_session_0_345'))
        self.heartbeat=node.create_publisher(String,'/control_independent_sync_heartbeat_0_345',10)
        self.exit_pub=node.create_publisher(String,'/control_independent_sync_session_command_0_345',10)

    def enter(self,spin):
        deadline=time.monotonic()+2.
        while not self.client.service_is_ready() or (self.requires_grasp_finish and not self.finish_client.service_is_ready()):
            spin()
            if time.monotonic()>deadline:raise ValueError('保持式SYNC入口不可用；未开始抓取，不降级执行')
        if self.requires_grasp_finish and not self.finish_client.service_is_ready():
            raise ValueError('抓取保持交接服务未加载，请重启已更新的机械臂服务；未开始抓取')
        self.requested=True
        future=self.client.call_async(SetBool.Request(data=True));deadline=time.monotonic()+12.
        while not future.done():
            spin()
            if time.monotonic()>deadline:raise ValueError('保持式SYNC进入超时；停止抓取')
        result=future.result()
        if not result.success:raise ValueError('保持式SYNC拒绝：'+result.message)
        # 原厂切模式回调完成不等于新关节反馈已送达。动作尚未开始，
        # 只等待切换之后的连续新反馈；不延长运动中的过期阈值。
        accepted=time.monotonic();deadline=accepted+1.;stable=None;last=None;count=0
        while True:
            spin()
            now=time.monotonic()
            if not self.node._hardware_ready:
                raise ValueError('SYNC切换后硬件状态失效，未开始运动')
            stamp=self.node._feedback_monotonic
            if self.node._is_feedback_fresh() and stamp is not None and stamp>accepted:
                self.node.snapshot()  # 完整关节、离线标志和数值有效性检查。
                if last is None or stamp<last or stamp-last>.15:
                    stable=stamp;count=1
                elif stamp>last:count+=1
                last=stamp
                if count>=3 and stamp-stable>=.1:break
            else:
                stable=None;last=None;count=0
            if now>deadline:
                raise ValueError('SYNC切换完成但新关节反馈未恢复，未开始运动')
        self.active=True;self.tick()

    def tick(self):
        if not self.active:return
        if not self.node._hardware_ready:
            reason=getattr(self.node,'last_lock_reason','') or '硬件尚未就绪'
            raise ValueError('控制保护停止：'+reason)
        if not self.node._is_feedback_fresh():
            raise ValueError('关节反馈过期，停止运动；未将其误判为SYNC会话失败')
        now=time.monotonic()
        if now-self.last_heartbeat>=.1:
            self.heartbeat.publish(String(data=json.dumps({'active':True,'source':'shelf_grasp306','monotonic_time':now})))
            self.last_heartbeat=now

    def close(self, preserve_grippers=False):
        if not self.requested:return True
        # Called only AFTER cancelling queued targets and requesting measured hold.
        self.active=False
        if preserve_grippers:
            if not self.finish_client.service_is_ready():return False
            future=self.finish_client.call_async(SetBool.Request(data=False))
            deadline=time.monotonic()+3.
            while not future.done() and time.monotonic()<deadline:rclpy.spin_once(self.node,timeout_sec=.02)
            ok=future.done() and future.result().success
            if ok:self.requested=False
            return bool(ok)
        for _ in range(0 if self.preserve_gripper_targets else 3):self.exit_pub.publish(String(data='{"enable":false,"reason":"grasp stopped/completed"}'))
        if not self.client.service_is_ready():return False
        future=self.client.call_async(SetBool.Request(data=False))
        deadline=time.monotonic()+3.
        while not future.done() and time.monotonic()<deadline:rclpy.spin_once(self.node,timeout_sec=.02)
        ok=future.done() and future.result().success
        self.requested=False
        return bool(ok)
