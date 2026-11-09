"""Heartbeat for a grasp-owned navigation launch; never publishes velocity.

Both the Qt event loop and navigation bridge must stay responsive. A stale
source latches failure, stops heartbeats, then exits so launch shuts down its
own backend. A reopened GUI creates a new launch/session, never revives a goal.
"""
import argparse
from pathlib import Path
import time


class SourceLease:
    def __init__(self, run, timeout=.30):
        self.paths=[Path(run)/'heartbeat',Path(run)/'navigation_bridge_heartbeat']
        self.timeout=timeout;self.failed_at=None;self.reason=''

    def check(self, wall=None, now=None):
        now=time.monotonic() if now is None else now
        if self.failed_at is not None:return False
        for p in self.paths:
            try:
                stamp=p.stat().st_mtime
                # Read wall time AFTER stat: the writer may touch the file
                # between our reads; a fresh timestamp must not look future.
                age=(time.time() if wall is None else wall)-stamp
            except OSError:age=float('inf')
            if not 0<=age<=self.timeout:
                self.failed_at=now;self.reason=p.name+f' 已中断（age={age:.3f}s）'
                return False
        return True

    def should_exit(self, now=None):
        now=time.monotonic() if now is None else now
        return self.failed_at is not None and now-self.failed_at>=.5


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--topic',required=True)
    args=parser.parse_args();run=Path(args.run).resolve()
    # A dedicated topic prevents us from impersonating or conflicting with
    # a control-center/VR owner of a different navigation stack.
    expected='/grasp306/operator_heartbeat/'+run.name
    if args.topic!=expected or not run.name.startswith('shelf306_'):raise ValueError('导航心跳会话不匹配')
    import rclpy
    from std_msgs.msg import String
    rclpy.init();node=rclpy.create_node('grasp_navigation_owner_watchdog')
    publisher=node.create_publisher(String,args.topic,10)
    lease=SourceLease(run);reported=False
    try:
        while rclpy.ok():
            if lease.check():publisher.publish(String(data='grasp-nav-'+run.name))
            elif not reported:
                node.get_logger().error(lease.reason+'；停止心跳并关闭本窗口启动的导航后台');reported=True
            if lease.should_exit():break
            rclpy.spin_once(node,timeout_sec=.04)
    finally:
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
