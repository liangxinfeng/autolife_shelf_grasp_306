"""Start the existing navigation stack once; never launch a second planner.

The stack deliberately survives this GUI closing, like a shared service. Only
the GUI's own goal is cancelled by navigation_bridge on exit. Control-center
process records make the backend discoverable to its existing service manager.
"""
import fcntl
import os
from pathlib import Path
import subprocess
import threading
import time


class NavigationRuntime:
    def __init__(self, run, emit, map_owners):
        self.run=Path(run);self.emit=emit;self.thread=None;self.error='';self.started=0.
        self.map_owners=map_owners;self.process=None

    @property
    def starting(self):return self.thread is not None and self.thread.is_alive()

    def start(self, map_path):
        if os.environ.get('ROS_DOMAIN_ID','0')!='0':raise ValueError('非机器人ROS域，不允许启动真实导航后台')
        if self.starting:raise ValueError('导航后台正在启动，请稍候')
        if self.process is not None and self.process.poll() is not None:self.process=None
        self.error='';self.started=time.monotonic()
        self.thread=threading.Thread(target=self._start,args=(str(map_path),),daemon=True)
        self.thread.start()

    def _start(self, map_path):
        map_path=str(map_path)
        try:
            from control_center_shell import preflight as p
            with open('/tmp/grasp306_navigation_start.lock','a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                conflicts=p.module_process_conflicts('navigation')
                if conflicts:
                    self.emit('nav_message',text='已存在导航后台，正在连接；不会重复启动')
                    return
                if p.module_process_conflicts('slam'):
                    raise ValueError('当前正在建图，请先保存地图并关闭建图模块')
                # Use the same vendor-navigation handover as the control center;
                # never restart sensor, chassis or arm services here.
                state=subprocess.run(['systemctl','--user','is-active','--quiet','gv-slam-service.service'],timeout=3)
                if state.returncode==0:
                    self.emit('nav_message',text='正在停止原厂导航，由共享导航后台接管…')
                    result=subprocess.run(['systemctl','--user','stop','gv-slam-service.service'],capture_output=True,text=True,timeout=25)
                    if result.returncode:raise ValueError('原厂导航未停止：'+result.stderr[-300:])
                # Graph reads are thread-safe; do not spin rclpy's shared
                # executor from a second thread while the bridge is spinning.
                owners=[]
                for _ in range(10):
                    owners=self.map_owners()
                    if not owners:break
                    time.sleep(.2)
                if owners:raise ValueError('已有其他地图发布者，未启动重复定位：'+str(owners))
                processes=p._processes()
                markers=('/nav2_amcl/amcl','/control_center_navigation/amcl_startup_compat','cartographer_node','cmd_vel_arbiter_node.py')
                if any(any(m in v['command'] for m in markers) for v in processes.values()):
                    raise ValueError('仍有其他定位或底盘仲裁进程，请在控制中心服务页检查残留后台')
                if not Path(map_path).is_file():raise ValueError('当前地图文件不存在')
                with (self.run/'navigation_backend.log').open('ab',buffering=0) as stream:
                    process=subprocess.Popen(['/usr/bin/python3','/opt/ros/jazzy/bin/ros2','launch',
                        'control_center_navigation','bringup_all.launch.py','open_navigation_ui:=false',
                        'text_voice_robot_id:=300','text_voice_ros_domain_id:=0','map_yaml:='+map_path],
                        stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                time.sleep(.3)
                if process.poll() is not None:raise ValueError('导航启动退出，请查看 navigation_backend.log')
                self.process=process
                p.record_process('navigation',process.pid)
                self.emit('nav_message',text='导航后台已启动，等待双激光、地图和定位；无需打开控制中心')
        except Exception as exc:
            self.error=str(exc);self.emit('nav_error',text='启动导航失败：'+self.error)
