"""Nonblocking Qt client; slow disk/ROS/recording never runs in the GUI loop."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from PyQt5 import QtCore as C
from openarmx_teleop_vr_navigation_306.teaching_ipc import request


class TeachingClient(C.QObject):
    changed=C.pyqtSignal(dict)
    failed=C.pyqtSignal(str)
    def __init__(self,robot_id,library,run,parent=None):
        super().__init__(parent);self.robot_id=robot_id;self.library=Path(library);self.run=Path(run)
        self.process=None;self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='teaching-rpc')
        self.future=None;self.command_future=None;self.last_poll=0.;self.closed=False
        self.data=dict(available=False,enabled=False,phase='idle');self.pending=[];self.log=None
        self.timer=C.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(100)
    def ensure(self):
        if self.process and self.process.poll() is None:return
        self.run.mkdir(parents=True,exist_ok=True);self.log=(self.run/'teaching-recorder.log').open('ab',buffering=0)
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        self.process=subprocess.Popen([sys.executable,'-m','autolife_shelf_grasp_306.teaching_service',
            '--robot-id',str(self.robot_id),'--library',str(self.library)],stdin=subprocess.PIPE,
            stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    def command(self,op,**values):
        self.ensure()
        if len(self.pending)>8:raise ValueError('正在处理上次操作，请稍候')
        self.pending.append(dict(op=op,request_id=uuid.uuid4().hex,**values))
    def _send(self,body):
        # Retry only the same idempotent request, never manufacture a second click.
        for attempt in range(2):
            try:return request(self.robot_id,'/command',body,timeout=2.,desktop=True)
            except (TimeoutError,ConnectionError,OSError):
                if attempt:raise
    def _accept(self,data):
        self.data=data;self.changed.emit(data)
    def tick(self):
        if self.closed:return
        if self.command_future and self.command_future.done():
            try:self._accept(self.command_future.result())
            except Exception as exc:self.failed.emit(str(exc))
            self.command_future=None
        if self.future and self.future.done():
            try:self._accept(self.future.result())
            except Exception as exc:
                self._accept(dict(self.data,available=False,message='录制服务暂未连接：'+str(exc)))
            self.future=None
        if self.command_future or self.future or not self.process:return
        if self.process.poll() is not None:
            self._accept(dict(self.data,available=False,enabled=False,message='录制服务已退出，请重新开启示教'))
            self.pending.clear();return
        if self.pending and self.data.get('available'):
            body=self.pending.pop(0);body.setdefault('session',self.data.get('session',''))
            self.command_future=self.pool.submit(self._send,body)
        elif time.monotonic()-self.last_poll>.2:
            self.last_poll=time.monotonic();self.future=self.pool.submit(request,self.robot_id)
    def close(self):
        self.closed=True;self.timer.stop();self.pending.clear()
        if self.process and self.process.stdin:
            self.process.stdin.close() # EOF stops sampling and keeps incomplete recordings.
        self.pool.shutdown(wait=False,cancel_futures=True)
        if self.log:self.log.close()
