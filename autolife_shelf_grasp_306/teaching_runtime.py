"""Nonblocking status polling and ownership of a windowless VR navigation stack."""
import json,ssl,subprocess,sys,time,urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PyQt5 import QtCore as C


def get_status(url):
    try:
        with urllib.request.urlopen(url+'/api/status',context=ssl._create_unverified_context(),timeout=.7) as r:data=json.load(r)
        if not isinstance(data,dict) or 'teleop' not in data:return None
        return data
    except (OSError,ValueError,TimeoutError):return None


def backend_state(data):
    status=(data or {}).get('teleop',{})
    return status.get('backend',status) if isinstance(status,dict) else {}


class TeachingRuntime(C.QObject):
    changed=C.pyqtSignal()
    def __init__(self,robot_id,log_dir,parent=None,entry=None,url='https://127.0.0.1:8445'):
        super().__init__(parent);self.robot_id=int(robot_id);self.url=url
        self.entry=Path(entry or f'/home/ubuntu/ros2_ws/bin/vr_navigation{robot_id}_stack')
        self.log_dir=Path(log_dir);self.process=None;self.log=None;self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='teaching-vr-status')
        self.future=None;self.last_poll=0.;self.data=None;self.phase='stopped';self.message='遥操未开启';self.started=0.;self.closed=False;self.start_pending=False
        self.timer=C.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(100)
    @property
    def owned(self):return self.process is not None and self.process.poll() is None
    @property
    def busy(self):return self.owned or self.start_pending or bool(self.data)
    def start(self):
        if self.owned or self.start_pending:raise ValueError('遥操正在运行或启动中')
        if not self.entry.is_file():raise ValueError('未找到VR导航后台启动器：'+str(self.entry))
        self.start_pending=True;self.phase='checking';self.message='检查遥操会话';self.last_poll=0.;self.changed.emit()
    def _spawn(self):
        self.log_dir.mkdir(parents=True,exist_ok=True);self.log=(self.log_dir/'teaching_vr.log').open('ab',buffering=0)
        try:
            self.process=subprocess.Popen([sys.executable,'-m','autolife_shelf_grasp_306.teaching_vr_runner','--entry',str(self.entry),'--url',self.url],
                stdin=subprocess.PIPE,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        except Exception:
            self.log.close();self.log=None;raise
        self.started=time.monotonic();self.phase='starting';self.message='后台启动中，不会打开桌面窗口'
    def stop(self):
        self.start_pending=False
        if not self.owned:
            if self.data:raise ValueError('这是其他窗口启动的VR会话，请先在原窗口关闭')
            self.phase='stopped';self.message='遥操已关闭';self.changed.emit();return
        if self.phase=='stopping':return
        self.phase='stopping';self.message='正在停用遥操并关闭后台'
        try:self.process.stdin.write(b'stop\n');self.process.stdin.flush();self.process.stdin.close()
        except (BrokenPipeError,OSError,ValueError):pass
        self.changed.emit()
    def tick(self):
        now=time.monotonic()
        if self.future is not None and self.future.done():
            try:self.data=self.future.result()
            except Exception:self.data=None
            self.future=None
            if self.start_pending:
                self.start_pending=False
                if self.data:self.phase='external';self.message='已有VR会话，请先在原窗口关闭后再从这里开启'
                else:
                    try:self._spawn()
                    except Exception as exc:self.phase='error';self.message='启动失败：'+str(exc)
            elif self.data and self.phase!='stopping':
                self.phase='running' if self.owned else 'external'
                self.message='遥操后台已就绪' if self.owned else '检测到其他窗口的VR会话'
            elif not self.data and not self.owned and self.phase=='external':self.phase='stopped';self.message='遥操未开启'
            elif not self.data and self.owned and self.phase=='running':self.message='遥操状态暂时不可用，正在重新连接'
            self.changed.emit()
        if self.process is not None and self.process.poll() is not None:
            code=self.process.returncode;closing=self.phase=='stopping';self.process=None
            if self.log:self.log.close();self.log=None
            self.phase='stopped' if closing or code==0 else 'error'
            errors={2:'另一套VR导航正在运行，请先关闭原窗口',3:'V5或其他遥操正在运行，请先关闭原程序',4:'导航冲突未能清理，请双击此处查看日志'}
            self.message='遥操已关闭' if closing or code==0 else errors.get(code,f'启动/运行异常（{code}），双击此处查看日志')
            self.data=None;self.changed.emit()
        if self.owned and self.phase=='starting' and now-self.started>90.:
            self.stop();self.message='启动超过90秒，已请求关闭；请查看后台日志';self.changed.emit()
        if self.future is None and now-self.last_poll>=.8 and not self.closed:
            self.last_poll=now;self.future=self.pool.submit(get_status,self.url)
    def close(self):
        if self.owned:self.stop()
        self.closed=True;self.timer.stop();self.pool.shutdown(wait=False,cancel_futures=True)
