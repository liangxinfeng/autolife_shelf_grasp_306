"""VR teaching UI facade; passive capture and exclusive task execution."""
import fcntl,json,os,time
from pathlib import Path
import yaml
from PyQt5 import QtCore as C,QtWidgets as Q
from .teaching_feedback import FeedbackState,create_monitor
from .teaching_runtime import TeachingRuntime
from .teaching_library import TeachingLibrary


class TeachingStudio(Q.QWidget):
    completed=C.pyqtSignal(str,bool,str)
    def __init__(self,owner):
        super().__init__();self.owner=owner;self.panel=self;self.monitor=None;self.lease=None
        config=owner.root/'config/grasp.yaml'
        if not config.exists():
            from ament_index_python.packages import get_package_share_directory
            config=Path(get_package_share_directory('autolife_shelf_grasp_306'))/'config/grasp.yaml'
        self.cfg=yaml.safe_load(config.read_text('utf8'))
        self.robot_id=int(self.cfg['robot_id']);self.feedback=FeedbackState()
        self.library=TeachingLibrary(owner.root/'data/fixed_actions')
        from .teaching_client import TeachingClient
        self.recorder=TeachingClient(self.robot_id,self.library.directory,owner.run,self)
        self.runtime=TeachingRuntime(self.robot_id,owner.run,self)
        from .action_studio import ActionExecutor
        self.engine=ActionExecutor(owner);self.engine.setParent(self);self.engine.hide()
        self.engine.completed.connect(self._completed)
        from .teaching_composer import TeachingComposer
        self.editor=TeachingComposer(self);layout=Q.QVBoxLayout(self);layout.setContentsMargins(8,6,8,6);layout.addWidget(self.editor)
        self.runtime.changed.connect(self.editor.update_runtime)
        self.recorder.changed.connect(self.editor.update_recorder)
        self.recorder.failed.connect(self.editor.recorder_failed)
        self.timer=C.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(50)
    def ensure(self):
        self.recorder.ensure()
        if self.monitor is None:self.monitor=create_monitor(self.robot_id,self.feedback)
        from ament_index_python.packages import get_package_share_directory
        model = Path(get_package_share_directory('control_center_joint_control'))/'urdf/robot_v3_1.urdf'
        self.editor.ensure_model(str(model))
    def records(self):return self.library.records()
    def moving(self):return self.runtime.busy or self.engine.moving()
    def _acquire(self):
        if self.runtime.busy:raise ValueError('请先停止并关闭遥操，再执行动作串或抓取任务')
        if self.editor.remote_locked():raise ValueError('请先完成手柄示教和录制，再执行任务')
        if self.lease:return
        path=Path(f'/run/user/{os.getuid()}/vr345-arm-control.lock')
        handle=path.open('a+')
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close();raise ValueError('V5/VR导航正在控制关节，请先停用并关闭对应程序')
        self.lease=handle
    def _unlock(self):
        if self.lease:self.lease.close();self.lease=None
    def start_vr(self):
        if self.engine.moving():raise ValueError('动作正在执行，请先停止并等待停稳')
        if not self.editor.begin_remote():return
        self.engine.release();self._unlock();self.runtime.start()
    def execute(self,token,record,**kwargs):
        self._acquire()
        try:self.engine.execute(token,record,**kwargs)
        except Exception:
            if self.engine.bridge is None:self._unlock()
            raise
    def adjust_posture(self,token,setting):
        self._acquire();self.engine.adjust_posture(token,setting)
    def prepare_posture(self,token):
        self._acquire();self.engine.prepare_posture(token)
    def _completed(self,token,ok,text):
        if self.engine.bridge is None:self._unlock()
        self.editor.message(text,not ok);self.completed.emit(token,ok,text)
    def release(self):
        if self.runtime.busy:raise ValueError('请先点击“停止并关闭”，结束遥操后再切换到抓取与任务流程')
        self.engine.release();self._unlock()
    def stop(self):
        self.engine.stop()
        if self.runtime.owned or self.runtime.start_pending:self.runtime.stop()
    def tick(self):
        if self.monitor:
            import rclpy
            for _ in range(3):rclpy.spin_once(self.monitor,timeout_sec=0.)
        self.editor.tick()
    def shutdown(self):
        self.recorder.close()
        self.editor.persist_draft();self.timer.stop();self.runtime.close();self.engine.shutdown();self._unlock()
        if self.monitor:self.monitor.destroy_node();self.monitor=None
