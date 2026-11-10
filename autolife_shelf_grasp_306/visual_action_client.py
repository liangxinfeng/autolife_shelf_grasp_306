"""专用视觉请求通道，复用唯一视觉进程，不修改抓取计划/商品选择状态。"""
import json,time,uuid
from PyQt5 import QtCore as C


class VisualClient(C.QObject):
    def __init__(self,studio):
        super().__init__(studio);self.studio=studio;self.owner=studio.owner;self.pending=None
        self.timer=C.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(250)
    @property
    def busy(self):return self.pending is not None
    def request(self,action,callback,**values):
        o=self.owner
        if self.busy or o.busy or o.motion is not None:raise ValueError('当前视觉或动作还在运行，请完成后再操作')
        if self.studio.runtime.busy or self.studio.editor.remote_locked():raise ValueError('请先停止VR示教，再进行物体参照纠偏')
        if o.navigation.inflight:raise ValueError('请等待导航结束后定位物体参照')
        if not o.ready or not o.process:raise ValueError('视觉后台尚未就绪，普通姿态动作仍可使用')
        self.studio._acquire();ident=uuid.uuid4().hex
        self.pending=dict(id=ident,callback=callback,started=time.monotonic(),cancelled=False)
        payload=dict(action=action,visual_id=ident,**values)
        if o.process.write((json.dumps(payload,ensure_ascii=False,allow_nan=False)+'\n').encode())<0:
            self.pending=None
            if self.studio.engine.bridge is None:self.studio._unlock()
            raise ValueError('视觉请求发送失败')
        o.status.setText('物体参照：'+{'visual_commit_view':'正在保存已确认的物体轮廓','visual_freeze':'正在取景','visual_remember':'正在记住整物体','visual_add_view':'正在关联补录视角','visual_bind':'正在关联姿态','visual_correct':'重新定位并计算纠偏轨迹'}.get(action,action))
        o.controls()
    def receive(self,data):
        p=self.pending
        if not p or data.get('visual_id')!=p['id']:return
        self.pending=None;(self.owner.run/('visual_cancel_'+p['id'])).unlink(missing_ok=True)
        if p['cancelled']:data=dict(error='操作已取消，未执行纠偏动作')
        self.owner.controls()
        try:p['callback'](data)
        finally:
            if not self.busy and self.studio.engine.bridge is None:self.studio._unlock()
    def cancel(self):
        if self.pending:
            self.pending['cancelled']=True
            (self.owner.run/('visual_cancel_'+self.pending['id'])).touch()
    def fail(self,text):
        if self.pending:
            p=self.pending;self.pending=None
            try:p['callback'](dict(error=text))
            finally:
                if self.studio.engine.bridge is None:self.studio._unlock()
    def tick(self):
        if self.pending and not self.pending['cancelled'] and time.monotonic()-self.pending['started']>120:
            self.cancel();self.owner.status.setText('物体参照计算超时，已取消，等待只读后台退出本次计算')
