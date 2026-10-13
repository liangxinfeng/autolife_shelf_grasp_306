"""Hold-to-drive dialog; camera stays visible behind it. No motion on opening."""
import json
import math
import time
from PyQt5 import QtCore as C,QtWidgets as Q


class HoldButton(Q.QPushButton):
    def leaveEvent(self,event):
        if self.isDown():self.setDown(False);self.released.emit()
        super().leaveEvent(event)


class ChassisDialog(Q.QDialog):
    MOVES={C.Qt.Key_W:(1,0,0),C.Qt.Key_S:(-1,0,0),C.Qt.Key_Q:(0,1,0),
           C.Qt.Key_E:(0,-1,0),C.Qt.Key_A:(0,0,1),C.Qt.Key_D:(0,0,-1)}

    def __init__(self,owner,start_backend=True):
        super().__init__(owner);self.owner=owner;self.keys=set();self.mouse=(0,0,0)
        self.sequence=0;self.buffer=b'';self.last_status=0.;self.stationary=False;self.active=False
        self.closing_since=None
        self.setWindowTitle('底盘调整');self.setWindowModality(C.Qt.WindowModal);self.resize(360,320)
        layout=Q.QVBoxLayout(self)
        label=Q.QLabel('按住移动 · 松开停止');label.setStyleSheet('font-size:20px;font-weight:600');layout.addWidget(label)
        self.speed=Q.QComboBox()
        for text,value in [('精细  0.05 m/s',.05),('低速  0.10 m/s',.10),('移动  0.15 m/s',.15)]:self.speed.addItem(text,value)
        self.speed.setFocusPolicy(C.Qt.NoFocus);layout.addWidget(self.speed)
        grid=Q.QGridLayout();layout.addLayout(grid)
        self.buttons=[]
        for text,key,r,c in [('Q 左移',C.Qt.Key_Q,0,0),('W 前进',C.Qt.Key_W,0,1),('E 右移',C.Qt.Key_E,0,2),
                             ('A 左转',C.Qt.Key_A,1,0),('D 右转',C.Qt.Key_D,1,2),('S 后退',C.Qt.Key_S,2,1)]:
            button=HoldButton(text);button.setMinimumHeight(45);button.setFocusPolicy(C.Qt.NoFocus)
            button.pressed.connect(lambda k=key:self.mouse_press(k));button.released.connect(self.stop)
            self.buttons.append(button);grid.addWidget(button,r,c)
        stop=Q.QPushButton('X 停止');stop.setObjectName('stop');stop.clicked.connect(self.stop);grid.addWidget(stop,1,1)
        self.message=Q.QLabel('连接底盘反馈…');self.message.setWordWrap(True);layout.addWidget(self.message)
        hint=Q.QLabel('仅手动低速调整，请观察货架、双臂及周围人员。');hint.setWordWrap(True);layout.addWidget(hint)
        close=Q.QPushButton('完成调整，返回抓取');close.clicked.connect(self.close);layout.addWidget(close)
        self.proc=None
        if start_backend:
            self.proc=C.QProcess(self);self.proc.readyReadStandardOutput.connect(self.read)
            self.proc.readyReadStandardError.connect(self.log)
            self.proc.finished.connect(lambda *_:self.failed('底盘控制进程已退出'))
            self.proc.errorOccurred.connect(lambda _:self.failed('底盘控制进程不可用'))
            self.proc.start('/usr/bin/python3',['-m','autolife_shelf_grasp_306.chassis'])
        self.timer=C.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(50)

    def available(self):return self.last_status and time.monotonic()-self.last_status<.6

    def mouse_press(self,key):
        if self.available() and self.closing_since is None:self.mouse=self.MOVES[key];self.tick()

    def stop(self):
        self.keys.clear();self.mouse=(0,0,0)
        for b in self.buttons:b.setDown(False)
        self.send((0.,0.,0.))

    def send(self,values):
        if not self.proc or self.proc.state()!=C.QProcess.Running:return
        self.sequence+=1
        self.proc.write((json.dumps({'sequence':self.sequence,'stamp':time.monotonic(),'velocity':values})+'\n').encode())

    def tick(self):
        if self.closing_since is not None:
            self.stop()
            if self.can_close():self.close()
            return
        if not self.isActiveWindow() or not self.available():
            self.stop();return
        vectors=[self.MOVES[k] for k in self.keys]+[self.mouse]
        x,y,w=(max(-1.,min(1.,sum(v[i] for v in vectors))) for i in range(3))
        scale=self.speed.currentData();norm=max(1.,math.hypot(x,y))
        self.send((x/norm*scale,y/norm*scale,w*(.2 if scale<=.05 else .3 if scale<=.1 else .4)))

    def read(self):
        self.buffer+=bytes(self.proc.readAllStandardOutput())
        while b'\n' in self.buffer:
            line,self.buffer=self.buffer.split(b'\n',1)
            try:data=json.loads(line)
            except ValueError:continue
            self.last_status=time.monotonic();self.stationary=data['stationary'];self.active=data['active']
            if data.get('error'):self.stop();self.message.setText(data['error'])
            else:self.message.setText('调整中 · 松开停止' if self.active else '已停稳' if self.stationary else '等待底盘停稳…')

    def log(self):
        with (self.owner.run/'chassis.log').open('ab') as f:f.write(bytes(self.proc.readAllStandardError()))

    def failed(self,text):self.stop();self.last_status=0.;self.message.setText(text)

    def keyPressEvent(self,event):
        if event.isAutoRepeat():event.accept();return
        if event.key() in self.MOVES:
            if self.available() and self.closing_since is None:self.keys.add(event.key());self.tick()
            event.accept();return
        if event.key() in (C.Qt.Key_X,C.Qt.Key_Space):self.stop();event.accept();return
        super().keyPressEvent(event)

    def keyReleaseEvent(self,event):
        if event.isAutoRepeat():event.accept();return
        if event.key() in self.MOVES:
            self.keys.discard(event.key());self.tick();event.accept();return
        super().keyReleaseEvent(event)

    def event(self,event):
        if event.type() in (C.QEvent.WindowDeactivate,C.QEvent.Hide) and hasattr(self,'keys'):self.stop()
        return super().event(event)

    def reject(self):self.close()

    def can_close(self):
        return (self.closing_since is not None and time.monotonic()-self.closing_since>.4
                and self.available() and self.last_status>self.closing_since and self.stationary and not self.active)

    def closeEvent(self,event):
        self.stop()
        if self.proc and self.proc.state()==C.QProcess.Running:
            if not self.can_close():
                if self.closing_since is None:self.closing_since=time.monotonic()
                self.message.setText('已请求停车，停稳后自动返回。');event.ignore();return
        self.shutdown();event.accept()

    def shutdown(self):
        self.timer.stop();self.stop()
        if self.proc:
            self.proc.closeWriteChannel()
            if not self.proc.waitForFinished(1000):self.proc.terminate();self.proc.waitForFinished(1000)
            if self.proc.state()!=C.QProcess.NotRunning:self.proc.kill();self.proc.waitForFinished(500)
