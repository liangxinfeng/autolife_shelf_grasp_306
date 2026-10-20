"""Compact parameter editor and bounded process adapter for fixed base motion."""
import json
from PyQt5 import QtCore as C,QtWidgets as Q
from .base_sequence import MOVES,normalize,summary,LINEAR_SPEED_MAX_CM_S,ANGULAR_SPEED_MAX_DEG_S

class BaseEditor(Q.QDialog):
    def __init__(self,parent,value):
        super().__init__(parent);self.setWindowTitle('底盘定量动作顺序');self.resize(640,450)
        layout=Q.QVBoxLayout(self);self.enabled=Q.QCheckBox('启用本步骤底盘动作');layout.addWidget(self.enabled)
        self.table=Q.QTableWidget(0,3);self.table.setHorizontalHeaderLabels(['动作','距离 / 角度','速度'])
        self.table.horizontalHeader().setSectionResizeMode(Q.QHeaderView.Stretch);layout.addWidget(self.table)
        row=Q.QHBoxLayout();layout.addLayout(row)
        for text,fn in [('添加',lambda:self.add()),('删除',self.remove),('上移',lambda:self.move(-1)),('下移',lambda:self.move(1))]:
            b=Q.QPushButton(text);b.clicked.connect(fn);row.addWidget(b)
        note=Q.QLabel('按顺序执行，以每段开始时的车头方向为准；距离单位cm、角度单位°。\n无需导航地图。请留出底盘、手臂和持物空间；近障碍先停车，清除后自动续行；持续阻挡或反馈中断会停止。')
        note.setWordWrap(True);layout.addWidget(note)
        buttons=Q.QDialogButtonBox(Q.QDialogButtonBox.Ok|Q.QDialogButtonBox.Cancel);layout.addWidget(buttons)
        buttons.accepted.connect(self.accept_checked);buttons.rejected.connect(self.reject)
        # Enable controls execution, not editing of the saved sequence.
        self.enabled.setToolTip('未勾选也可编辑和保存；勾选后任务流程才会执行。')
        v=normalize(value);self.enabled.setChecked(v['enabled'])
        for r in v['moves']:self.add(r)
    def add(self,r=None):
        r=r or dict(kind='forward',amount=50,speed=10);i=self.table.rowCount();self.table.insertRow(i)
        kind=Q.QComboBox()
        for k,t in MOVES.items():kind.addItem(t,k)
        kind.setCurrentIndex(kind.findData(r['kind']));amount=Q.QDoubleSpinBox();speed=Q.QDoubleSpinBox()
        for b in (amount,speed):b.setDecimals(1)
        def units():
            turn=kind.currentData().startswith('turn_');amount.setRange(.5 if turn else 1,360 if turn else 300)
            amount.setSuffix(' °' if turn else ' cm');speed.setRange(5 if turn else 3,ANGULAR_SPEED_MAX_DEG_S if turn else LINEAR_SPEED_MAX_CM_S)
            speed.setSuffix(' °/s' if turn else ' cm/s')
        kind.currentIndexChanged.connect(units);units();amount.setValue(r['amount']);speed.setValue(r['speed'])
        for j,w in enumerate((kind,amount,speed)):self.table.setCellWidget(i,j,w)
        self.table.selectRow(i)
    def record(self):
        return normalize(dict(enabled=self.enabled.isChecked(),moves=[dict(kind=self.table.cellWidget(i,0).currentData(),
            amount=self.table.cellWidget(i,1).value(),speed=self.table.cellWidget(i,2).value()) for i in range(self.table.rowCount())]))
    def remove(self):
        if self.table.currentRow()>=0:self.table.removeRow(self.table.currentRow())
    def move(self,d):
        i=self.table.currentRow();j=i+d
        if not 0<=j<self.table.rowCount():return
        rows=self.record()['moves'];rows[i],rows[j]=rows[j],rows[i];self.table.setRowCount(0)
        for r in rows:self.add(r)
        self.table.selectRow(j)
    def accept_checked(self):
        try:self.record()
        except ValueError as exc:Q.QMessageBox.warning(self,'参数有误',str(exc));return
        self.accept()

class BaseCell(Q.QPushButton):
    def __init__(self,parent,value=None):
        super().__init__(parent);self.value=normalize(value);self.refresh();self.clicked.connect(self.edit)
    def currentText(self):return summary(self.value)
    def refresh(self):self.setText(self.currentText()+' · 设置');self.setToolTip(self.currentText())
    def edit(self):
        dlg=BaseEditor(self,self.value)
        if dlg.exec_()==Q.QDialog.Accepted:self.value=dlg.record();self.refresh()

class BaseRunner(C.QObject):
    completed=C.pyqtSignal(str,bool,str)
    def __init__(self,owner):
        super().__init__(owner);self.owner=owner;self.proc=None;self.token=None;self.buffer=b'';self.result=None
    def moving(self):return self.proc is not None
    def start(self,token):
        if self.moving():raise ValueError('底盘动作尚未结束')
        self.token=token;self.result=None;self.buffer=b'';p=C.QProcess(self);self.proc=p
        p.readyReadStandardOutput.connect(self.read);p.readyReadStandardError.connect(self.log)
        p.finished.connect(self.finished);p.errorOccurred.connect(self.error)
        p.start('/usr/bin/python3',['-m','autolife_shelf_grasp_306.base_execution','--run',str(self.owner.run),'--token',token])
    def read(self):
        self.buffer+=bytes(self.proc.readAllStandardOutput())
        while b'\n' in self.buffer:
            line,self.buffer=self.buffer.split(b'\n',1)
            try:d=json.loads(line)
            except ValueError:continue
            if d.get('token')!=self.token:continue
            if d.get('event')=='done':self.result=d
            elif d.get('event')=='progress':self.owner.flow_label.setText(d.get('message') or f'底盘动作 {min(d["index"]+1,d["total"])}/{d["total"]}')
    def log(self):
        with (self.owner.run/'base_execution.log').open('ab') as f:f.write(bytes(self.proc.readAllStandardError()))
    def error(self,code):
        if code==C.QProcess.FailedToStart:self.finished(-1)
    def finished(self,*_):
        if self.proc is None:return
        self.read();self.log();p=self.proc;self.proc=None;p.deleteLater();r=self.result or {}
        self.completed.emit(self.token,r.get('status')=='completed',r.get('reason','底盘动作进程未正常完成'))
        self.owner.controls()
    def stop(self):
        if self.proc:(self.owner.run/'stop').touch()
    def shutdown(self):
        self.stop();p=self.proc
        if p and not p.waitForFinished(2200):p.terminate();p.waitForFinished(1000)
