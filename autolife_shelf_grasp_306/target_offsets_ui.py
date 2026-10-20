import json,yaml
from PyQt5 import QtCore as C,QtWidgets as Q
from .core import atomic_json
from .no_wheel import NoWheelDoubleSpinBox,NoWheelComboBox
from .target_offsets import FRAME,defaults,checked_profiles,from_display,to_display,description

class TargetOffsets(Q.QGroupBox):
    changed=C.pyqtSignal()
    def __init__(self,root,parent=None):
        super().__init__('夹取中心补偿（mm）',parent)
        self.path=root/'data/target_offsets.json';self.problem=''
        config=root/'config/grasp.yaml'
        cfg=yaml.safe_load(config.read_text('utf-8')) if config.exists() else {}
        self.values=defaults(cfg)
        if self.path.exists():
            try:
                data=json.loads(self.path.read_text('utf-8'))
                if data.get('frame')!=FRAME:raise ValueError('补偿坐标系不匹配')
                self.values=checked_profiles(data['arms'])
            except (OSError,ValueError,KeyError,TypeError) as exc:self.problem='补偿配置无效，请重新设置：'+str(exc)
        form=Q.QFormLayout(self);form.setContentsMargins(8,8,8,8)
        self.arm=NoWheelComboBox();self.arm.addItem('左手','left');self.arm.addItem('右手','right');form.addRow('手臂',self.arm)
        self.spins=[]
        for label in ('前后：＋前 / −后','左右：＋右 / −左','上下：＋上 / −下'):
            spin=NoWheelDoubleSpinBox();spin.setRange(-50,50);spin.setDecimals(1);spin.setSuffix(' mm')
            self.spins.append(spin);form.addRow(label,spin)
        self.hint=Q.QLabel();self.hint.setWordWrap(True);form.addRow(self.hint)
        self.arm.currentIndexChanged.connect(self.refresh);self.refresh()
        for spin in self.spins:spin.valueChanged.connect(self.edit)

    def refresh(self,*_):
        side=self.arm.currentData()
        for spin,value in zip(self.spins,to_display(self.values[side])):
            spin.blockSignals(True);spin.setValue(value);spin.blockSignals(False)
        self.hint.setText(self.problem or ('当前：'+description(self.values[side])+'\n机器人躯干方向，三轴叠加；不修改标定TCP。'))

    def profiles(self):
        if self.problem:raise ValueError(self.problem)
        return checked_profiles(self.values)

    def edit(self,*_):
        next_values={k:list(v) for k,v in self.values.items()}
        next_values[self.arm.currentData()]=from_display(*(s.value() for s in self.spins))
        try:
            checked_profiles(next_values)
            atomic_json(self.path,dict(version=1,frame=FRAME,arms=next_values))
        except (ValueError,OSError) as exc:
            self.refresh();Q.QMessageBox.warning(self,'补偿未保存',str(exc));return
        self.values=next_values;self.problem='';self.refresh();self.changed.emit()
