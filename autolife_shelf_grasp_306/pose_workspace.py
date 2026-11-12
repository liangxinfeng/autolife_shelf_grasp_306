"""姿态与动作：一个模型、两步工作区，编辑预览与真机运行明确分开。"""
import json,time
from copy import deepcopy
from pathlib import Path
import numpy as np
from PyQt5 import QtCore as C,QtWidgets as Q
from control_center_joint_control.joint_definitions import ALL_JOINTS,JOINT_LABELS,JOINT_LIMITS,GROUPS,group_joint_names
from .pose_workspace_data import PoseStore,frame,compose,checked_targets
from .teaching_motion import compile_action,total_time
from .teaching_trajectory import plan,sample
from .teaching_library import TeachingLibrary


class PoseWorkspace(Q.QWidget):
    def __init__(self,owner):
        super().__init__();self.owner=owner;self.studio=owner.studio;self.model=None;self.tcp_axes=None;self.tcp_geometry=None
        self.store=PoseStore(self.studio.library.directory);self.actions=TeachingLibrary(self.studio.library.directory)
        self.edits={};self.rows=[];self.dirty=False;self.loaded='';self.revision=None;self.preview=None
        self.playing=False;self.elapsed=0.;self.last_tick=time.monotonic();self.ik_proc=None;self.ik_pending=None
        self.ik_session=None;self.ik_result=None;self.mirror_result=None;self.ik_buffer=b'';self.changing=False;self.curves={}
        self.draft_path=self.studio.library.directory/'manual_workspace_draft.json';self.draft_checked=False
        layout=Q.QVBoxLayout(self);layout.setContentsMargins(12,10,12,10)
        title=Q.QLabel('姿态与动作');title.setObjectName('title');layout.addWidget(title)
        self.notice=Q.QLabel('① 调姿态并保存　→　② 选择姿态组合动作　→　预演或真机运行');self.notice.setWordWrap(True);self.notice.setFixedHeight(54);layout.addWidget(self.notice)
        split=Q.QSplitter(C.Qt.Horizontal);split.setChildrenCollapsible(False);layout.addWidget(split,1)
        left=Q.QWidget();self.model_box=Q.QVBoxLayout(left);self.model_box.setContentsMargins(0,0,8,0)
        self.placeholder=Q.QLabel('进入页面后加载本机模型；不会自动使能');self.placeholder.setAlignment(C.Qt.AlignCenter);self.model_box.addWidget(self.placeholder,1)
        self.live=Q.QLabel('等待关节反馈');self.live.setWordWrap(True);self.model_box.addWidget(self.live)
        self.legend=Q.QLabel('实色＝实机当前　蓝色＝编辑/播放目标\n拖动空白旋转视角；点击“调关节”可直接点选模型。');self.legend.setWordWrap(True);self.model_box.addWidget(self.legend)
        left.setMinimumWidth(420);split.addWidget(left)
        self.tools=Q.QTabWidget();split.addWidget(self.tools);self.tools.setMinimumWidth(390);split.setSizes([720,460])
        self.pose_page=Q.QWidget();scroll=Q.QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(self.pose_page);scroll.setFrameShape(Q.QFrame.NoFrame);self.tools.addTab(scroll,'① 调姿态');p=Q.QVBoxLayout(self.pose_page)
        self.mode=Q.QTabWidget();p.addWidget(self.mode)
        joint=Q.QWidget();j=Q.QVBoxLayout(joint);j.setAlignment(C.Qt.AlignTop);self.mode.addTab(joint,'关节')
        row=Q.QHBoxLayout();self.group=Q.QComboBox();self.joint=Q.QComboBox()
        for key,label in [('right_arm','右臂与右爪'),('left_arm','左臂与左爪'),('leg_waist','腰腿'),('neck','头部')]:self.group.addItem(label,key)
        row.addWidget(self.group);row.addWidget(self.joint,1);j.addLayout(row)
        self.actual=Q.QLabel('实测 —');j.addWidget(self.actual)
        row=Q.QHBoxLayout();self.slider=Q.QSlider(C.Qt.Horizontal);self.value=self.spin(-180,360,.5,' °');row.addWidget(self.slider,1);row.addWidget(self.value);j.addLayout(row)
        self.joint_go=self.button('执行选中关节',self.execute_joint);self.row(j,self.joint_go,self.button('恢复当前值',self.clear_edits))
        self.value.valueChanged.connect(self.numeric);self.slider.valueChanged.connect(lambda v:self.numeric(v/10))
        self.group.currentIndexChanged.connect(self.group_changed);self.joint.currentIndexChanged.connect(self.joint_changed)
        self.make_ik();self.make_height();self.make_mirror()
        row=Q.QHBoxLayout();row.addWidget(Q.QLabel('姿态 / 复位速度'));self.manual_speed=self.spin(25,100,5,' %');self.manual_speed.setValue(self.pose_speed());self.manual_speed.setToolTip('已保存姿态与抓取页全身复位共用；25～100%，不改变原厂电机参数');self.manual_speed.valueChanged.connect(self.save_pose_speed);row.addWidget(self.manual_speed);p.addLayout(row)
        library=Q.QGroupBox('姿态库');form=Q.QVBoxLayout(library);p.addWidget(library)
        self.pose_list=Q.QComboBox();self.pose_list.setMinimumContentsLength(12);form.addWidget(self.pose_list)
        self.pose_use_object=Q.QCheckBox('执行时使用物体参照纠偏')
        self.pose_use_object.setToolTip('取消勾选：直接执行保存的固定姿态，不拍摄或识别物体；不修改已保存的绑定。')
        form.addWidget(self.pose_use_object)
        self.pose_list.currentTextChanged.connect(self.update_pose_binding)
        self.row(form,self.button('预览姿态',self.load_pose),self.button('执行姿态',self.execute_pose),self.button('删除',self.delete_pose))
        self.pose_name=Q.QLineEdit();self.pose_name.setPlaceholderText('姿态名称，如：右手准备位');self.pose_name.setMaxLength(32);form.addWidget(self.pose_name)
        self.save_source=Q.QComboBox();self.save_source.addItem('保存实机姿态＋已执行的夹爪目标',False);self.save_source.addItem('保存蓝色编辑预览',True);form.addWidget(self.save_source)
        self.row(form,self.button('保存姿态',self.save_pose),self.button('加入动作串 →',self.add_pose))
        from .visual_action_panel import VisualPanel
        self.visual_panel=VisualPanel(self);p.insertWidget(0,self.visual_panel)
        self.row(form,self.button('选中姿态设为抓取复位',self.save_selected_reset),
                 self.button('实机当前姿态设为复位',self.save_reset))
        self.reset_note=Q.QLabel();self.reset_note.setWordWrap(True);form.addWidget(self.reset_note);self.refresh_reset()
        tip=Q.QLabel('保存全身姿态（含夹爪），便于完整复现。滑块/模型只编辑预览；单关节执行不发送其他组的运动目标。');tip.setWordWrap(True);p.addWidget(tip);p.addStretch()
        self.make_sequence();self.tools.currentChanged.connect(self.page_changed);self.mode.currentChanged.connect(self.mode_changed)
        self.stop_button=self.button('停止当前动作',self.stop);self.stop_button.setObjectName('stop');layout.addWidget(self.stop_button)
        self.timer=C.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(100)
        self.studio.completed.connect(self.completed)
        self.setStyleSheet('QGroupBox{font-weight:600;margin-top:8px;padding-top:10px;} QPushButton{padding:6px 10px;min-height:24px;} QLineEdit,QComboBox,QDoubleSpinBox{padding:5px;} QListWidget{background:#101b25;color:#e8eff6;border:1px solid #34505e;border-radius:6px;} QListWidget::item{padding:9px;} QListWidget::item:selected{background:#245867;}')
        self.group_changed();self.reload_poses();self.reload_actions();self.fit_mode()

    def spin(self,lo,hi,step,suffix):
        v=Q.QDoubleSpinBox();v.setRange(lo,hi);v.setSingleStep(step);v.setDecimals(1);v.setSuffix(suffix);v.setKeyboardTracking(False);return v
    def button(self,label,fn):
        b=Q.QPushButton(label);b.clicked.connect(lambda _=False:self.guard(fn));return b
    def row(self,layout,*widgets):
        row=Q.QHBoxLayout()
        for w in widgets:row.addWidget(w)
        layout.addLayout(row)
    def message(self,text,error=False):self.notice.setText(text);self.notice.setToolTip(text);self.notice.setStyleSheet('color:#ff9a8d' if error else 'color:#81d8c3')
    def guard(self,fn):
        try:return fn()
        except Exception as exc:self.message(str(exc),True)
    def available(self):
        if self.studio.moving() or self.studio.editor.remote_locked():raise ValueError('请先结束当前动作或VR示教')
        if self.owner.busy or self.owner.motion or self.owner.workflow.active or self.owner.automatic_active() or self.owner.navigation.inflight:raise ValueError('当前任务尚未结束')
    def capture(self):
        self.available();return self.studio.feedback.capture()
    def ensure(self):
        self.studio.ensure()
        if not self.model:
            from control_center_joint_control.robot_view import RobotViewPanel
            from ament_index_python.packages import get_package_share_directory
            path=Path(get_package_share_directory('control_center_joint_control'))/'urdf/robot_v3_1.urdf'
            self.model=RobotViewPanel(str(path));self.placeholder.hide();self.model_box.insertWidget(0,self.model,1)
            from .pose_tcp_axes import TcpAxesOverlay
            from .pose_geometry import ModelGeometry
            self.tcp_axes=TcpAxesOverlay(self.model.view);self.tcp_geometry=ModelGeometry(self.studio.cfg['urdf_path'])
            self.tcp_tools={side:self.tcp_geometry.tool_transform(side) for side in ('left','right')}
            self.model.view.setMinimumSize(340,260)
            header=self.model.layout().itemAt(0).layout();second=Q.QHBoxLayout()
            while header.count()>4:
                item=header.takeAt(4)
                if item.widget():second.addWidget(item.widget())
            self.model.layout().insertLayout(1,second);self.model.adjust_card.hide();self.model.hint.setWordWrap(True)
            self.model.joint_selected.connect(self.select_model_joint);self.model.joint_target_changed.connect(self.model_edit)
            from .grip_width import load_curve
            for side in ('left','right'):
                widths,positions,_=load_curve({**self.studio.cfg,'arm':side});self.curves['Joint_'+side.title()+'_Gripper']=(widths,positions)
            self.joint_changed()
        if not self.draft_checked:
            self.draft_checked=True
            if self.draft_path.exists():
                try:
                    d=json.loads(self.draft_path.read_text('utf8'));d=compose(d['name'],d['frames'],d.get('speed',70))
                    if Q.QMessageBox.question(self,'恢复动作草稿','恢复上次未保存的手动动作串？',Q.QMessageBox.Yes|Q.QMessageBox.No,Q.QMessageBox.Yes)==Q.QMessageBox.Yes:
                        self.rows=d['frames'];self.sequence_name.setText(d['name']);self.overall_speed.setValue(100);self.dirty=True;self.render_rows()
                except Exception as exc:self.message('草稿未加载：'+str(exc),True)
        self.reload_poses();self.reload_actions()
    def group_changed(self):
        key=self.group.currentData();names=list(group_joint_names(key))
        if key in ('left_arm','right_arm'):names.append('Joint_'+key.split('_')[0].title()+'_Gripper')
        self.joint.blockSignals(True);self.joint.clear()
        for n in names:self.joint.addItem(JOINT_LABELS.get(n,n),n)
        self.joint.blockSignals(False);self.joint_changed()
    def joint_changed(self):
        n=self.joint.currentData()
        if not n:return
        lo,hi=JOINT_LIMITS[n];unit=' °'
        if n in self.curves:lo,hi=float(self.curves[n][0][0]),float(self.curves[n][0][-1]);unit=' mm'
        self.changing=True;self.value.setRange(lo,hi);self.value.setSuffix(unit);self.slider.setRange(round(lo*10),round(hi*10));self.changing=False
        if self.model:self.model.view.set_selected_joint(n)
        self.update_joint()
    def display_value(self,n,v):
        if n not in self.curves:return v
        w,p=self.curves[n];return float(np.interp(v,p[::-1],w[::-1]))
    def raw_value(self,n,v):
        if n not in self.curves:return v
        w,p=self.curves[n];return float(np.interp(v,w,p))
    def update_joint(self):
        n=self.joint.currentData();values=self.studio.feedback.values
        if n not in values:return
        actual=self.display_value(n,values[n]);target=self.display_value(n,self.edits.get(n,values[n]));unit='mm' if n in self.curves else '°'
        held=getattr(self.studio.engine,'gripper_targets',{}).get(n)
        extra=f'　保持目标 {self.display_value(n,held):.1f} {unit}' if held is not None else ''
        self.actual.setText(f'实测 {actual:.1f} {unit}　→　预览 {target:.1f} {unit}'+extra)
        self.changing=True;self.value.setValue(target);self.slider.setValue(round(target*10));self.changing=False
    def numeric(self,value):
        if self.changing:return
        if self.studio.moving() or self.ik_pending:return
        n=self.joint.currentData()
        if n not in self.studio.feedback.values:return
        self.end_preview();self.edits[n]=self.raw_value(n,value);self.ik_result=None;self.update_joint()
    def select_model_joint(self,name):
        if self.tools.currentIndex()!=0 or self.mode.currentIndex()!=0:return
        for i in range(self.group.count()):
            key=self.group.itemData(i);names=list(group_joint_names(key))
            if key in ('left_arm','right_arm'):names.append('Joint_'+key.split('_')[0].title()+'_Gripper')
            if name in names:self.group.setCurrentIndex(i);self.joint.setCurrentIndex(self.joint.findData(name));break
    def model_edit(self,name,value):
        if self.tools.currentIndex()!=0 or self.mode.currentIndex()!=0 or self.studio.moving() or self.ik_pending:return
        if self.guard(lambda:checked_targets({name:value})) is None:return
        self.edits[name]=value;self.select_model_joint(name);self.update_joint()
    def clear_edits(self):
        self.edits={};self.ik_result=None;self.end_preview();self.update_joint()
    def confirm(self,text):return Q.QMessageBox.question(self,'真机执行',text+'\n确认运动范围净空，执行期间可随时点击停止。',Q.QMessageBox.Yes|Q.QMessageBox.No,Q.QMessageBox.No)==Q.QMessageBox.Yes
    def manual(self,values,source=None,profile=''):
        source=source or self.capture();self.studio._acquire()
        try:
            from .pose_workspace_motion import begin
            (self.owner.run/'stop').unlink(missing_ok=True)
            begin(self.studio.engine,'manual-pose-'+str(time.time_ns()),values,source,self.manual_speed.value()/100.,profile)
        except Exception:
            if self.studio.engine.bridge is None:self.studio._unlock()
            raise
        self.message('已提交选中目标；复用当前保持会话。' if getattr(self.studio.engine,'manual_session',False) else '首次执行：正在取得原厂保持会话，随后只执行选中目标。')
    def execute_joint(self):
        source=self.capture();n=self.joint.currentData();v=self.raw_value(n,self.value.value())
        self.manual({n:v},source)
    def reload_poses(self):
        records=self.store.read()
        for box in (self.pose_list,getattr(self,'available_pose',None)):
            if box is None:continue
            name=box.currentText();box.blockSignals(True);box.clear();box.addItems(sorted(records));box.setCurrentText(name);box.blockSignals(False)
        self.update_pose_binding()
    def update_pose_binding(self,*_):
        name=self.pose_list.currentText();bound=bool(name and self.store.binding(name))
        if getattr(self,'_binding_selection',None)!=name:
            self.pose_use_object.setChecked(bound);self._binding_selection=name
        self.pose_use_object.setEnabled(bound)
        if not bound:self.pose_use_object.setChecked(False)
    def load_pose(self):
        self.available();name=self.pose_list.currentText()
        if not name:raise ValueError('请先保存姿态')
        self.end_preview();self.edits=self.store.read()[name];self.pose_name.setText(name);self.visual_panel.select_binding(self.store.binding(name));self.update_joint();self.message('预览姿态：'+name+'；尚未驱动机器人')
    def save_pose(self):
        q=self.capture();name=self.pose_name.text().strip()
        if not name:raise ValueError('请填写姿态名称')
        # 实机身体姿态仍取反馈；夹爪持物时保存用户已下发的期望开度。
        q.update(getattr(self.studio.engine,'gripper_targets',{}))
        if self.save_source.currentData():q.update(self.edits)
        exists=name in self.store.read()
        if exists and Q.QMessageBox.question(self,'覆盖姿态','覆盖“'+name+'”？已保存动作串保留原快照，不会随之改变。')!=Q.QMessageBox.Yes:return
        snapshot=self.store.path.read_bytes() if self.store.path.exists() else b''
        def saved(bound):
            if (self.store.path.read_bytes() if self.store.path.exists() else b'')!=snapshot:
                self.message('保存期间姿态库被其他窗口更新，请重新保存；当前编辑保留',True);return
            self.store.save(name,q,exists,visual_binding=bound);self.reload_poses();self.pose_list.setCurrentText(name);self.available_pose.setCurrentText(name)
            self.message('姿态已保存：'+name+(' · 已关联物体初始参照，预览/执行时识别' if bound else ' · 普通姿态'))
            from .workspace_reset import sync_named_pose
            if sync_named_pose(self.owner.root,self.studio.cfg,name,q):
                self.reset_changed();self.message('姿态与抓取复位目标已同步更新：'+name+'；复位始终使用固定全身姿态')
        self.visual_panel.bind(q,saved)
    def delete_pose(self):
        self.available();name=self.pose_list.currentText()
        if not name:return
        if Q.QMessageBox.question(self,'删除姿态','删除“'+name+'”？已保存动作串不受影响。')!=Q.QMessageBox.Yes:return
        self.store.delete(name);self.reload_poses();self.message('姿态已删除')
    def pose_speed(self):
        from .pose_execution_profile import read_speed
        return read_speed(self.owner.root)
    def save_pose_speed(self,value):
        from .pose_execution_profile import save_speed
        save_speed(self.owner.root,value)
    def refresh_reset(self):
        from .workspace_reset import reset_path
        path=reset_path(self.owner.root)
        if path.exists():
            name=json.loads(path.read_text('utf-8')).get('reference',{}).get('pose_name')
            self.reset_note.setText('抓取页复位目标：'+(name+'（已保存全身姿态）' if name else '实机全身快照')+'；速度与本页共用')
        else:self.reset_note.setText('尚未设置复位目标：选择已保存姿态后点击“设为抓取复位”，或保存实机当前姿态。')
    def reset_changed(self):
        from .core import atomic_json
        self.owner.invalidate_plan()
        atomic_json(self.owner.run/'prepare_result.json',dict(status='reset_target_changed'))
        self.refresh_reset()
    def save_selected_reset(self):
        self.available();name=self.pose_list.currentText()
        if not name:raise ValueError('请先选择已保存姿态')
        from .workspace_reset import save
        save(self.owner.root,self.studio.cfg,self.store.read()[name],pose_name=name)
        self.reset_changed();self.message('抓取复位已设为“'+name+'”；仅保存，未执行运动。')
    def save_reset(self):
        from .workspace_reset import save
        from .core import atomic_json
        q=self.capture()
        if Q.QMessageBox.question(self,'保存全身复位姿态','保存实机当前的双臂、头部、腰腿及左右夹爪状态，作为抓取页“复位并准备”的目标。\n此操作只保存，不运动；下次复位会恢复身体高度及夹爪开度。',Q.QMessageBox.Yes|Q.QMessageBox.No,Q.QMessageBox.No)!=Q.QMessageBox.Yes:return
        q=self.capture()  # 对话框期间可能有反馈变化，确认后重新取得实测值。
        save(self.owner.root,self.studio.cfg,q);self.owner.invalidate_plan()
        atomic_json(self.owner.run/'prepare_result.json',dict(status='reset_target_changed'))
        self.refresh_reset();self.message('已保存实机全身复位姿态；抓取页下次复位立即使用，无需重启。')
    def execute_pose(self):
        q=self.capture();name=self.pose_list.currentText()
        if not name:raise ValueError('请选择已保存姿态')
        from .pose_execution_profile import pose_record
        record=pose_record(self.store.read()[name],q,self.manual_speed.value(),name);bound=self.store.binding(name)
        if bound and self.pose_use_object.isChecked():record['visual_bindings']={'0':bound}
        self.execute_record(record)
    def execute_record(self,record):
        self.end_preview();(self.owner.run/'stop').unlink(missing_ok=True)
        self.studio.execute('pose-workspace-'+str(time.time_ns()),record,keep_session=True);self.message('已提交，等待控制器准备并执行。')

    def make_height(self):
        page=Q.QWidget();p=Q.QVBoxLayout(page);self.mode.addTab(page,'身体高度')
        self.height_input=self.spin(0,40,1,' cm');self.height_input.setValue(0);p.addWidget(Q.QLabel('相对直立的下降量：0＝站直，40＝下降40厘米'));p.addWidget(self.height_input)
        self.squat=Q.QComboBox();self.squat.addItem('正向下蹲',1);self.squat.addItem('反向下蹲',-1);p.addWidget(self.squat)
        p.addWidget(self.button('预览高度',self.preview_height));p.addWidget(self.button('执行身体高度',self.execute_height))
    def height_input_targets(self):
        from control_center_joint_control.body_height import body_height_targets
        return body_height_targets(self.height_input.value()/100.*self.squat.currentData(),allow_reverse=True)
    def preview_height(self):
        self.capture();self.edits=self.height_input_targets();self.message('身体高度已显示为蓝色预览，尚未运动')
    def execute_height(self):
        q=self.capture();targets=self.height_input_targets()
        self.manual(targets,q,'body_height')

    def make_sequence(self):
        page=Q.QWidget();p=Q.QVBoxLayout(page);self.tools.addTab(page,'② 编动作')
        self.action_list=Q.QComboBox();p.addWidget(self.action_list)
        self.row(p,self.button('载入',self.load_action),self.button('新建',self.new_action),self.button('删除动作',self.delete_action))
        self.sequence_name=Q.QLineEdit();self.sequence_name.setMaxLength(32);self.sequence_name.setPlaceholderText('动作串名称，如：拿起并放入篮子');self.sequence_name.textEdited.connect(self.mark_dirty);p.addWidget(self.sequence_name)
        self.available_pose=Q.QComboBox();self.row(p,self.available_pose,self.button('＋ 添加姿态',self.add_pose))
        self.sequence=Q.QListWidget();self.sequence.setMinimumHeight(150);self.sequence.currentRowChanged.connect(self.select_frame);self.sequence.itemChanged.connect(self.row_visual_changed);p.addWidget(self.sequence,1)
        hint=Q.QLabel('列表勾选＝启用该姿态的物体纠偏；取消后仍保留关联，可随时勾回。');hint.setWordWrap(True);p.addWidget(hint)
        self.row(p,self.button('上移',lambda:self.move_row(-1)),self.button('下移',lambda:self.move_row(1)),self.button('移除',self.remove_row))
        self.row_hold=self.spin(0,30,.1,' 秒');self.row_hold.setDecimals(1);self.row_hold.valueChanged.connect(self.change_row_hold)
        self.row(p,Q.QLabel('选中姿态停顿'),self.row_hold,self.button('全部连续通过',self.make_continuous))
        row=Q.QHBoxLayout();row.addWidget(Q.QLabel('整体速度'));self.overall_speed=self.spin(25,100,5,' %');self.overall_speed.setValue(70);self.overall_speed.valueChanged.connect(self.mark_dirty);row.addWidget(self.overall_speed);p.addLayout(row)
        self.row(p,self.button('保存动作串',self.save_action),self.button('▶ 模型预演',self.toggle_preview),self.button('真机运行…',self.run_action))
        self.visual_toggle=self.button('暂停选中姿态纠偏',self.clear_visual)
        self.row(p,self.button('定位参照并预览纠偏',self.preview_visual),self.visual_toggle)
        self.row(p,self.button('查看参照画面',self.visual_panel.show_image))
        self.playback=Q.QLabel('点击列表查看对应姿态。动作串保存后可直接用于任务流程。');self.playback.setWordWrap(True);p.addWidget(self.playback)
        note=Q.QLabel('停顿0秒＝连续通过，新加姿态默认0秒。夹爪需开合的交互点仍先到位再开合；末帧停稳。连续录制动作保留原时序，仍在遥操动作编排器编辑。');note.setWordWrap(True);p.addWidget(note)
    def mark_dirty(self,*_):self.dirty=True;self.end_preview()
    def render_rows(self,index=None):
        row=self.sequence.currentRow() if index is None else index;self.sequence.blockSignals(True);self.sequence.clear()
        for i,r in enumerate(self.rows):
            timing=f'停顿{r["hold_sec"]:g}秒' if r.get('hold_sec',0)>0 else '连续通过'
            item=Q.QListWidgetItem(f'{i+1:02d}　{r["label"]}　· {timing}'+(('　◎ 物体纠偏' if r.get('visual_enabled',True) else '　○ 纠偏已暂停') if r.get('visual_binding') else '　· 固定姿态'))
            if r.get('visual_binding'):
                item.setFlags(item.flags()|C.Qt.ItemIsUserCheckable);item.setCheckState(C.Qt.Checked if r.get('visual_enabled',True) else C.Qt.Unchecked)
                item.setToolTip('仅切换此动作中这一帧的纠偏；保留原始物体关联，不改变姿态库。')
            self.sequence.addItem(item)
        self.sequence.setCurrentRow(min(max(0,row),len(self.rows)-1));self.sequence.blockSignals(False);self.select_frame(self.sequence.currentRow())
    def add_pose(self):
        self.available();name=self.pose_list.currentText() if self.tools.currentIndex()==0 else self.available_pose.currentText()
        if not name:raise ValueError('先保存一个姿态，再添加到动作串')
        if len(self.rows)>=100:raise ValueError('单个动作串最多100个姿态')
        row=frame(name,self.store.read()[name]);bound=self.store.binding(name)
        if bound:row['visual_binding']=bound
        self.rows.append(row);self.mark_dirty();self.tools.setCurrentIndex(1);self.render_rows(len(self.rows)-1)
    def move_row(self,d):
        i=self.sequence.currentRow();j=i+d
        if 0<=i<len(self.rows) and 0<=j<len(self.rows):self.rows[i],self.rows[j]=self.rows[j],self.rows[i];self.mark_dirty();self.render_rows(j)
    def remove_row(self):
        i=self.sequence.currentRow()
        if 0<=i<len(self.rows):self.rows.pop(i);self.mark_dirty();self.render_rows(i)
    def select_frame(self,index):
        self.row_hold.blockSignals(True);self.row_hold.setEnabled(0<=index<len(self.rows))
        self.row_hold.setValue(self.rows[index].get('hold_sec',0.) if 0<=index<len(self.rows) else 0.);self.row_hold.blockSignals(False)
        bound=0<=index<len(self.rows) and self.rows[index].get('visual_binding')
        self.visual_toggle.setEnabled(bool(bound))
        self.visual_toggle.setText('恢复选中姿态纠偏' if bound and not self.rows[index].get('visual_enabled',True) else '暂停选中姿态纠偏')
        if 0<=index<len(self.rows):self.playback.setText(f'选中：{self.rows[index]["label"]}；蓝色虚影仅预览，不运动。')
    def row_visual_changed(self,item):
        i=self.sequence.row(item)
        if not 0<=i<len(self.rows) or not self.rows[i].get('visual_binding'):return
        try:
            self.available();self.rows[i]['visual_enabled']=item.checkState()==C.Qt.Checked
            self.mark_dirty();self.render_rows(i)
        except Exception as exc:self.render_rows(i);self.message(str(exc),True)
    def change_row_hold(self,value):
        i=self.sequence.currentRow()
        if not 0<=i<len(self.rows):return
        try:
            self.available();self.rows[i]['hold_sec']=float(value);self.mark_dirty();self.render_rows(i)
        except Exception as exc:self.select_frame(i);self.message(str(exc),True)
    def make_continuous(self):
        self.available()
        for row in self.rows:row['hold_sec']=0.
        self.mark_dirty();self.render_rows();self.message('中间姿态已设为连续通过；夹爪交互点仍先到位再开合。保存动作串后，任务调用同样生效。')
    def draft(self):return compose(self.sequence_name.text().strip(),self.rows,self.overall_speed.value())
    def reload_actions(self):
        selected=self.action_list.currentText();records=self.actions.records();self.action_list.clear()
        self.action_list.addItems(sorted(n for n,r in records.items() if 'trajectory' not in r));self.action_list.setCurrentText(selected)
    def discard(self):
        if not self.dirty:return True
        reply=Q.QMessageBox.question(self,'未保存动作串','是否先保存当前动作串？',Q.QMessageBox.Save|Q.QMessageBox.Discard|Q.QMessageBox.Cancel,Q.QMessageBox.Save)
        if reply==Q.QMessageBox.Cancel:return False
        if reply==Q.QMessageBox.Save:return self.save_action() is True
        return True
    def new_action(self):
        self.available()
        if not self.discard():return
        self.rows=[];self.loaded='';self.revision=None;self.sequence_name.clear();self.overall_speed.setRange(25,100);self.overall_speed.setValue(70);self.dirty=False;self.end_preview();self.render_rows();self.draft_path.unlink(missing_ok=True)
    def load_action(self):
        self.available();name=self.action_list.currentText()
        if not name:raise ValueError('请选择动作串')
        if not self.discard():return
        d=self.actions.load(name)
        if d.get('kind')=='trajectory':raise ValueError('连续轨迹请在遥操动作编排器编辑')
        self.rows=d['frames'];self.loaded=name;self.revision=self.actions.revision;self.sequence_name.setText(name)
        self.overall_speed.blockSignals(True);self.overall_speed.setRange(max(25,int(np.ceil(2500/min(r['speed_percent'] for r in self.rows)))),100);self.overall_speed.setValue(100);self.overall_speed.blockSignals(False)
        # 100%表示保持载入时的节奏；整体速度按比例缩放各帧，不抹去原节奏。
        self.dirty=False;self.end_preview();self.render_rows(0)
    def save_action(self):
        self.available();d=self.draft()
        if not d['name']:raise ValueError('请填写动作串名称')
        existing=self.actions.records();revision=self.revision if self.loaded==d['name'] else self.actions.revision
        if d['name'] in existing and (self.loaded!=d['name'] or 'trajectory' in existing[d['name']]):
            if 'trajectory' in existing[d['name']]:raise ValueError('此名称属于连续轨迹，请换一个动作串名称')
            if Q.QMessageBox.question(self,'覆盖动作串','覆盖“'+d['name']+'”？')!=Q.QMessageBox.Yes:return False
        self.actions.save(d,revision);self.loaded=d['name'];self.revision=self.actions.revision;self.dirty=False
        self.draft_path.unlink(missing_ok=True);self.reload_actions();self.action_list.setCurrentText(self.loaded);self.studio.editor.refresh_library();self.message('已保存；任务流程可选择：'+self.loaded);return True
    def delete_action(self):
        self.available();name=self.action_list.currentText()
        if not name:return
        if Q.QMessageBox.question(self,'删除动作串','删除“'+name+'”？引用该动作的任务需重新选择。')!=Q.QMessageBox.Yes:return
        self.actions.records();self.actions.delete(name,self.actions.revision);self.reload_actions();self.studio.editor.refresh_library()
        if self.loaded==name:self.loaded='';self.revision=None
    def run_action(self):
        q=self.capture();d=self.draft();record=compile_action(d,q)
        if self.confirm(f'执行当前动作串，共{len(self.rows)}个全身姿态，包含腰腿与夹爪。模型预演不能确认环境障碍。'):
            self.execute_record(record)
    def clear_visual(self):
        self.available();i=self.sequence.currentRow()
        if 0<=i<len(self.rows) and self.rows[i].get('visual_binding'):
            self.rows[i]['visual_enabled']=not self.rows[i].get('visual_enabled',True)
            self.mark_dirty();self.render_rows(i)
    def preview_visual(self):
        self.available();self.end_preview();self.studio.ensure()
        record=compile_action(self.draft(),self.capture())
        if not record.get('visual_bindings'):raise ValueError('请先添加已关联物体的姿态，普通动作使用模型预演即可')
        def done(data):
            try:
                if data.get('error'):raise ValueError(data['error'])
                r=data['result'];self.visual_panel.show_correction(r);self.preview=plan(r['record'],r['source'],self.studio.library.directory)
                self.elapsed=0.;self.playing=True;self.last_tick=time.monotonic()
                detail='；'.join(f'{v["name"]}：XYZ '+ '/'.join(f'{x:+.1f}' for x in v['offset_mm'])+' mm' for v in r['diagnostics'])
                self.message('纠偏预演（不运动）：'+detail+'。真机运行会重新拍摄定位。')
                if self.model:self.model.set_action_preview_active(True)
            except Exception as exc:self.message(str(exc),True)
            finally:
                if self.studio.engine.bridge is None:self.studio._unlock()
        try:self.studio.visual.request('visual_correct',done,record=record)
        except Exception:
            if self.studio.engine.bridge is None:self.studio._unlock()
            raise
    def toggle_preview(self):
        if self.playing:self.end_preview();return
        self.available();q=self.studio.feedback.values
        if not q:raise ValueError('等待实机反馈后预演，不能把未知关节补零')
        record=compile_action(self.draft(),q)
        if record.get('visual_bindings'):self.preview_visual();return
        self.preview=plan(record,q,self.studio.library.directory);self.elapsed=0.;self.playing=True;self.last_tick=time.monotonic()
        if self.model:self.model.set_action_preview_active(True)
    def end_preview(self):
        self.preview=None;self.playing=False
        if self.model:self.model.set_action_preview_active(False)
    def page_changed(self,*_):self.end_preview();self.cancel_ik()
    def fit_mode(self):self.mode.setFixedHeight(max(200,min(470,self.mode.currentWidget().sizeHint().height()+42)))
    def mode_changed(self,*_):
        self.cancel_ik();self.ik_session=None;self.ik_result=None;self.mirror_result=None
        if hasattr(self,'mirror_go'):self.mirror_go.setEnabled(False)
        self.fit_mode()
        if self.model:self.model.camera_mode_button.click()
        self.update_tcp_axes()

    def update_tcp_axes(self):
        if not self.tcp_axes:return
        if self.tools.currentIndex()!=0 or self.mode.currentIndex()!=1:
            self.tcp_axes.set_frame(None);return
        target=self.ik_result or self.ik_session;side=self.side.currentData()
        if target:matrix=target['target'];label=self.side.currentText()+'末端目标'
        else:
            values=self.studio.feedback.values
            if not values:self.tcp_axes.set_frame(None);return
            matrix=self.tcp_geometry.frames(values)['Link_'+side.title()+'_Wrist_Lower_to_Gripper']@self.tcp_tools[side]
            label=self.side.currentText()+'当前末端'
        self.tcp_axes.set_frame(matrix,self.axis_frame.currentData(),label)
    def completed(self,token,ok,text):
        if token.startswith(('manual-pose-','pose-workspace-')):self.message(text,not ok);self.edits={};self.ik_result=None;self.ik_session=None
    def stop(self):
        self.cancel_ik();self.end_preview();self.owner.stop_motion();self.message('已请求停止当前动作，保持夹持状态')
    def tick(self):
        if not self.isVisible():return
        now=time.monotonic();dt=min(.25,max(0,now-self.last_tick));self.last_tick=now
        busy=self.studio.moving() or self.studio.editor.remote_locked() or bool(self.ik_pending)
        self.tools.setEnabled(not busy);self.stop_button.setEnabled(True)
        f=self.studio.feedback
        self.live.setText(('实时反馈正常　|　保持会话已就绪，点击执行即可运动' if getattr(self.studio.engine,'manual_session',False) else '实时反馈正常　|　编辑仅预览，点击执行才运动') if f.fresh() else '等待新鲜关节反馈，暂不能执行')
        target={**f.values,**self.edits}
        if self.tools.currentIndex()==1 and 0<=self.sequence.currentRow()<len(self.rows):target=self.rows[self.sequence.currentRow()]['joints_deg']
        if self.preview:
            record,segments,phases=self.preview
            if self.playing:self.elapsed+=dt
            target,i,done=sample(record,segments,self.elapsed);self.playback.setText(f'模型预演　{min(self.elapsed,total_time(record)):.1f} / {total_time(record):.1f} 秒　·　{phases[i][1]}')
            if done:self.playing=False
        if self.model:
            self.model.joint_mode_button.setEnabled(not busy and self.tools.currentIndex()==0 and self.mode.currentIndex()==0)
            if busy and self.model.joint_mode_button.isChecked():self.model.camera_mode_button.click()
            self.model.set_connected(f.fresh())
            if f.values:self.model.set_joint_states(f.values,target)
        self.update_tcp_axes()
        if not self.slider.isSliderDown() and not self.value.hasFocus():self.update_joint()
        if self.ik_pending and now-self.ik_pending['started']>25:self.cancel_ik();self.message('IK计算超时，未发送动作',True)
    def shutdown(self):
        self.timer.stop();self.cancel_ik()
        if self.dirty and self.rows:
            from .core import atomic_json
            self.draft_path.parent.mkdir(parents=True,exist_ok=True);atomic_json(self.draft_path,dict(name=self.sequence_name.text(),frames=self.rows,speed=self.overall_speed.value()))

    def make_mirror(self):
        page=Q.QWidget();p=Q.QVBoxLayout(page);self.mode.addTab(page,'双臂镜像')
        self.mirror_side=Q.QComboBox();self.mirror_side.addItem('左臂 → 右臂','right');self.mirror_side.addItem('右臂 → 左臂','left')
        p.addWidget(self.mirror_side);self.mirror_side.currentIndexChanged.connect(self.clear_mirror)
        note=Q.QLabel('以来源手臂的实测姿态为准；先预览，再执行。夹爪开度保持不变。');note.setWordWrap(True);p.addWidget(note)
        p.addWidget(self.button('生成镜像预览',self.preview_mirror))
        self.mirror_text=Q.QLabel('选择镜像方向，点击生成预览；此时机器人不会运动。');self.mirror_text.setWordWrap(True);p.addWidget(self.mirror_text)
        self.mirror_go=self.button('执行镜像姿态',self.execute_mirror);self.mirror_go.setEnabled(False);p.addWidget(self.mirror_go)
    def clear_mirror(self,*_):
        self.cancel_ik();self.mirror_result=None;self.mirror_go.setEnabled(False);self.edits={}
    def preview_mirror(self):
        q=self.capture();self.end_preview();self.mirror_result=None;self.mirror_go.setEnabled(False);self.edits={}
        self.mirror_text.setText('按本机模型计算镜像并检查手臂路径…')
        self.request_ik('mirror',q,target_side=self.mirror_side.currentData())
    def execute_mirror(self):
        q=self.capture();r=self.mirror_result
        if not r:raise ValueError('请先生成镜像预览')
        if max(abs(q[n]-v) for n,v in r['source'].items())>1.:raise ValueError('实机姿态已变化，请重新生成镜像预览')
        self.manual(r['joints'],q);self.mirror_result=None;self.mirror_go.setEnabled(False)
    def make_ik(self):
        page=Q.QWidget();p=Q.QVBoxLayout(page);p.setAlignment(C.Qt.AlignTop);self.mode.addTab(page,'末端IK')
        self.side=Q.QComboBox();self.side.addItem('右手','right');self.side.addItem('左手','left');self.side.currentIndexChanged.connect(self.mode_changed)
        self.row(p,self.side,self.button('取当前末端',self.capture_ik))
        self.axis_frame=Q.QComboBox();self.axis_frame.addItem('机器人方向（X前 / Y左 / Z上）',False);self.axis_frame.addItem('夹爪自身方向',True);p.addWidget(self.axis_frame)
        self.axis_frame.currentIndexChanged.connect(self.update_tcp_axes)
        self.xyz_step=self.spin(1,30,1,' mm');self.xyz_step.setValue(5);self.rpy_step=self.spin(.5,10,.5,' °');self.rpy_step.setValue(2)
        self.row(p,Q.QLabel('位移步长'),self.xyz_step,Q.QLabel('角度步长'),self.rpy_step)
        grid=Q.QGridLayout();p.addLayout(grid)
        for i,axis in enumerate('XYZ'):
            grid.addWidget(self.button(axis+' −',lambda a=i:self.nudge_ik(a,-1,False)),0,i)
            grid.addWidget(self.button(axis+' ＋',lambda a=i:self.nudge_ik(a,1,False)),1,i)
            grid.addWidget(self.button('绕'+axis+' −',lambda a=i:self.nudge_ik(a,-1,True)),2,i)
            grid.addWidget(self.button('绕'+axis+' ＋',lambda a=i:self.nudge_ik(a,1,True)),3,i)
        self.ik_text=Q.QLabel('直接点击XYZ或旋转按钮，自动以当前末端开始预览；固定腰腿、头部和另一只手。');self.ik_text.setWordWrap(True);self.ik_text.setFixedHeight(76);self.ik_text.setAlignment(C.Qt.AlignTop);p.addWidget(self.ik_text)
        p.addWidget(self.button('执行IK预览',self.execute_ik))
    def request_ik(self,operation,source,**extra):
        if self.ik_pending:raise ValueError('正在计算，请稍候')
        if self.ik_proc is None:
            p=C.QProcess(self);self.ik_proc=p;p.setProcessChannelMode(C.QProcess.SeparateChannels)
            p.readyReadStandardOutput.connect(self.read_ik);p.readyReadStandardError.connect(lambda:self.ik_proc and self.ik_proc.readAllStandardError())
            p.errorOccurred.connect(self.ik_error);p.finished.connect(self.ik_finished)
            p.start(self.studio.cfg['python'],['-u','-m','autolife_shelf_grasp_306.pose_ik_worker'])
        side=extra.pop('target_side',self.side.currentData())
        cfg={**self.studio.cfg,'arm':side};request=dict(operation=operation,config=cfg,source=source,**extra)
        self.ik_pending=dict(operation=operation,source=deepcopy(source),side=side,started=time.monotonic())
        self.ik_proc.write((json.dumps(request,allow_nan=False)+'\n').encode());self.ik_text.setText('计算中；不发送运动指令…')
    def capture_ik(self):
        q=self.capture();self.edits={};self.ik_session=None;self.ik_result=None;self.ik_after_capture=None;self.request_ik('capture',q)
    def nudge_ik(self,axis,sign,rotate):
        from scipy.spatial.transform import Rotation
        self.available()
        if self.ik_pending:raise ValueError('正在计算，请稍候')
        from .pose_workspace_motion import source_changed
        q=self.capture()
        if not self.ik_session or source_changed(q,self.ik_session['source']):
            # 只自动刷新预览参考，不自动执行；原点击在FK返回后应用一次。
            self.edits={};self.ik_session=None;self.ik_result=None
            self.ik_after_capture=(axis,sign,rotate)
            self.request_ik('capture',q);return
        source=self.ik_session['source']
        target=np.array((self.ik_result or self.ik_session)['target'],float);local=self.axis_frame.currentData()
        direction=np.eye(3)[axis]
        if rotate:
            delta=Rotation.from_rotvec(direction*np.deg2rad(sign*self.rpy_step.value())).as_matrix()
            target[:3,:3]=target[:3,:3]@delta if local else delta@target[:3,:3]
        else:target[:3,3]+=(target[:3,:3]@direction if local else direction)*sign*self.xyz_step.value()/1000.
        seed={**source,**(self.ik_result or {}).get('joints',{})}
        self.request_ik('solve',source,target=target.tolist(),seed=seed)
    def read_ik(self):
        if not self.ik_proc:return
        self.ik_buffer+=bytes(self.ik_proc.readAllStandardOutput())
        while b'\n' in self.ik_buffer:
            line,self.ik_buffer=self.ik_buffer.split(b'\n',1)
            if not line.startswith(b'POSE_IK=') or not self.ik_pending:continue
            pending=self.ik_pending;self.ik_pending=None
            try:
                response=json.loads(line[8:])
                if 'error' in response:raise ValueError(response['error'])
                r=response['result']
                if r['source']!=pending['source']:raise ValueError('IK参考不匹配')
                if pending['operation']=='mirror':
                    names=set(group_joint_names(pending['side']+'_arm'))
                    if set(r['joints'])!=names:raise ValueError('镜像结果含非目标侧手臂关节')
                    checked_targets(r['joints']);self.mirror_result=r;self.edits=dict(r['joints'])
                    self.mirror_text.setText('镜像预览已生成；只调整目标侧7轴，腰腿、头部、来源臂及夹爪开度不变。')
                    self.mirror_go.setEnabled(True);continue
                if pending['operation']=='capture':self.ik_session=r;self.ik_result=None
                else:
                    names=set(group_joint_names(pending['side']+'_arm'))
                    if set(r['joints'])!=names:raise ValueError('IK结果包含非所选手臂关节')
                    checked_targets(r['joints']);self.ik_result=r;self.edits=dict(r['joints'])
                xyz=np.array(r['target'])[:3,3]*1000
                self.ik_text.setText(f'基座系末端：X {xyz[0]:.1f} / Y {xyz[1]:.1f} / Z {xyz[2]:.1f} mm\n'+('已取当前末端；可调位置与朝向。' if pending['operation']=='capture' else f'可达：位置误差 {r["position_error_mm"]:.2f} mm，朝向误差 {r["angle_error_deg"]:.2f}°'))
                notes=r.get('fixed_collision_notes',[])
                if notes:self.ik_text.setText(self.ik_text.text()+'\n固定部件模型提示（本次不运动）：'+'；'.join(notes))
                self.ik_text.setToolTip(self.ik_text.text())
                if pending['operation']=='capture' and getattr(self,'ik_after_capture',None):
                    click=self.ik_after_capture;self.ik_after_capture=None
                    self.nudge_ik(*click)
            except Exception as exc:
                self.ik_after_capture=None;self.ik_result=None;self.mirror_result=None;self.mirror_go.setEnabled(False)
                self.ik_text.setText(str(exc));self.mirror_text.setText(str(exc));self.message(str(exc),True)
    def execute_ik(self):
        q=self.capture();r=self.ik_result
        if not r:raise ValueError('请先计算出可达的IK预览')
        from .pose_workspace_motion import source_changed
        if source_changed(q,r['source']):raise ValueError('实机姿态已变化，请点击方向键重新生成预览')
        self.manual(r['joints'],q)
    def ik_error(self,*_):
        if self.ik_pending:self.message('IK计算进程启动或通信失败，未发送动作',True)
        self.ik_pending=None;self.ik_after_capture=None
    def ik_finished(self,*_):
        if self.ik_pending:self.ik_error()
        if self.ik_proc:self.ik_proc.deleteLater();self.ik_proc=None
    def cancel_ik(self):
        p=self.ik_proc;self.ik_proc=None;self.ik_pending=None;self.ik_after_capture=None;self.ik_buffer=b''
        if p:
            p.blockSignals(True);p.kill();p.waitForFinished(500);p.deleteLater()
