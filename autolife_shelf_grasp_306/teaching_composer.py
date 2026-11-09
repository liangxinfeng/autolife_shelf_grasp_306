"""Small, passive teaching workspace. All model edits affect only the draft."""
import json,time
from copy import deepcopy
from PyQt5 import QtCore as C,QtWidgets as Q
from control_center_joint_control.joint_definitions import ALL_JOINTS,JOINT_LABELS,JOINT_LIMITS,GRIPPER_JOINTS
from .teaching_motion import normalize_draft,compile_action,stage_action,total_time
from .teaching_trajectory import build,sample
from .teaching_runtime import backend_state
from .teaching_controls import TeachingControls


class TeachingComposer(TeachingControls,Q.QWidget):
    def __init__(self,studio):
        super().__init__();self.studio=studio;self.model=None;self.draft={'name':'','frames':[]};self.dirty=False
        self.changing=False;self.undo_stack=[];self.redo_stack=[];self.preview=None;self.playing=False;self.elapsed=0.;self.last_tick=time.monotonic()
        self.revision=None;self.loaded_name='';self.autosave_due=0.;self.last_connected=None
        self.last_joint_edit=(None,0.)
        self.draft_path=studio.owner.root/'data/fixed_actions/teaching_unsaved.json'
        layout=Q.QVBoxLayout(self);layout.setContentsMargins(0,0,0,0);layout.setSpacing(8)
        title=Q.QLabel('遥操动作编排器');title.setStyleSheet('font-size:22px;font-weight:600;color:#e8eff6');layout.addWidget(title)
        line=Q.QHBoxLayout();self.start=self.button('开启遥操',lambda:self.guard(studio.start_vr));self.close_vr=self.button('停止并关闭',lambda:self.guard(studio.runtime.stop))
        line.addWidget(self.start);line.addWidget(self.close_vr);self.runtime_status=Q.QLabel();line.addWidget(self.runtime_status,1)
        self.copy_url=self.button('复制VR网址',self.copy_address);line.addWidget(self.copy_url);layout.addLayout(line)
        self.connection=Q.QLabel('后台运行，无弹出窗口；在头显网页使能后即可示教。');self.connection.setWordWrap(True);layout.addWidget(self.connection)
        self.runtime_status.setToolTip('双击查看后台日志');self.runtime_status.mouseDoubleClickEvent=lambda event:self.show_log()
        self.setup_recording_controls(layout)
        split=Q.QSplitter(C.Qt.Horizontal);layout.addWidget(split,1)
        left=Q.QWidget();self.left_layout=Q.QVBoxLayout(left);self.left_layout.setContentsMargins(0,0,4,0)
        library_row=Q.QHBoxLayout();self.actions=Q.QComboBox();self.actions.setMinimumWidth(120);self.actions.activated.connect(self.load_selected)
        library_row.addWidget(self.actions,1);library_row.addWidget(self.button('新建',self.new));library_row.addWidget(self.button('保存',lambda:self.guard(self.save)))
        library_row.addWidget(self.button('管理',self.manage_library))
        self.left_layout.addLayout(library_row)
        self.name=Q.QLineEdit();self.name.setPlaceholderText('动作串名称，例如：开门取饮料');self.name.setMaxLength(32);self.name.textEdited.connect(self.rename);self.left_layout.addWidget(self.name)
        self.frames=Q.QListWidget();self.frames.setMinimumHeight(155);self.frames.currentRowChanged.connect(self.select_frame);self.left_layout.addWidget(self.frames,1)
        capture_row=Q.QHBoxLayout();self.capture_button=self.button('＋ 记录当前姿态',lambda:self.capture(False));self.replace_button=self.button('替换实测',lambda:self.capture(True));capture_row.addWidget(self.capture_button,1);capture_row.addWidget(self.replace_button);self.left_layout.addLayout(capture_row)
        order=Q.QHBoxLayout()
        for label,fn in [('上移',lambda:self.move(-1)),('下移',lambda:self.move(1)),('复制',self.duplicate),('删除',self.remove),('撤销',self.undo),('重做',self.redo)]:order.addWidget(self.button(label,fn))
        self.left_layout.addLayout(order)
        self.properties=Q.QGroupBox('选中姿态');form=Q.QFormLayout(self.properties)
        self.frame_name=Q.QLineEdit();self.frame_name.setMaxLength(48);self.frame_name.editingFinished.connect(self.change_properties)
        self.hold=self.spin(0,30,.1,' 秒');self.speed=self.spin(25,100,5,' %');self.minimum=self.spin(.2,30,.1,' 秒')
        self.hold.setToolTip('0秒：普通姿态连续通过；有夹爪变化则完成开合后立即继续。')
        self.minimum.setToolTip('最短移动时间；系统会根据关节跨度自动延长，确保平滑。')
        for widget in (self.hold,self.speed,self.minimum):widget.editingFinished.connect(self.change_properties)
        form.addRow('姿态名称',self.frame_name);form.addRow('完成后停顿',self.hold);form.addRow('移动速度',self.speed);form.addRow('最短移动时间',self.minimum)
        self.ghost=Q.QCheckBox('显示这一姿态的虚影');self.ghost.toggled.connect(self.change_properties);form.addRow(self.ghost)
        self.sequence_note=Q.QLabel('顺序：关节到位 → 夹爪开合 → 停顿\n普通姿态停顿为0：平滑通过，不逐点停留');self.sequence_note.setWordWrap(True);self.sequence_note.setStyleSheet('color:#91a4b5;font-size:12px');form.addRow(self.sequence_note)
        self.left_layout.addWidget(self.properties)
        split.addWidget(left)
        right=Q.QWidget();self.right_layout=Q.QVBoxLayout(right);self.right_layout.setContentsMargins(4,0,0,0)
        self.model_placeholder=Q.QLabel('打开此页后加载实时机器人模型');self.model_placeholder.setAlignment(C.Qt.AlignCenter);self.right_layout.addWidget(self.model_placeholder,1)
        edit=Q.QHBoxLayout();self.joint=Q.QComboBox()
        for n in ALL_JOINTS:self.joint.addItem(JOINT_LABELS.get(n,n),n)
        self.joint.currentIndexChanged.connect(self.select_joint);self.angle=self.spin(-180,360,.5,' °');self.angle.setDecimals(2);self.angle.editingFinished.connect(self.numeric_joint)
        edit.addWidget(Q.QLabel('微调虚影'));edit.addWidget(self.joint,1);edit.addWidget(self.angle);self.right_layout.addLayout(edit)
        self.edit_hint=Q.QLabel('观察：拖动旋转视角。调关节：点击模型关节后拖动。微调只修改草稿，不驱动机器人。夹爪数值为原厂位置量。');self.edit_hint.setWordWrap(True);self.edit_hint.setStyleSheet('font-size:12px;color:#91a4b5');self.right_layout.addWidget(self.edit_hint)
        playback=Q.QHBoxLayout();self.play=self.button('▶ 模型预演',self.toggle_preview);self.stop_preview=self.button('结束预演',self.end_preview)
        self.execute_button=self.button('▶ 真机运行…',self.execute_current);self.stop_execution=self.button('停止真机',studio.stop)
        self.execute_button.setToolTip('运行当前动作；连续轨迹按录制时间播放。起点准备与结束停稳检查另计。')
        playback.addWidget(self.play);playback.addWidget(self.execute_button);playback.addWidget(self.stop_execution);playback.addWidget(self.stop_preview);playback.addStretch();self.right_layout.addLayout(playback)
        timeline=Q.QHBoxLayout();self.progress=Q.QSlider(C.Qt.Horizontal);self.progress.setRange(0,1000);self.progress.sliderMoved.connect(self.seek);timeline.addWidget(self.progress,1);self.time_label=Q.QLabel('0.00 / 0.00 秒');timeline.addWidget(self.time_label);self.right_layout.addLayout(timeline)
        self.timing_label=Q.QLabel();self.timing_label.setWordWrap(True);self.timing_label.setStyleSheet('color:#91cbd8;font-size:12px');self.right_layout.addWidget(self.timing_label)
        self.phase_label=Q.QLabel('真实模型为实测姿态，蓝色虚影为选中姿态或预演。');self.phase_label.setWordWrap(True);self.right_layout.addWidget(self.phase_label)
        split.addWidget(right);split.setSizes([340,680]);split.setChildrenCollapsible(False);left.setMinimumWidth(320);right.setMinimumWidth(440)
        self.notice=Q.QLabel('先开启遥操，再新建动作串；每到一个姿态，停稳后点击“记录当前姿态”。');self.notice.setWordWrap(True);layout.addWidget(self.notice)
        self.refresh_library();self.refresh_frames();self.update_runtime()
        self.setStyleSheet('QListWidget{border:1px solid #d9e3eb;border-radius:7px;background:#101b25;color:#e8eff6;} QListWidget::item{padding:9px 7px;} QListWidget::item:selected{background:#245867;color:#e8eff6;} QGroupBox{font-weight:600;} QPushButton{min-height:24px;padding:5px 8px;font-size:13px;} QLineEdit,QDoubleSpinBox,QComboBox{padding:5px;font-size:14px;}')
        Q.QShortcut('Ctrl+S',self,activated=lambda:self.guard(self.save));Q.QShortcut('Ctrl+Z',self,activated=self.undo)
        Q.QShortcut('Ctrl+Shift+Z',self,activated=self.redo)
    def button(self,text,fn):
        b=Q.QPushButton(text);b.clicked.connect(fn);return b
    def spin(self,lo,hi,step,suffix):
        w=Q.QDoubleSpinBox();w.setRange(lo,hi);w.setSingleStep(step);w.setDecimals(1);w.setSuffix(suffix);w.setKeyboardTracking(False);return w
    def guard(self,fn):
        try:return fn()
        except Exception as exc:self.message(str(exc),True);return None
    def message(self,text,error=False):self.notice.setText(str(text));self.notice.setStyleSheet('color:#ff9a8d' if error else 'color:#81d8c3')
    def ensure_model(self,path):
        if self.model:return
        from control_center_joint_control.robot_view import RobotViewPanel
        self.model=RobotViewPanel(path);self.model_placeholder.hide();self.right_layout.insertWidget(0,self.model,1)
        # Reflow the existing viewer header for a narrow on-robot screen.
        top=self.model.layout().itemAt(0).layout();second=Q.QHBoxLayout()
        for _ in range(top.count()-4):
            item=top.takeAt(4)
            if item.widget():second.addWidget(item.widget())
        second.addStretch();self.model.layout().insertLayout(1,second)
        self.model.adjust_card.hide() # One shared numeric editor includes jaws.
        self.model.hint.setWordWrap(True)
        self.model.joint_target_changed.connect(self.model_joint)
        self.model.joint_selected.connect(self.model_selected)
        self.model.target_toggle.toggled.connect(self.model_ghost)
        if self.draft_path.exists() and not self.draft['frames']:
            try:
                saved=normalize_draft(json.loads(self.draft_path.read_text('utf8')))
                if saved['frames'] and Q.QMessageBox.question(self,'恢复草稿','发现未保存的动作串草稿，是否恢复？')==Q.QMessageBox.Yes:
                    self.draft=saved;self.dirty=True;self.name.setText(saved['name']);self.refresh_frames(0)
            except (ValueError,OSError,KeyError):pass
    def snapshot(self):
        self.undo_stack.append(deepcopy(self.draft));self.undo_stack=self.undo_stack[-60:];self.redo_stack.clear();self.end_preview()
    def changed(self):
        self.dirty=True;self.preview=None;self.autosave_due=time.monotonic()+.7
    def persist_draft(self):
        if not self.dirty or not self.draft['frames']:return
        from .core import atomic_json
        self.draft_path.parent.mkdir(parents=True,exist_ok=True);atomic_json(self.draft_path,self.draft)
    def discard_ok(self):
        if not self.dirty:return True
        reply=Q.QMessageBox.question(self,'未保存的动作串','是否保存当前动作串？',Q.QMessageBox.Save|Q.QMessageBox.Discard|Q.QMessageBox.Cancel,Q.QMessageBox.Save)
        if reply==Q.QMessageBox.Cancel:return False
        if reply==Q.QMessageBox.Save:
            try:self.save()
            except Exception as exc:self.message(str(exc),True);return False
        return True
    def new(self):
        if not self.edit_allowed():return
        if not self.discard_ok():return
        self.end_preview();self.draft={'name':'','frames':[]};self.loaded_name='';self.dirty=False;self.undo_stack=[];self.redo_stack=[]
        if self.draft_path.exists():self.draft_path.unlink()
        self.name.clear();self.refresh_library();self.refresh_frames();self.message('新动作串已准备好，请遥操到第一个姿态后记录。')
    def rename(self,text):self.draft['name']=text;self.changed()
    def refresh_library(self):
        records=self.studio.library.records();self.revision=self.studio.library.revision
        self.actions.blockSignals(True);self.actions.clear();self.actions.addItem('选择已保存动作串…',None)
        for name in sorted(records):self.actions.addItem(name,name)
        i=self.actions.findData(self.loaded_name);self.actions.setCurrentIndex(max(0,i));self.actions.blockSignals(False)
    def load_selected(self,index):
        if not self.edit_allowed():return
        name=self.actions.itemData(index)
        if not name or not self.discard_ok():return
        def work():
            self.draft=self.studio.library.load(name);self.revision=self.studio.library.revision;self.loaded_name=name
            if self.draft_path.exists():self.draft_path.unlink()
            self.name.setText(name);self.dirty=False;self.undo_stack=[];self.redo_stack=[];self.end_preview();self.refresh_frames(0);self.message('已载入动作串；选中姿态即可微调。')
        self.guard(work)
    def save(self):
        if not self.edit_allowed():return
        self.change_properties();self.draft['name']=self.name.text().strip()
        if not self.draft['name']:raise ValueError('请先填写动作串名称')
        if self.loaded_name!=self.draft['name'] and self.actions.findData(self.draft['name'])>=0:
            if Q.QMessageBox.question(self,'覆盖动作串','同名动作串已存在，是否覆盖？')!=Q.QMessageBox.Yes:return
        record=self.studio.library.save(self.draft,self.revision);self.loaded_name=self.draft['name'];self.dirty=False;self.refresh_library()
        if self.draft_path.exists():self.draft_path.unlink()
        self.message(f'已保存“{self.loaded_name}”，共{len(record["keyframes"])}个姿态；可在任务流程中选择。')
    def manage_library(self):
        if not self.edit_allowed():return
        from PyQt5.QtGui import QCursor
        menu=Q.QMenu(self);remove=menu.addAction('删除当前已保存的动作串…');remove.setEnabled(bool(self.loaded_name))
        if menu.exec_(QCursor.pos())!=remove:return
        name=self.loaded_name
        if Q.QMessageBox.question(self,'删除动作串',f'删除“{name}”并清空当前编辑？引用这个名称的任务需重新选择动作。',Q.QMessageBox.Yes|Q.QMessageBox.No,Q.QMessageBox.No)!=Q.QMessageBox.Yes:return
        def work():
            self.studio.library.delete(name,self.revision);self.dirty=False;self.new();self.message('动作串已删除。')
        self.guard(work)
    def row(self):return self.frames.currentRow()
    def selected(self):
        i=self.row();return self.draft['frames'][i] if 0<=i<len(self.draft['frames']) else None
    def refresh_frames(self,index=None):
        if index is None:index=self.row()
        self.frames.blockSignals(True);self.frames.clear()
        previous=None
        for i,f in enumerate(self.draft['frames']):
            recorded=self.draft.get('kind')=='trajectory'
            jaws='' if recorded else ' · 到位后夹爪' if previous is None or any(abs(f['joints_deg'][n]-previous[n])>.001 for n in GRIPPER_JOINTS) else ''
            dwell=(f'额外停顿 {f["hold_sec"]:g}s' if f['hold_sec'] else '按录制时序') if recorded else f'停顿 {f["hold_sec"]:g}s' if f['hold_sec'] else ('无额外停顿' if jaws else '连续通过')
            self.frames.addItem(f'{i+1:02d}  {f["label"]}\n{f["speed_percent"]:g}% · {dwell}{jaws}');previous=f['joints_deg']
        self.frames.setCurrentRow(min(max(0,index),self.frames.count()-1));self.frames.blockSignals(False);self.select_frame(self.row())
    def select_frame(self,index):
        f=self.selected();self.changing=True;self.properties.setEnabled(f is not None)
        recorded=self.draft.get('kind')=='trajectory'
        self.minimum.setEnabled(not recorded)
        self.minimum.setVisible(not recorded);self.properties.layout().labelForField(self.minimum).setVisible(not recorded)
        self.minimum.setToolTip('连续轨迹使用原始时间，不自动增加每个标记的最短时间。' if recorded else '根据关节跨度计算最短到达时间。')
        self.sequence_note.setText('连续轨迹：按录制时刻同步身体与夹爪。\n100%速度、无额外停顿：保持原录制时长。' if recorded else '顺序：关节到位 → 夹爪开合 → 停顿\n普通姿态停顿为0：平滑通过，不逐点停留')
        self.timing_label.setText(f'原录制 {self.draft["recorded_sec"]:.2f} 秒；预演从录制起点开始。真机起点准备与结束确认另计。' if recorded else '逐点动作按平滑运动及到位后夹爪顺序执行。')
        self.properties.layout().labelForField(self.hold).setText('额外停顿' if recorded else '完成后停顿')
        self.hold.setToolTip('保留录制节奏，在这个节点额外停顿；0秒不额外停车。' if recorded else '0秒连续通过；夹爪交互仍需关节先到位。')
        if f:
            self.frame_name.setText(f['label']);self.hold.setValue(f['hold_sec']);self.speed.setValue(f['speed_percent']);self.minimum.setValue(f['minimum_sec']);self.ghost.setChecked(f['ghost'])
            if self.model:self.model.target_toggle.setChecked(f['ghost'])
        self.select_joint();self.changing=False
    def change_properties(self,*args):
        if self.remote_locked():return
        f=self.selected()
        if self.changing or not f:return
        values=dict(label=self.frame_name.text().strip() or f'姿态 {self.row()+1}',hold_sec=self.hold.value(),speed_percent=self.speed.value(),minimum_sec=self.minimum.value(),ghost=self.ghost.isChecked())
        if all(f[k]==v for k,v in values.items()):return
        self.snapshot();f.update(values);self.changed();self.refresh_frames()
    def capture(self,replace=False):
        if self.remote_locked():
            if not replace:self.guard(lambda:self.studio.recorder.command('capture'))
            return
        if not replace and self.draft.get('kind')=='trajectory':
            self.message('连续轨迹保留完整采样；请微调已有节点，或新录制一条轨迹。');return
        def work():
            if self.studio.engine.moving():raise ValueError('动作执行中不能记录')
            values=self.studio.feedback.capture()
            if replace and not self.selected():raise ValueError('先选择需要替换的姿态')
            if not replace and len(self.draft['frames'])>=100:raise ValueError('最多100个姿态，请另建动作串')
            self.snapshot()
            if replace:self.selected()['joints_deg']=values;index=self.row()
            else:
                index=self.row()+1 if self.selected() else len(self.draft['frames'])
                self.draft['frames'].insert(index,dict(label=f'姿态 {len(self.draft["frames"])+1}',joints_deg=values,hold_sec=.3,speed_percent=100.,minimum_sec=.6,ghost=True))
            self.changed();self.refresh_frames(index);self.message('已记录实测全身姿态，包括双臂、腰腿、头部和双夹爪。')
        self.guard(work)
    def move(self,delta):
        if not self.edit_allowed():return
        if self.draft.get('kind')=='trajectory':self.message('连续轨迹按时间顺序播放，不能移动节点顺序。');return
        i=self.row();j=i+delta
        if not (0<=i<len(self.draft['frames']) and 0<=j<len(self.draft['frames'])):return
        self.snapshot();self.draft['frames'][i],self.draft['frames'][j]=self.draft['frames'][j],self.draft['frames'][i];self.changed();self.refresh_frames(j)
    def duplicate(self):
        if not self.edit_allowed():return
        if self.draft.get('kind')=='trajectory':self.message('连续轨迹节点对应真实时刻，不能复制；可增加停顿。');return
        if not self.selected() or len(self.draft['frames'])>=100:return
        self.snapshot();i=self.row()+1;f=deepcopy(self.selected());f['label']+=' 副本';self.draft['frames'].insert(i,f);self.changed();self.refresh_frames(i)
    def remove(self):
        if not self.edit_allowed():return
        if self.draft.get('kind')=='trajectory' and self.row() in (0,len(self.draft['frames'])-1):
            self.message('连续轨迹必须保留起点与终点。');return
        if self.selected():self.snapshot();self.draft['frames'].pop(self.row());self.changed();self.refresh_frames()
    def undo(self):
        if not self.edit_allowed():return
        if self.undo_stack:self.redo_stack.append(deepcopy(self.draft));self.draft=self.undo_stack.pop();self.name.setText(self.draft['name']);self.end_preview();self.changed();self.refresh_frames();self.last_joint_edit=(None,0.)
    def redo(self):
        if not self.edit_allowed():return
        if self.redo_stack:self.undo_stack.append(deepcopy(self.draft));self.draft=self.redo_stack.pop();self.name.setText(self.draft['name']);self.end_preview();self.changed();self.refresh_frames()
    def select_joint(self,*args):
        n=self.joint.currentData();f=self.selected();self.angle.blockSignals(True);self.angle.setRange(*JOINT_LIMITS[n]);self.angle.setEnabled(f is not None)
        if f:self.angle.setValue(f['joints_deg'][n])
        self.angle.blockSignals(False)
    def model_selected(self,name):
        i=self.joint.findData(name)
        if i>=0:self.joint.setCurrentIndex(i)
    def numeric_joint(self):self.model_joint(self.joint.currentData(),self.angle.value())
    def model_joint(self,name,value):
        if self.remote_locked():return
        f=self.selected()
        if not f or self.playing or self.studio.engine.moving():return
        lo,hi=JOINT_LIMITS[name];value=max(lo,min(hi,float(value)))
        if abs(f['joints_deg'][name]-value)<1e-5:return
        now=time.monotonic()
        if self.last_joint_edit[0]!=name or now-self.last_joint_edit[1]>.5:self.snapshot()
        self.last_joint_edit=(name,now);f['joints_deg'][name]=value;self.changed();self.model_selected(name);self.select_joint()
    def model_ghost(self,value):
        if self.changing or self.preview:return
        self.ghost.setChecked(value)
    def prepare_preview(self):
        if not self.draft['frames']:raise ValueError('请先记录姿态')
        entry=dict(self.studio.feedback.values) if self.studio.feedback.fresh() else dict(self.draft['frames'][0]['joints_deg'])
        if self.draft.get('kind')=='trajectory':entry=dict(self.draft['frames'][0]['joints_deg'])
        from .teaching_trajectory import plan
        flat=compile_action(self.draft,entry);record,segments,phases=plan(flat,entry,self.studio.library.directory)
        self.preview=(record,segments,phases);self.elapsed=0.;self.last_tick=time.monotonic()
    def toggle_preview(self):
        def work():
            if not self.preview:self.prepare_preview()
            self.playing=not self.playing
            if self.elapsed>=total_time(self.preview[0]):self.elapsed=0.
            self.play.setText('暂停预演' if self.playing else '继续预演')
            if self.model:self.model.set_action_preview_active(True)
        self.guard(work)
    def end_preview(self):
        self.playing=False;self.preview=None;self.play.setText('▶ 模型预演')
        if self.model:
            self.model.set_action_preview_active(False)
            if self.selected():self.model.target_toggle.setChecked(self.selected()['ghost'])
    def seek(self,value):
        def work():
            if not self.preview:self.prepare_preview()
            self.elapsed=total_time(self.preview[0])*value/1000.;self.playing=False;self.play.setText('继续预演')
            if self.model:self.model.set_action_preview_active(True)
        self.guard(work)
    def execute_current(self):
        if not self.edit_allowed():return
        def work():
            self.studio.feedback.capture()
            if self.studio.runtime.busy:raise ValueError('先停止并关闭遥操，才能执行动作串')
            record=compile_action(self.draft,self.studio.feedback.values)
            timing=''
            if 'trajectory' in record:
                from .teaching_replay import edited_samples
                times,_,original=edited_samples(record,self.studio.library.directory)
                timing=f'\n原录制 {original:.2f} 秒，本次动作 {times[-1]:.2f} 秒；先对齐录制起点，准备时间另计。身体与夹爪按录制时序运行。'
            if Q.QMessageBox.question(self,'执行真实机器人动作',f'将从当前姿态执行“{self.draft["name"] or "未命名动作串"}”，包含腰腿、头部和夹爪。{timing}\n请确认机器人及物体相对位置与示教时一致、运动路径有空间。模型预演不是碰撞检查。\n现在执行？',Q.QMessageBox.Yes|Q.QMessageBox.No,Q.QMessageBox.No)!=Q.QMessageBox.Yes:return
            self.end_preview();self.studio.execute('teaching-'+str(time.time_ns()),record);self.message('动作串已提交，等待控制器准备。')
        self.guard(work)
    def copy_address(self):
        # Robot addresses are read from the same configuration, not a browser launch.
        address=self.studio.cfg.get('teaching_vr_url','https://192.168.8.176:8445/')
        Q.QApplication.clipboard().setText(address);self.message('已复制：'+address)
    def update_runtime(self):
        runtime=self.studio.runtime;self.runtime_status.setText(runtime.message);data=runtime.data or {};state=data.get('teleop',{})
        vr='已连接' if data.get('vrConnected') else '等待连接'
        enabled='模拟模式' if state.get('dry_run') else '复位/使能中' if state.get('hardware_enable_pending') or state.get('quick_reset',{}).get('active') else '已使能' if state.get('hardware_enabled') else '未使能'
        if data and not data.get('teleopStatusFresh',False):enabled='状态待更新'
        arms=state.get('active_arms',[]);active='、'.join({'left':'左臂','right':'右臂'}.get(a,str(a)) for a in arms) or '未接管'
        self.connection.setText(f'VR：{vr}　遥操：{enabled}　接管：{active}　|　头显网页连接后按原有方式操作。')
        self.start.setEnabled(not runtime.busy and not self.studio.engine.moving());self.close_vr.setEnabled(runtime.owned or runtime.start_pending)
    def show_log(self):
        path=self.studio.owner.run/'teaching_vr.log';dialog=Q.QDialog(self);dialog.setWindowTitle('遥操后台日志');dialog.resize(850,540)
        box=Q.QVBoxLayout(dialog);text=Q.QPlainTextEdit();text.setReadOnly(True);text.setPlainText(path.read_text('utf8',errors='replace')[-35000:] if path.exists() else '暂无后台日志');box.addWidget(text)
        button=Q.QPushButton('关闭');button.clicked.connect(dialog.accept);box.addWidget(button);dialog.exec_()
    def tick(self):
        now=time.monotonic();dt=max(0.,now-self.last_tick);self.last_tick=now
        feedback=self.studio.feedback;busy=self.studio.engine.moving() or self.remote_locked();f=self.selected()
        self.capture_button.setEnabled(feedback.fresh() and not busy);self.replace_button.setEnabled(feedback.fresh() and f is not None and not busy)
        if self.studio.recorder.data.get('enabled'):
            self.capture_button.setEnabled(feedback.fresh() and (self.studio.recorder.data.get('mode')=='poses' or self.studio.recorder.data.get('phase')=='recording'))
        self.capture_button.setText('＋ 标记当前时刻' if self.studio.recorder.data.get('enabled') and self.studio.recorder.data.get('mode')=='trajectory' else '＋ 记录当前姿态')
        self.name.setEnabled(not busy);self.actions.setEnabled(not busy);self.play.setEnabled(not busy);self.progress.setEnabled(not busy)
        for button in self.findChildren(Q.QPushButton):
            if button.text() in ('新建','保存','管理','上移','下移','复制','删除','撤销','重做'):
                button.setEnabled(not busy and not (self.draft.get('kind')=='trajectory' and button.text() in ('上移','下移','复制')))
        self.execute_button.setEnabled(bool(self.draft['frames']) and not self.studio.runtime.busy and not busy and feedback.fresh())
        self.stop_execution.setEnabled(self.studio.engine.moving());self.frames.setEnabled(not self.studio.engine.moving());self.properties.setEnabled(f is not None and not busy);self.angle.setEnabled(f is not None and not busy)
        target=f['joints_deg'] if f else feedback.values
        if self.preview:
            record,segments,phases=self.preview
            if self.playing:self.elapsed+=dt
            target,index,done=sample(record,segments,self.elapsed);pose,phase,_=phases[index]
            self.phase_label.setText(f'模型预演 · 姿态 {pose+1} · {phase}');total=total_time(record)
            self.time_label.setText(f'{min(total,self.elapsed):.2f} / {total:.2f} 秒')
            if not self.progress.isSliderDown():self.progress.setValue(round(1000*min(1.,self.elapsed/total)))
            if done:self.playing=False;self.play.setText('重新预演')
        elif self.remote_locked():
            self.phase_label.setText('实测模型持续更新；示教中自动保存，完成并编辑后可微调与预演。')
        elif busy:
            b=self.studio.engine.bridge;state=b._action_state if b else None
            if state:
                phases=getattr(self.studio.engine,'teaching_phases',[]);i=state['frame_index']
                if i<len(phases):self.phase_label.setText(f'真机执行 · 姿态 {phases[i][0]+1} · {phases[i][1]} · {state["stage_label"]}')
                if 'recorded_elapsed' in state:
                    self.time_label.setText(f'{state["recorded_elapsed"]:.2f} / {state["recorded_duration"]:.2f} 秒')
                    self.progress.setValue(round(1000*state['recorded_elapsed']/state['recorded_duration']))
                elif self.draft.get('kind')=='trajectory':self.time_label.setText('起点准备（另计）')
                target=dict(b._preview)
            elif self.studio.engine.job and self.studio.engine.job.get('phase')=='settle':self.phase_label.setText('录制时间轴已完成，正在确认停稳（另计）')
        elif f:self.phase_label.setText(f'选中：{f["label"]}　|　蓝色虚影可编辑，真实模型只读。')
        if self.model:
            self.model.joint_mode_button.setEnabled(not busy)
            if busy and self.model.joint_mode_button.isChecked():self.model.camera_mode_button.click()
            connected=feedback.fresh()
            if connected!=self.last_connected:self.model.set_connected(connected);self.last_connected=connected
            if feedback.values and target:self.model.set_joint_states(feedback.values,target)
        if self.autosave_due and now>=self.autosave_due:
            self.autosave_due=0.;self.guard(self.persist_draft)
