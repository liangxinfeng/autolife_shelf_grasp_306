"""Compact desktop controls shared by the two capture modes."""
from PyQt5 import QtWidgets as Q


class TeachingControls:
    def setup_recording_controls(self,layout):
        self.remote_revision=-1;self.remote_session='';self.remote_arming=False;self.remote_service=''
        row=Q.QHBoxLayout();row.addWidget(Q.QLabel('示教模式'))
        self.record_mode=Q.QComboBox();self.record_mode.addItem('逐点记录','poses');self.record_mode.addItem('连续轨迹','trajectory')
        self.record_mode.activated.connect(self.remote_mode_changed);row.addWidget(self.record_mode)
        self.arm_recording=self.button('开始手柄示教',self.toggle_remote);row.addWidget(self.arm_recording)
        self.record_action=self.button('＋ 记录姿态',self.remote_action);row.addWidget(self.record_action)
        self.next_recording=self.button('下一条动作',lambda:self.guard(lambda:self.studio.recorder.command('new')));row.addWidget(self.next_recording)
        self.recover_recording=self.button('恢复未完成…',self.recover_remote);row.addWidget(self.recover_recording);row.addStretch()
        layout.addLayout(row)
        self.record_hint=Q.QLabel('逐点：双击右摇杆记录。连续：长按右摇杆1秒开始/结束。单击菜单与底盘微调均保留。')
        self.record_hint.setWordWrap(True);self.record_hint.setStyleSheet('color:#93c9da;font-size:12px');layout.addWidget(self.record_hint)

    def remote_locked(self):
        d=self.studio.recorder.data
        return bool(self.remote_arming or d.get('enabled') or d.get('phase') in ('recording','processing','countdown'))

    def edit_allowed(self):
        if self.remote_locked():
            self.message('手柄示教中自动保存；点击“完成并编辑”后再修改动作。');return False
        return True

    def begin_remote(self):
        if self.remote_locked():return True
        if not self.discard_ok():return False
        self.end_preview();self.remote_arming=True
        self.studio.recorder.command('enable',mode=self.record_mode.currentData())
        return True

    def toggle_remote(self):
        def work():
            if self.studio.recorder.data.get('enabled'):
                self.studio.recorder.command('finish')
            else:self.begin_remote()
        self.guard(work)

    def remote_mode_changed(self,index):
        if self.studio.recorder.data.get('enabled'):
            self.guard(lambda:self.studio.recorder.command('mode',mode=self.record_mode.currentData()))

    def remote_action(self):
        def work():
            d=self.studio.recorder.data
            if not d.get('enabled'):
                if self.begin_remote():self.message('手柄示教已准备，状态就绪后即可记录。')
                return
            self.studio.recorder.command('toggle_record' if d.get('mode')=='trajectory' else 'capture')
        self.guard(work)

    def recorder_failed(self,message):
        self.remote_arming=False;self.message(message,True)

    def update_recorder(self,d):
        enabled=d.get('enabled',False);phase=d.get('phase','idle');busy=phase in ('recording','countdown','processing')
        self.arm_recording.setText('完成并编辑' if enabled else '开始手柄示教')
        self.arm_recording.setEnabled(phase!='processing' and not self.studio.engine.moving())
        self.next_recording.setEnabled(enabled and not busy)
        self.record_mode.setEnabled(not busy and not self.studio.engine.moving())
        self.record_action.setEnabled(d.get('available',False) and enabled and phase!='processing')
        self.recover_recording.setVisible(bool(d.get('recoverable')))
        self.recover_recording.setEnabled(not busy)
        if d.get('available'):self.remote_arming=False
        if enabled:
            self.record_mode.blockSignals(True);self.record_mode.setCurrentIndex(1 if d.get('mode')=='trajectory' else 0);self.record_mode.blockSignals(False)
        trajectory=d.get('mode')=='trajectory'
        self.record_action.setText('正在整理…' if phase=='processing' else '■ 结束并保存' if phase=='recording'
            else '取消倒计时' if phase=='countdown' else '● 开始录制' if trajectory else '＋ 记录姿态')
        summary=(f'● 正在录制 {d.get("elapsed",0):.1f} 秒 · {d.get("samples",0)} 个实测采样'+(' · A+X复位自动接入' if d.get('resetting') else '')
            if phase=='recording' else f'准备开始：{d.get("countdown",0):.1f} 秒' if phase=='countdown' else d.get('message',''))
        self.record_hint.setText(summary+'\n双击右摇杆记录/标记；长按1秒开始/结束轨迹。底盘微调保留，底盘位移不录入本动作。')
        revision=d.get('revision',-1);service=d.get('service_id','')
        if revision==self.remote_revision and service==self.remote_service:return
        if service==self.remote_service and revision<self.remote_revision:return
        self.remote_revision=revision;self.remote_service=service
        if enabled and d.get('session')!=self.remote_session:
            self.remote_session=d.get('session','');self.undo_stack=[];self.redo_stack=[];self.end_preview()
        name=d.get('saved_name','')
        if name:
            def load():
                # This path is only used while the recorder owns editing, or for
                # its final save. Later library polling never discards user edits.
                self.draft=self.studio.library.load(name);self.loaded_name=name;self.dirty=False
                self.name.setText(name);self.refresh_library();self.refresh_frames(len(self.draft['frames'])-1)
                self.message(d.get('message','已保存'))
            if enabled or phase in ('saved','processing') or not self.dirty:self.guard(load)
        elif enabled and d.get('count',0)==0:
            self.draft=dict(name=d.get('name',''),frames=[]);self.loaded_name='';self.dirty=False
            self.name.setText(self.draft['name']);self.refresh_library();self.refresh_frames()

    def recover_remote(self):
        files=self.studio.recorder.data.get('recoverable',[])
        if not files:return
        name,ok=Q.QInputDialog.getItem(self,'未完成录制','选择需要恢复的录制（保留已录部分，恢复后先预演）',files,0,False)
        if not ok:return
        box=Q.QMessageBox(self);box.setWindowTitle('处理未完成录制');box.setText('将已记录的完整部分整理为新的动作串，不会驱动机器人。')
        restore=box.addButton('恢复为新动作',Q.QMessageBox.AcceptRole)
        delete=box.addButton('删除这份未完成录制',Q.QMessageBox.DestructiveRole);box.addButton(Q.QMessageBox.Cancel);box.exec_()
        if box.clickedButton()==restore:
            if self.discard_ok():self.guard(lambda:self.studio.recorder.command('recover',file=name))
        elif box.clickedButton()==delete:self.guard(lambda:self.studio.recorder.command('delete_partial',file=name))
