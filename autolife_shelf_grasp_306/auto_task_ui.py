"""Automatic runs reuse Workflow; only completed runs may enter another cycle."""
import json
import math
import time
from copy import deepcopy
from PyQt5 import QtCore as C, QtGui as G, QtWidgets as Q
from .core import atomic_json
from .task_flow import validate, KINDS


from .task_display import TaskDisplay


class AutomaticTaskPage(Q.QWidget):
    def __init__(self, owner):
        super().__init__(owner); self.owner = owner
        self.paused=False;self.deferred={};self.deferred_event=None;self.round_prepared=False;self.paused_at=0.
        self.active = False; self.mode = 'single'; self.phase = 'idle'
        self.record = None; self.task_name = ''; self.actions = None; self.map = None; self.run_id = None
        self.completed = 0; self.limit = 1; self.interval_s = 10.; self.deadline = 0.
        self.settle_since = None; self.launching = False; self.last_error = ''; self.display = None
        self.path = owner.root/'data/auto_task_settings.json'
        layout = Q.QVBoxLayout(self); layout.setContentsMargins(24, 20, 24, 20)
        title = Q.QLabel('自动任务'); title.setStyleSheet('font-size:26px;font-weight:700;'); layout.addWidget(title)
        row = Q.QHBoxLayout(); self.names = Q.QComboBox(); self.names.setMinimumWidth(240)
        row.addWidget(Q.QLabel('已保存流程')); row.addWidget(self.names, 1)
        self.refresh_button = Q.QPushButton('刷新任务库'); self.refresh_button.clicked.connect(self.refresh_library)
        row.addWidget(self.refresh_button); layout.addLayout(row)
        row = Q.QHBoxLayout()
        self.interval = Q.QSpinBox(); self.interval.setRange(1, 86400); self.interval.setValue(10); self.interval.setSuffix(' 秒')
        self.cycles = Q.QSpinBox(); self.cycles.setRange(0, 9999); self.cycles.setValue(0); self.cycles.setSpecialValueText('持续循环')
        row.addWidget(Q.QLabel('每轮完成后间隔')); row.addWidget(self.interval)
        row.addWidget(Q.QLabel('循环次数')); row.addWidget(self.cycles); row.addStretch(); layout.addLayout(row)
        self.display_direction=Q.QComboBox()
        for label,key in [('机器人竖屏（默认）','robot'),('跟随桌面','desktop'),('机器人竖屏（反向）','reverse')]:self.display_direction.addItem(label,key)
        layout.addWidget(self.display_direction)
        self.display_direction.currentIndexChanged.connect(lambda: self.display.relayout() if self.display else None)
        self.steps = Q.QListWidget(); layout.addWidget(self.steps, 1)
        self.state = Q.QLabel('选择流程后启动；每轮重新识别、规划与执行。'); self.state.setWordWrap(True)
        self.state.setStyleSheet('font-size:19px;padding:12px;background:#e5f3f4;color:#123e50;'); layout.addWidget(self.state)
        row = Q.QHBoxLayout()
        self.single = Q.QPushButton('启动单次'); self.single.clicked.connect(lambda: self.start('single'))
        self.loop = Q.QPushButton('开始自动循环任务'); self.loop.clicked.connect(lambda: self.start('loop'))
        self.show_button = Q.QPushButton('显示 Logo / 状态'); self.show_button.clicked.connect(self.show_display)
        for button in (self.single, self.loop, self.show_button): row.addWidget(button)
        layout.addLayout(row)
        self.pause_button=Q.QPushButton('暂停任务');self.pause_button.clicked.connect(self.toggle_pause);layout.addWidget(self.pause_button)
        self.stop_button = Q.QPushButton('紧急停止任务'); self.stop_button.setObjectName('stop')
        self.stop_button.setMinimumHeight(54); self.stop_button.clicked.connect(self.stop); layout.addWidget(self.stop_button)
        note = Q.QLabel('新任务从空载开始：归档旧持物记录，检查并自动复位准备。异常停止循环。\n关闭 Logo 窗口会停止任务；Esc 仅退出全屏。危险时使用实体急停。')
        note.setWordWrap(True); layout.addWidget(note)
        self.names.currentTextChanged.connect(self.preview)
        self.library = {}; self.refresh_library()
        try:
            settings = json.loads(self.path.read_text('utf-8'))
            self.names.setCurrentText(settings.get('name', ''))
            self.interval.setValue(int(settings.get('interval_s', 10)))
            self.cycles.setValue(int(settings.get('cycles', 0)))
            self.display_direction.setCurrentIndex(max(0,self.display_direction.findData(settings.get('display_direction','robot'))))
        except (OSError, ValueError, TypeError): pass
        owner.workflow.started.connect(self.workflow_started)
        owner.workflow.completed.connect(self.workflow_completed)
        self.timer = C.QTimer(self); self.timer.setInterval(250); self.timer.timeout.connect(self.tick); self.timer.start()

    def refresh_library(self):
        if self.active: return
        name = self.names.currentText()
        try:
            path = self.owner.root/'data/task_flows.json'
            library = json.loads(path.read_text('utf-8')) if path.exists() else {}
            if not isinstance(library, dict): raise ValueError('任务库格式错误')
            self.library = library; self.names.clear(); self.names.addItems(sorted(library))
            if name in library: self.names.setCurrentText(name)
            self.preview()
        except (OSError, ValueError) as exc:
            self.library = {}; self.names.clear(); self.state.setText('任务库读取失败：'+str(exc))

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_library()

    def preview(self):
        self.steps.clear()
        try:
            record = validate(self.library[self.names.currentText()])
            for i, step in enumerate(record['steps']): self.steps.addItem(f'{i+1:02d}  {KINDS[step["kind"]]}  ·  {step["target"]}')
        except (KeyError, ValueError, TypeError): pass

    def available(self):
        o = self.owner
        return (o.ready and not o.busy and o.motion is None and not o.cancel_pending
                and not o.demo_active and not o.placement_active and not o.workflow.active
                and not o.workflow.base.moving() and not o.studio.moving() and not o.navigation.inflight
                and o.chassis_dialog is None)

    def start(self, mode, *, confirmed=False, repeat=False):
        if self.active: return
        try:
            if not self.available(): raise ValueError('请等待当前操作停止、后台就绪后再启动')
            record = deepcopy(self.record) if repeat else validate(self.library[self.names.currentText()])
            name = self.task_name if repeat else self.names.currentText()
            if mode not in ('single', 'loop'): raise ValueError('运行模式无效')
            if not confirmed:
                text = ('单次执行' if mode == 'single' else '自动循环执行')+'流程“'+name+'”。\n将清理旧持物记录并自动复位张开夹爪。请确认双爪空载、场地适合运行，人员离开运动范围。'
                if Q.QMessageBox.question(self, '开始自动任务', text, defaultButton=Q.QMessageBox.No) != Q.QMessageBox.Yes: return
            if not self.available(): raise ValueError('确认期间状态已改变，请等待操作停止后再启动')
            self.owner.invalidate_plan(); self.owner.studio.release()
            self.record = deepcopy(record); self.task_name = name; self.mode = mode; self.completed = 0
            self.limit = 1 if mode == 'single' else self.cycles.value(); self.interval_s = self.interval.value()
            # Freeze fixed actions along with the task definition for this session.
            if not repeat:
                self.actions = deepcopy(self.owner.studio.records())
                self.map = self.owner.navigation.data.get('active_map')
            atomic_json(self.path, dict(name=name, interval_s=self.interval.value(), cycles=self.cycles.value(),display_direction=self.display_direction.currentData()))
            self.paused=False;self.round_prepared=False;self.deferred.clear();self.deferred_event=None
            (self.owner.run/'task_pause').unlink(missing_ok=True)
            self.active = True; self.phase = 'waiting'; self.last_error = ''; self.run_id = None
            self.deadline = time.monotonic(); self.settle_since = None
            self.show_display(); self.tick()
        except Exception as exc:
            if self.active: self.owner.stop_motion(reason=str(exc), source='automatic_task')
            self.phase = 'stopped'; self.last_error = '未启动：'+str(exc); self.update_view()

    def show_display(self):
        if self.display is None: self.display = TaskDisplay(self)
        self.display.fullscreen.setText('缩小窗口'); self.display.showFullScreen(); self.update_view()

    def repeat_single(self):
        if self.active or self.mode != 'single' or self.phase != 'completed' or not self.available(): return
        # The explicit repeat button authorizes exactly one new run of the snapshot.
        self.start('single', confirmed=True, repeat=True)

    def workflow_started(self, run_id):
        if self.active and self.launching: self.run_id = run_id

    def workflow_completed(self, run_id):
        if not self.active or self.phase != 'running' or run_id != self.run_id: return
        self.completed += 1; self.run_id = None
        if self.limit and self.completed >= self.limit:
            self.active = False; self.phase = 'completed'
        else:
            self.phase = 'waiting'; self.round_prepared=False; self.deadline = time.monotonic()+self.interval_s; self.settle_since = None
        self.owner.controls(); self.update_view()

    def abort(self, reason):
        if not self.active: return
        (self.owner.run/'stop').touch()
        (self.owner.run/'task_pause').unlink(missing_ok=True)
        self.paused=False;self.deferred.clear();self.deferred_event=None
        self.active = False; self.phase = 'stopped'; self.run_id = None; self.last_error = str(reason)
        self.update_view()

    def stop(self):
        self.owner.stop_motion(reason='停止自动任务', source='automatic_task')

    def tick(self):
        if self.active and not self.paused and self.phase == 'waiting' and time.monotonic() >= self.deadline:
            if self.available():
                try:
                    if not self.round_prepared:
                        from .task_pause import archive_held
                        archive_held(self.owner.run)
                        (self.owner.run/'stop').unlink(missing_ok=True)
                        self.phase='preparing';self.owner.send('auto_prepare');self.update_view();return
                    self.phase = 'running'; self.launching = True
                    self.owner.workflow.start(deepcopy(self.record), confirmed=True,
                        actions_snapshot=self.actions, expected_map=self.map)
                except Exception as exc:
                    self.owner.stop_motion(reason='自动任务未开始：'+str(exc), source='automatic_task')
                finally: self.launching = False
            else:
                if self.settle_since is None: self.settle_since = time.monotonic()
                if time.monotonic()-self.settle_since > 30:
                    self.owner.stop_motion(reason='上一轮执行通道未释放或后台未就绪', source='automatic_task')
        self.update_view()

    def update_view(self):
        for widget in (self.names, self.refresh_button, self.interval, self.cycles): widget.setEnabled(not self.active)
        ready = self.available()
        self.single.setEnabled(not self.active and ready and bool(self.names.count()))
        self.loop.setEnabled(self.single.isEnabled())
        self.pause_button.setEnabled(self.active)
        self.pause_button.setText('继续任务' if self.paused else '暂停任务')
        if self.phase == 'preparing':text='检查复位状态 · 未准备则自动复位并张开夹爪'
        elif self.phase == 'running':
            flow = self.owner.workflow; cursor = flow.cursor
            step = (f'步骤 {cursor.index+1}/{len(cursor.record["steps"])} · {KINDS[cursor.step["kind"]]} · {cursor.step["target"]}'
                    if cursor and cursor.active else '处理中')
            phase = {'prepare':'复位准备','prepare_navigation':'出发准备','navigate':'导航行驶',
                     'scan':'识别物品','plan':'规划抓取','confirm':'准备执行抓取','grasp':'抓取执行',
                     'place':'识别桌面 / 规划并放置','action':'执行固定动作','posture':'身体与头部调姿',
                     'base':'执行底盘定量动作'}.get(flow.phase,'处理中')
            text = f'第 {self.completed+1} 轮 · {step}\n{phase} · '+self.owner.flow_label.text()
        elif self.phase == 'waiting':
            remaining = max(0, math.ceil(self.deadline-(self.paused_at if self.paused else time.monotonic())))
            text = f'已完成 {self.completed} 轮 · 距下一轮 {remaining} 秒' if remaining else '正在准备下一轮…'
        elif self.phase == 'completed': text = f'已完成 {self.completed} 轮 · 任务执行完毕'
        elif self.phase == 'stopped': text = '自动任务已停止 · '+self.last_error
        else: text = '选择已保存的任务流程，启动单次或自动循环。'
        if self.paused:text='已请求暂停 · 保持当前步骤\n'+text
        self.state.setText(text)
        if self.display:
            self.display.title.setText(self.task_name or '自动任务')
            self.display.state.setText(text)
            self.display.pause.setEnabled(self.active);self.display.pause.setText('继续任务' if self.paused else '暂停任务')
            self.display.again.setVisible(self.mode == 'single')
            self.display.again.setEnabled(self.phase == 'completed' and not self.active and ready)

    def toggle_pause(self):
        if not self.active:return
        if not self.paused:
            self.paused=True;self.paused_at=time.monotonic()
            (self.owner.run/'task_pause').touch()
        else:
            elapsed=time.monotonic()-self.paused_at;self.paused=False
            if self.phase=='waiting':self.deadline+=elapsed
            if self.settle_since is not None:self.settle_since+=elapsed
            (self.owner.run/'task_pause').unlink(missing_ok=True)
            event=self.deferred_event;self.deferred_event=None
            queued=list(self.deferred.values());self.deferred.clear()
            for method,args,token in queued:
                cursor=self.owner.workflow.cursor
                if self.active and (token is None or (cursor and cursor.active and cursor.token==token)):method(*args)
            if event and self.active:
                data,request,token=event
                cursor=self.owner.workflow.cursor
                if token is None or (cursor and cursor.active and cursor.token==token):
                    if data['event']=='auto_prepared':
                        # This is a completion receipt, not a plan. Never reset
                        # or open the grippers again just because it was paused.
                        self.owner.receive_event(data)
                    else:
                        action,values=request
                        self.owner.send(action,**values)
        self.update_view()

    def defer_worker_event(self,data):
        if self.active and self.paused and data.get('event') in ('scene','reset_planned','planned','observe_planned','place_planned','auto_prepared'):
            cursor=self.owner.workflow.cursor
            token=cursor.token if cursor and cursor.active else None
            self.deferred_event=(deepcopy(data),deepcopy(self.owner.last_worker_request),token)
            self.owner.busy=False;return True
        return False

    def prepared(self,result):
        if not self.active or self.phase!='preparing':return False
        if result.get('status')!='prepared':
            self.owner.stop_motion(reason=result.get('reason','自动复位准备未成功'),source='automatic_task')
        else:
            self.round_prepared=True;self.phase='waiting';self.deadline=time.monotonic()
            if self.paused:self.paused_at=self.deadline
            self.owner.controls();self.tick()
        return True

    def shutdown(self):
        self.timer.stop(); self.abort('关闭抓取窗口')
        if self.display: self.display.close()
