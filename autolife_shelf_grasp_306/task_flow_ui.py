"""Workflow editor and Qt adapter over existing navigation/grasp/place APIs."""
import json
import time
from copy import deepcopy
from PyQt5 import QtCore as C, QtWidgets as Q
from .task_flow import Cursor, KINDS, validate, select_candidates
from .core import atomic_json
from .task_pause import defer_call


class FlowDialog(Q.QDialog):
    def __init__(self, owner):
        super().__init__(owner); self.owner = owner; self.resize(1100, 580)
        self.setWindowTitle('任务流程 · 导航 / 抓取 / 放置 / 固定动作 / 底盘定量')
        self.path = owner.root / 'data/task_flows.json'
        self.library = json.loads(self.path.read_text('utf-8')) if self.path.exists() else {}
        layout = Q.QVBoxLayout(self); row = Q.QHBoxLayout(); layout.addLayout(row)
        self.names = Q.QComboBox(); self.names.addItem('新建流程'); self.names.addItems(sorted(self.library)); row.addWidget(self.names)
        self.name = Q.QLineEdit(); self.name.setPlaceholderText('流程名称'); row.addWidget(self.name, 1)
        nav_row=Q.QHBoxLayout();layout.addLayout(nav_row)
        self.nav_note=Q.QLabel();self.nav_note.setWordWrap(True);nav_row.addWidget(self.nav_note,1)
        refresh=Q.QPushButton('刷新航点');refresh.clicked.connect(self.refresh_waypoints);nav_row.addWidget(refresh)
        self.navigation_button=Q.QPushButton('导航地图 / 定位');self.navigation_button.clicked.connect(self.navigation_map);nav_row.addWidget(self.navigation_button)
        self.table = Q.QTableWidget(0, 5); self.table.setHorizontalHeaderLabels(['步骤类型', '航点 / 物品 / 固定动作名称','下蹲方向','下降高度(cm)','头部俯仰(°)'])
        for col,width in ((0,85),(2,120),(3,135),(4,185)):self.table.setColumnWidth(col,width)
        self.table.horizontalHeader().setSectionResizeMode(1, Q.QHeaderView.Stretch)
        layout.addWidget(self.table, 1)
        row = Q.QHBoxLayout(); layout.addLayout(row)
        for kind, label in KINDS.items():
            button = Q.QPushButton('＋' + label); row.addWidget(button)
            button.clicked.connect(lambda _, k=kind: self.add(k, ''))
        for label, fn in [('删除步骤', self.remove), ('上移', lambda: self.move(-1)), ('下移', lambda: self.move(1))]:
            button = Q.QPushButton(label); row.addWidget(button); button.clicked.connect(fn)
        note = Q.QLabel('高度为从最高站姿下降：正向0～56cm，反向0～53cm。头部0°正前方，正数俯视，负数仰视。\n导航途中调姿，二者完成后继续；抓取复位后调姿再识别；放置/固定动作开始前调姿。固定动作随后按记录的全身姿态回放。')
        note.setWordWrap(True); layout.addWidget(note)
        row = Q.QHBoxLayout(); layout.addLayout(row)
        for label, fn in [('保存流程', self.save), ('删除流程', self.delete), ('开始流程', self.start)]:
            button = Q.QPushButton(label); row.addWidget(button); button.clicked.connect(fn)
        self.names.currentTextChanged.connect(self.load)
        self.waypoint_timer=C.QTimer(self);self.waypoint_timer.setInterval(750)
        self.waypoint_timer.timeout.connect(self.refresh_waypoints)
        self.refresh_waypoints()

    def showEvent(self,event):
        super().showEvent(event);self.refresh_waypoints();self.waypoint_timer.start()

    def navigation_map(self):
        from .navigation_ui import NavigationDialog
        panel=self.owner.navigation
        if panel.dialog:panel.dialog.close()
        dialog=NavigationDialog(panel,parent=self);panel.dialog=dialog
        try:dialog.exec_()
        finally:panel.dialog=None;dialog.deleteLater()

    def done(self,result):
        self.waypoint_timer.stop();super().done(result)

    def refresh_waypoints(self):
        names=self.choices('navigate');data=self.owner.navigation.data
        from pathlib import Path
        label=Path(data.get('active_map') or '').stem or '当前地图'
        unavailable=False
        for i in range(self.table.rowCount()):
            if self.table.item(i,0).data(C.Qt.UserRole)!='navigate':continue
            box=self.table.cellWidget(i,1);selected=box.currentText()
            if box.property('waypoint_choices')!=names:
                cursor=box.lineEdit().cursorPosition() if box.isEditable() else 0
                box.blockSignals(True);box.clear();box.addItems(names)
                if selected:box.setCurrentText(selected);box.lineEdit().setCursorPosition(cursor)
                box.setProperty('waypoint_choices',names);box.blockSignals(False)
            invalid=bool(selected and selected not in names);unavailable|=invalid
            box.setToolTip('该航点不在当前列表，请重新选择' if invalid else '点击右侧下拉箭头选择控制中心航点')
        suffix='导航已连接' if data.get('ready') else '可先编排；执行前需启动导航并完成定位'
        if unavailable:suffix+='；部分步骤航点已失效，请重新选择'
        self.nav_note.setText(f'{label} · {len(names)}个航点 · {suffix}' if names else
            '尚未读取到航点，请在控制中心选择有航点的地图后刷新')

    def choices(self, kind):
        if kind == 'navigate':
            from .navigation_logic import waypoint_names
            return waypoint_names(self.owner.navigation.data)
        if kind == 'action': return sorted(self.owner.studio.records())
        if kind == 'grasp':
            path = self.owner.root/'data/products.json'
            return [p['name'] for p in json.loads(path.read_text()).get('products', [])] if path.exists() else []
        return ['当前桌面']

    def add(self, kind, target, posture=None, base=None):
        row = self.table.rowCount(); self.table.insertRow(row)
        item = Q.QTableWidgetItem(KINDS[kind]); item.setData(C.Qt.UserRole, kind)
        item.setFlags(item.flags() & ~C.Qt.ItemIsEditable); self.table.setItem(row, 0, item)
        box = Q.QComboBox(); box.setEditable(kind != 'place'); box.addItems(self.choices(kind))
        if kind=='navigate':
            box.setProperty('waypoint_choices',self.choices(kind))
            box.setMinimumContentsLength(12);box.setMaxVisibleItems(16)
            box.lineEdit().setPlaceholderText('点击右侧 ▼ 选择航点')
        if target: box.setCurrentText(target)
        if kind=='base':
            from .base_sequence_ui import BaseCell
            box.deleteLater();box=BaseCell(self.table,base)
        self.table.setCellWidget(row, 1, box); self.table.selectRow(row)
        p=posture or {};direction=Q.QComboBox()
        for label,value in [('保持当前','keep'),('正向下蹲','forward'),('反向下蹲','reverse')]:direction.addItem(label,value)
        direction.setCurrentIndex(max(0,direction.findData(p.get('body_mode','keep'))))
        height=Q.QDoubleSpinBox();height.setDecimals(1);height.setSuffix(' cm')
        def sync_height():
            height.setMaximum(53. if direction.currentData()=='reverse' else 56.)
            height.setEnabled(True)
        direction.currentIndexChanged.connect(sync_height);sync_height();height.setValue(p.get('lowering_cm',0))
        head=Q.QWidget();head_row=Q.QHBoxLayout(head);head_row.setContentsMargins(2,0,2,0)
        enable=Q.QCheckBox('设置');angle=Q.QDoubleSpinBox();angle.setRange(-30,45);angle.setDecimals(1);angle.setSuffix('°')
        enable.setChecked(p.get('head_pitch_deg') is not None);angle.setValue(p.get('head_pitch_deg') or 0)
        # Editing is always available; a user edit opts this field into the step.
        height.setToolTip('直接输入高度会启用正向下蹲；需要反向时选择“反向下蹲”。选择“保持当前”则不调身体。')
        height.valueChanged.connect(lambda _:direction.setCurrentIndex(direction.findData('forward')) if direction.currentData()=='keep' else None)
        angle.setToolTip('直接修改角度会自动勾选“设置”；取消勾选则保持当前头部角度。')
        angle.valueChanged.connect(lambda _:enable.setChecked(True))
        head_row.addWidget(enable);head_row.addWidget(angle);head.enable=enable;head.angle=angle
        self.table.setCellWidget(row,2,direction);self.table.setCellWidget(row,3,height);self.table.setCellWidget(row,4,head)
        self.table.setRowHeight(row,56)

    def record(self):
        return validate({'steps': [dict(kind=self.table.item(i, 0).data(C.Qt.UserRole),
            target=('底盘定量动作' if self.table.item(i,0).data(C.Qt.UserRole)=='base' else self.table.cellWidget(i, 1).currentText()),
            base=(self.table.cellWidget(i,1).value if self.table.item(i,0).data(C.Qt.UserRole)=='base' else None),posture=dict(
                body_mode=self.table.cellWidget(i,2).currentData(),lowering_cm=self.table.cellWidget(i,3).value(),
                head_pitch_deg=self.table.cellWidget(i,4).angle.value() if self.table.cellWidget(i,4).enable.isChecked() else None)) for i in range(self.table.rowCount())]})

    def load(self, name):
        self.table.setRowCount(0); self.name.setText('' if name == '新建流程' else name)
        for row in self.library.get(name, {}).get('steps', []): self.add(row['kind'], row['target'],row.get('posture'),row.get('base'))

    def remove(self):
        if self.table.currentRow() >= 0: self.table.removeRow(self.table.currentRow())

    def move(self, delta):
        i = self.table.currentRow(); j = i + delta
        if not 0 <= j < self.table.rowCount(): return
        try: rows = self.record()['steps']
        except ValueError: return
        rows[i], rows[j] = rows[j], rows[i]; self.table.setRowCount(0)
        for row in rows: self.add(row['kind'], row['target'],row.get('posture'),row.get('base'))
        self.table.selectRow(j)

    def save(self):
        try:
            name = self.name.text().strip()
            if not name or len(name)>80: raise ValueError('请填写简短的流程名称')
            record = self.record()
            if name in self.library and Q.QMessageBox.question(self, '覆盖流程', '覆盖同名流程？', defaultButton=Q.QMessageBox.No) != Q.QMessageBox.Yes: return
            self.library[name] = record; atomic_json(self.path, self.library)
            self.names.blockSignals(True); self.names.clear(); self.names.addItem('新建流程'); self.names.addItems(sorted(self.library)); self.names.setCurrentText(name); self.names.blockSignals(False)
        except (ValueError, OSError) as exc: Q.QMessageBox.warning(self, '保存失败', str(exc))

    def delete(self):
        name = self.names.currentText()
        if name in self.library and Q.QMessageBox.question(self, '删除流程', '删除“'+name+'”？', defaultButton=Q.QMessageBox.No) == Q.QMessageBox.Yes:
            del self.library[name]; atomic_json(self.path, self.library); self.names.removeItem(self.names.currentIndex())

    def start(self):
        try: self.owner.workflow.start(self.record())
        except Exception as exc: Q.QMessageBox.warning(self, '不能开始', str(exc)); return
        if self.owner.workflow.active: self.accept()


class Workflow(C.QObject):
    started=C.pyqtSignal(str)
    completed=C.pyqtSignal(str)
    def __init__(self, owner):
        super().__init__(owner); self.owner = owner; self.cursor = None; self.phase = ''; self.plan_id = None
        self.actions = {}; self.map = None
        self.failed_phase=''
        self.candidates=[];self.candidate_failures=[];self.candidate_index=0
        self.posture_resume=None;self.nav_arrived=False;self.nav_posture_done=True;self.nav_posture_started=False
        owner.studio.completed.connect(self.action_done)
        from .base_sequence_ui import BaseRunner
        self.base=BaseRunner(owner);self.base.completed.connect(self.base_done)

    @property
    def active(self): return bool(self.cursor and self.cursor.active)

    @property
    def retry_available(self):
        return bool(self.cursor and not self.active and self.failed_phase and
                    0<=self.cursor.index<len(self.cursor.record['steps']))

    def stopped(self,reason):
        if not self.active:return
        self.failed_phase=self.phase;self.cursor.stop(reason);self.phase=''

    def retry_current(self):
        """Explicit retry of this step only. Never resume a stale motor path."""
        o=self.owner
        try:
            if not self.retry_available:raise ValueError('没有已停止的步骤可重试')
            if (not o.ready or o.busy or o.motion or o.cancel_pending or o.demo_active or o.placement_active
                    or o.navigation.inflight or o.studio.moving() or self.base.moving()):
                raise ValueError('请等待前一次操作完全停止、后台就绪后再重试')
            row=self.cursor.step;kind=row['kind'];scan_only=False
            if kind=='navigate':
                data=o.navigation.data
                if (time.time()-data.get('snapshot_time',0)>2 or not data.get('ready') or not data.get('pose')
                        or data.get('map_fault') or data.get('active_map')!=self.map or row['target'] not in data.get('names',[])):
                    raise ValueError('请先恢复原地图的导航与定位，再重试该航点')
            if kind=='grasp':
                if (o.run/'held_object.json').exists():raise ValueError('仍有持物记录，请先确认并处理手中物品，不能重复抓取或张爪复位')
                path=o.run/'prepare_result.json'
                prepared=json.loads(path.read_text('utf-8')) if path.exists() else {}
                scan_only=self.failed_phase in ('scan','plan','confirm') and prepared.get('status')=='prepared'
                detail=('重新识别并规划；'+('规划通过即自动抓取' if o.auto_grasp.isChecked() else '等待您确认抓取')+'，不重复复位。') if scan_only else '将重新复位并张开夹爪，再识别规划；请先确认双夹爪空载，未夹住物品。'
            elif kind=='place':
                from .placement_request import held_grasp
                held_grasp(o.run)
                detail='重新识别桌面并规划放置；请确认物品仍夹稳、未在上次操作中放下。'
            elif kind=='action':detail='从该固定动作的第一帧重新执行，不重复之前的导航或抓取。请确认当前姿态和现场适合重做。'
            elif kind=='base':detail='将从当前位置重新执行本步骤全部底盘动作，已经走过的部分也会重做。请确认当前位置适合重新执行；否则先修改流程。'
            else:detail='从当前位置重新规划到该航点，不重复之前已完成的步骤。'
            if Q.QMessageBox.question(o,'重试当前步骤',
                    f'第{self.cursor.index+1}步 · {KINDS[kind]} · {row["target"]}\n{detail}\n请清空运动范围并准备实体急停，是否重试？',
                    defaultButton=Q.QMessageBox.No)!=Q.QMessageBox.Yes:return
            o.studio.release();o.invalidate_plan();self.plan_id=None;self.posture_resume=None
            self.cursor.retry();self.failed_phase=''
            atomic_json(o.run/'task_flow.json',dict(run_id=self.cursor.run_id,map=self.map,record=self.cursor.record,
                        actions=self.actions,retry_index=self.cursor.index))
            (o.run/'stop').unlink(missing_ok=True)
            if scan_only:
                o.flow_label.setText(f'重试第{self.cursor.index+1}步 · 重新识别与规划')
                self.start_scan();o.controls()
            elif kind=='action':
                # The initial handoff consumed held_object.json. Preserve its
                # grip policy when retrying the same action, not a new task.
                self.action_hold_grippers=getattr(self,'action_hold_grippers',False)
                self.phase='action';self.adjust_posture(self.start_fixed_action);o.controls()
            else:self.next()
        except Exception as exc:
            if self.active:self.fail(str(exc))
            else:Q.QMessageBox.warning(o,'暂不能重试',str(exc))

    def start(self, record, *, confirmed=False, actions_snapshot=None, expected_map=None):
        o = self.owner
        auto=getattr(o,'automatic',None)
        if auto and auto.active and not (confirmed and auto.launching): raise ValueError('自动任务正在运行')
        if o.cancel_pending or not o.ready: raise ValueError('后台未就绪或正在停止')
        if self.active or o.busy or o.motion or o.demo_active or o.placement_active or o.navigation.inflight or self.base.moving() or o.studio.moving(): raise ValueError('当前操作未结束')
        record = validate(record); actions = deepcopy(actions_snapshot) if actions_snapshot is not None else o.studio.records()
        from .action_studio import full_record
        self.actions = {}
        for row in record['steps']:
            if row['kind'] == 'action':
                if row['target'] not in actions: raise ValueError('动作库中没有：'+row['target'])
                self.actions[row['target']] = full_record(actions[row['target']])
        data = o.navigation.data
        if any(r['kind']=='navigate' for r in record['steps']):
            if expected_map is not None and data.get('active_map')!=expected_map: raise ValueError('地图已变化，请重新选择并启动任务')
            if time.time()-data.get('snapshot_time', 0)>2 or not data.get('ready') or not data.get('pose') or data.get('map_fault'):
                raise ValueError('请点击抓取页“启动导航”，并在“放大地图 / 定位”中完成定位')
            names = data.get('names', [])
            for row in record['steps']:
                if row['kind']=='navigate' and row['target'] not in names: raise ValueError('当前地图没有航点：'+row['target'])
        text = '\n'.join(f'{i+1}. {KINDS[r["kind"]]} · {r["target"]}'+(' · 含身体/头部调姿' if r.get('posture') else '') for i,r in enumerate(record['steps']))
        from .base_sequence import summary
        text+=''.join('\n底盘：'+summary(r['base']) for r in record['steps'] if r['kind']=='base')
        grasp_mode='已勾选自动执行抓取，规划通过后立即抓取，无需逐次确认。' if o.auto_grasp.isChecked() else '抓取先预览再确认。'
        if not confirmed and Q.QMessageBox.question(o, '确认任务流程', text+'\n\n将自动执行导航、放置和固定动作（包括腰腿与夹爪）。每次抓取前会复位并张开夹爪，请确认此时夹爪空载；'+grasp_mode+'\n固定动作没有门阻力检测，请确认现场与示教时一致、全程看护并准备实体急停。', defaultButton=Q.QMessageBox.No) != Q.QMessageBox.Yes: return
        o.studio.release(); o.demo.setChecked(False); self.cursor = Cursor(record); self.map = data.get('active_map');self.failed_phase=''
        (o.run/'stop').unlink(missing_ok=True)
        atomic_json(o.run/'task_flow.json', dict(run_id=self.cursor.run_id, map=self.map,
            record=record, actions=self.actions))
        self.started.emit(self.cursor.run_id)
        self.next()

    def next(self):
        if getattr(self.owner,'automatic',None) and defer_call(self.owner,self.next):return
        if not self.active: return
        o = self.owner; row = self.cursor.step; self.phase = row['kind']; self.plan_id = None
        self.posture_resume=None;self.nav_arrived=False;self.nav_posture_done=not bool(row.get('posture'));self.nav_posture_started=False
        o.flow_label.setText(f'任务 {self.cursor.index+1}/{len(self.cursor.record["steps"])} · {KINDS[row["kind"]]} · {row["target"]}')
        try:
            o.studio.release()
            if row['kind']=='base':
                if not row['base']['enabled']:self.advance();return
                self.adjust_posture(self.start_base)
            elif row['kind']=='navigate':
                if not o.navigation.proc or o.navigation.proc.state()!=C.QProcess.Running: raise ValueError('导航连接未启动')
                if row.get('posture'):
                    # Confirm hold/SYNC while the base is stationary. Do not
                    # spend the whole navigation leg enabling a new controller.
                    self.phase='prepare_navigation'
                    o.flow_label.setText('出发准备：保持当前关节姿态；出发后同步调姿')
                    o.studio.prepare_posture(self.cursor.token+':nav_prepare')
                else:self.send_navigation()
            elif row['kind']=='action':
                held=o.run/'held_object.json'
                receipt=json.loads(held.read_text('utf-8')) if held.exists() else {}
                self.action_hold_grippers=receipt.get('status')=='completed' and bool(receipt.get('plan_id'))
                # A user-authored all-joint action may open the gripper; stale
                # grasp provenance must not be reused for automatic placement.
                (o.run/'held_object.json').unlink(missing_ok=True)
                (o.run/'prepare_result.json').unlink(missing_ok=True)
                self.adjust_posture(self.start_fixed_action)
            elif row['kind']=='grasp':
                if (o.run/'held_object.json').exists(): raise ValueError('仍有持物记录；先放置或执行已确认的释放动作，不能自动张爪复位')
                self.phase='prepare'; o.send('auto_prepare' if getattr(o,'automatic',None) and o.automatic.active else 'prepare')
            else:
                self.adjust_posture(self.start_place)
            o.controls()
        except Exception as exc: self.fail(str(exc))

    def send_navigation(self):
        if getattr(self.owner,'automatic',None) and defer_call(self.owner,self.send_navigation):return
        if not self.active:return
        self.phase='navigate'
        self.owner.navigation.command('navigate',name=self.cursor.step['target'],map=self.map,
            run_id=self.cursor.run_id,leg=self.cursor.token,workflow=True)
        self.owner.flow_label.setText('正在出发至 '+self.cursor.step['target'])
        self.owner.controls()

    def adjust_posture(self,resume):
        if defer_call(self.owner,self.adjust_posture,resume):return
        if not self.cursor.step.get('posture'):resume();return
        self.posture_resume=resume
        if self.phase!='navigate':self.phase='posture'
        self.owner.flow_label.setText('任务调姿中 · 保持手臂与夹爪；导航步骤会同时行驶')
        self.owner.studio.adjust_posture(self.cursor.token+':posture',self.cursor.step['posture'])
        self.owner.controls()

    def start_fixed_action(self):
        if getattr(self.owner,'automatic',None) and defer_call(self.owner,self.start_fixed_action):return
        self.phase='action';self.owner.studio.execute(self.cursor.token,deepcopy(self.actions[self.cursor.step['target']]),
            preserve_grippers_at_entry=self.action_hold_grippers)

    def start_base(self):
        if getattr(self.owner,'automatic',None) and defer_call(self.owner,self.start_base):return
        if self.owner.navigation.inflight:raise ValueError('导航尚未停止，未开始定量移动')
        self.phase='base';self.owner.studio.release();self.owner.invalidate_plan();self.owner.live()
        (self.owner.run/'prepare_result.json').unlink(missing_ok=True)
        self.base.start(self.cursor.token)

    def base_done(self,token,success,text):
        if not self.active or self.phase!='base' or token!=self.cursor.token:return
        if success:self.advance()
        else:self.fail(text)

    def start_place(self):
        if getattr(self.owner,'automatic',None) and defer_call(self.owner,self.start_place):return
        from .placement_request import new_standalone_request
        from .demo_config import load
        o=self.owner;self.phase='place';o.demo_context=new_standalone_request(o.run,load(o.root))
        o.placement_active=True;o.demo_phase='planning';o.send('standalone_place')

    def start_scan(self):
        if getattr(self.owner,'automatic',None) and defer_call(self.owner,self.start_scan):return
        o=self.owner;path=o.root/'data/products.json'
        products=json.loads(path.read_text()).get('products',[]) if path.exists() else []
        from .task_recognition import task_detection_request
        target=self.cursor.step['target'];request=task_detection_request(target,products)
        self.semantic_target=request['semantic']
        self.phase='scan';o.send('scan_task',target=target)

    def plan_next_candidate(self):
        if getattr(self.owner,'automatic',None) and defer_call(self.owner,self.plan_next_candidate):return
        if not self.active or self.candidate_index>=len(self.candidates):
            raise ValueError('同类候选均未通过规划检查：'+'；'.join(self.candidate_failures))
        o=self.owner;instance=self.candidates[self.candidate_index];self.candidate_index+=1
        o.list.setCurrentRow(next(i for i,v in enumerate(o.items) if v['id']==instance))
        self.phase='plan';o.send('plan',id=instance)
        o.flow_label.setText(f'自动选择 {self.cursor.step["target"]} · 候选 {self.candidate_index}/{len(self.candidates)} · 正在规划')
        atomic_json(o.run/'task_candidate_selection.json',dict(token=self.cursor.token,
            candidates=self.candidates,selected=instance,failures=self.candidate_failures))

    def navigation_started(self,data):
        if not self.active or self.phase!='navigate' or data.get('run_id')!=self.cursor.run_id or data.get('leg')!=self.cursor.token:return
        if self.nav_posture_started:return
        self.nav_posture_started=True
        try:self.adjust_posture(self.navigation_posture_done)
        except Exception as exc:self.fail(str(exc))

    def navigation_posture_done(self):
        self.nav_posture_done=True
        if self.nav_arrived:self.advance()
        else:self.owner.flow_label.setText('任务姿态已到位 · 继续导航至 '+self.cursor.step['target'])

    def advance(self):
        if getattr(self.owner,'automatic',None) and defer_call(self.owner,self.advance):return
        if not self.active: return
        self.cursor.complete(self.cursor.token)
        if self.active: C.QTimer.singleShot(0, self.next)
        else:
            self.phase=''; self.owner.flow_label.setText('任务流程已完成'); self.owner.controls()
            self.completed.emit(self.cursor.run_id)

    def fail(self, reason):
        if not self.active: return
        self.owner.stop_motion(reason=reason,source='workflow')
        self.owner.flow_label.setText('任务停止：'+reason); self.owner.controls()

    def worker_event(self, data):
        if not self.active: return
        o=self.owner; kind=data.get('event')
        try:
            if kind in ('error','fatal'):
                if kind=='error' and self.phase=='plan' and data.get('candidate_unavailable'):
                    self.candidate_failures.append(f'#{self.candidates[self.candidate_index-1]}: '+data['text'])
                    self.plan_next_candidate();return
                raise ValueError(data['text'])
            if kind=='reset_planned' and data.get('blockers'): raise ValueError('；'.join(data['blockers']))
            if kind=='scene' and self.phase=='scan':
                path=o.root/'data/products.json'; products=json.loads(path.read_text()).get('products', []) if path.exists() else []
                target=self.cursor.step['target']
                if data.get('task_recognition') is not None:
                    request=data['task_recognition']
                    if request['target']!=target:raise ValueError('识别结果与当前任务物品不一致')
                    self.semantic_target=request['semantic']
                self.candidates=select_candidates(o.items,target,products,semantic=self.semantic_target)
                self.candidate_index=0;self.candidate_failures=[];self.plan_id=None
                self.plan_next_candidate()
            elif kind=='planned' and self.phase=='plan':
                if data.get('blockers'): raise ValueError('；'.join(data['blockers']))
                if o.pending_plan is None: raise ValueError('规划目标校验未通过')
                self.plan_id=data['plan_id']; self.phase='confirm'
                o.flow_label.setText('路径已通过 · 准备自动抓取' if o.auto_grasp_gate.intent else
                    '路径已预览 · 请点击“确认抓取”继续任务流程')
            o.controls()
        except Exception as exc: self.fail(str(exc))

    def motion_done(self, result):
        if not self.active: return False
        o=self.owner
        if self.phase=='prepare':
            if result.get('status')!='prepared': self.fail(result.get('reason','复位失败')); return True
            try:self.adjust_posture(self.start_scan)
            except Exception as exc:self.fail(str(exc))
            return True
        if self.phase=='grasp':
            if result.get('status')=='completed' and result.get('plan_id')==self.plan_id: self.advance()
            else: self.fail(result.get('reason','抓取未成功，未继续后续任务'))
            return True
        return False

    def navigation_done(self, data):
        if not self.active or self.phase!='navigate' or data.get('run_id')!=self.cursor.run_id or data.get('leg')!=self.cursor.token: return
        if data.get('success') and data.get('map')==self.map:
            self.nav_arrived=True
            if self.nav_posture_done:self.advance()
            else:self.owner.flow_label.setText('航点已到达 · 等待身体和头部调姿完成')
        else: self.fail(data.get('text','导航失败或地图已变化'))

    def action_done(self, token, success, text):
        if self.active and self.phase=='prepare_navigation' and token==self.cursor.token+':nav_prepare':
            if success:self.send_navigation()
            else:self.fail(text)
            return
        if self.active and token==self.cursor.token+':posture' and self.posture_resume:
            resume=self.posture_resume;self.posture_resume=None
            if not success:self.fail(text);return
            try:resume();self.owner.controls()
            except Exception as exc:self.fail(str(exc))
            return
        if not self.active or self.phase!='action' or token!=self.cursor.token: return
        if success: self.advance()
        else: self.fail(text)
