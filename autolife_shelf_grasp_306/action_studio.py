"""Embedded full-body teaching adapter; reuse the verified joint controller.

The bridge exists only while teaching or executing a named action. Releasing
it clears software goals, not motor holding power. Grasp keeps its own lease.
"""
import json
import os
import time
from pathlib import Path
from PyQt5 import QtCore as C, QtWidgets as Q
from .task_flow import full_record
from copy import deepcopy


def selected_frame_action(record,row):
    record=full_record(record)
    if not 0<=row<len(record['keyframes']):raise ValueError('请先选中一个关键帧')
    result=deepcopy(record);result['keyframes']=[deepcopy(record['keyframes'][row])]
    result['loop_count']=1;result['keyframes'][0]['hold_sec']=0.
    return result


def measured_frame_values(status):
    from control_center_joint_control.joint_definitions import ALL_JOINTS,JOINT_LIMITS
    import math
    age=float(status.get('feedback_age_sec',float('inf')))
    if not status.get('connected') or not math.isfinite(age) or not 0<=age<=.6:
        raise ValueError('等待新鲜关节反馈')
    if status.get('action_running') or status.get('active_goal_groups') or status.get('hardware_prepare_stage'):
        raise ValueError('请等待关节到位停稳后再记录')
    values={}
    for n in ALL_JOINTS:
        d=status.get('joints',{}).get(n,{})
        if d.get('communication_lost',True) or d.get('current_deg') is None:raise ValueError('关节反馈不完整：'+n)
        v=float(d['current_deg']);lo,hi=JOINT_LIMITS[n]
        if not math.isfinite(v) or not lo-2<=v<=hi+2:raise ValueError('关节实测角度超限：'+n)
        speed=float(d.get('speed_deg_s',float('inf')))
        if not math.isfinite(speed) or abs(speed)>2.:raise ValueError('关节仍在运动或速度反馈无效，请停稳后再记录')
        values[n]=min(hi,max(lo,v))
    return values


def entry_gripper_hold(record,current):
    """Runtime-only handoff: do not squeeze carried objects to a taught width."""
    from control_center_joint_control.joint_definitions import GRIPPER_JOINTS,JOINT_LIMITS
    import math
    result=deepcopy(record)
    for n in GRIPPER_JOINTS:
        value=float(current[n]);lo,hi=JOINT_LIMITS[n]
        if not math.isfinite(value) or not lo-2.<=value<=hi+2.:
            raise ValueError('持物动作夹爪反馈无效：'+n)
        # This robot uses 10deg open / 330deg closed. A deliberate first-frame
        # opening must remain intact; only suppress additional closing.
        if result['keyframes'][0]['joints_deg'][n]>=value:
            result['keyframes'][0]['joints_deg'][n]=min(hi,max(lo,value))
    return full_record(result)


def make_panel(node):
    from control_center_joint_control.control_panel import ControlPanel
    from control_center_joint_control.action_editor import ActionLibraryPage
    from control_center_joint_control.joint_definitions import ALL_JOINTS, GROUP_ORDER

    class AllJointPage(ActionLibraryPage):
        def __init__(self):
            super().__init__()
            self.record_measured=Q.QPushButton('记录实测全身姿态')
            self.goto_button=Q.QPushButton('到达选中帧（真机）')
            self.replace_measured=Q.QPushButton('实测替换选中帧')
            line=Q.QHBoxLayout();line.addWidget(self.record_measured);line.addWidget(self.goto_button);line.addWidget(self.replace_measured)
            self.layout().insertLayout(1,line)
            self.replace_button.setText('替换为编辑目标')
            self.goto_button.setToolTip('确认后仅执行选中帧，包含腰腿和夹爪；到位后可继续微调，不会执行后续帧。')
            self._update_controls()

        def selected_groups(self):
            return list(GROUP_ORDER)

        def suggest_groups(self, groups):
            pass

        def _update_controls(self):
            for check in self.group_checks.values():
                check.blockSignals(True); check.setChecked(True); check.blockSignals(False)
            super()._update_controls()
            self.group_summary.setText('始终记录全身23个关节；到达时间越短，要求速度越快（受驱动限制）')
            if hasattr(self,'goto_button'):
                enabled=0<=self.frame_list.currentRow()<len(self._frames) and not getattr(self,'_robot_running',False)
                self.goto_button.setEnabled(enabled);self.replace_measured.setEnabled(enabled)

        def set_progress(self,running,*args,**kwargs):
            super().set_progress(running,*args,**kwargs)
            self._robot_running=bool(running);self._update_controls()

        def build_record(self):
            return full_record(super().build_record())

    class FullPanel(ControlPanel):
        def go_to_selected_frame(self):
            try:
                if self._status.get('action_running') or self._status.get('active_goal_groups'):
                    raise ValueError('当前关节正在运动，请先停止并停稳')
                measured_frame_values(self._status)
                row=self.action_page.frame_list.currentRow()
                record=selected_frame_action(self.action_page.build_record(),row)
            except (ValueError,KeyError,TypeError) as exc:
                Q.QMessageBox.warning(self,'不能到达关键帧',str(exc));return
            # Selecting/previewing a frame must never queue live motor goals.
            self._pending_live_groups.clear();self._pending_live_joint_names.clear()
            self._set_checked_silently(self.live_toggle,False)
            self.execute_joint_action(f'到达第{row+1}帧（含腰腿与夹爪）',record)

        def capture_measured_frame(self,replace=False):
            try:
                if replace and self.action_page.frame_list.currentRow()<0:raise ValueError('请先选中要替换的关键帧')
                self.action_page.add_captured_frame(measured_frame_values(self._status),replace)
                self.event_label.setText('已替换选中帧；点击“保存动作”写入动作库' if replace else '已记录实测全身姿态')
            except (ValueError,KeyError,TypeError) as exc:Q.QMessageBox.warning(self,'不能记录',str(exc))

        def capture_action_frame(self, replace):
            try:
                self.action_page.add_captured_frame(self._action_target_values(ALL_JOINTS), replace)
            except ValueError as exc:
                Q.QMessageBox.warning(self, '不能记录', str(exc))

        def save_joint_action(self, name, record):
            try: record = full_record(record)
            except ValueError as exc:
                Q.QMessageBox.warning(self, '不能保存', str(exc)); return
            super().save_joint_action(name, record)

    panel = FullPanel(node)
    old = panel.action_page
    index = panel.tabs.indexOf(old)
    panel.tabs.removeTab(index); old.deleteLater()
    page = AllJointPage(); panel.action_page = page
    page.findChild(Q.QFrame, 'actionGroupSelector').hide()
    page.capture_requested.connect(panel.capture_action_frame)
    page.frame_selected.connect(panel.preview_action_frame)
    page.save_requested.connect(panel.save_joint_action)
    page.delete_requested.connect(panel.delete_joint_action)
    page.preview_requested.connect(panel.preview_joint_action)
    page.stop_preview_requested.connect(panel.stop_action_preview)
    page.execute_requested.connect(panel.execute_joint_action)
    page.cancel_requested.connect(panel.cancel_joint_action)
    page.goto_button.clicked.connect(panel.go_to_selected_frame)
    page.replace_measured.clicked.connect(lambda:panel.capture_measured_frame(True))
    page.record_measured.clicked.connect(lambda:panel.capture_measured_frame(False))
    page.set_actions(panel._actions, panel._action_file)
    panel.tabs.insertTab(index, page, '全身动作库')
    panel.tabs.setCurrentWidget(page)
    return panel


class ActionStudio(Q.QWidget):
    """Lazy ROS/UI lifetime, exclusive vendor bridge, exact completion receipt."""
    completed = C.pyqtSignal(str, bool, str)

    def __init__(self, owner):
        super().__init__(); self.owner = owner; self.bridge = None; self.panel = None
        self.node = None; self.job = None; self._ros = False
        self.library = owner.root / 'data' / 'fixed_actions'
        self.layout_ = Q.QVBoxLayout(self)
        self.note = Q.QLabel('模型编辑与模拟不驱动真机。固定轨迹只适用于已示教的相对位置；不能感知门阻力或环境变化。')
        self.note.setWordWrap(True); self.layout_.addWidget(self.note)
        self.timer = C.QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(10)

    def ensure(self):
        import rclpy
        from control_center_joint_control.control_panel import JointControlRosNode
        from control_center_joint_control.joint_bridge import JointBridge
        if not self._ros:
            if not rclpy.ok():
                names = ['preview_command', 'apply_command', 'action_command', 'status',
                         'set_hardware_enabled', 'set_hardware_enabled_without_reset', 'set_leg_waist_enabled']
                args = ['--ros-args']
                for n in names: args += ['-r', f'/control_center_joint_control/{n}:=/shelf_grasp_joint_control/{n}']
                rclpy.init(args=args)
            self._ros = True
        if self.panel is None:
            self.library.mkdir(parents=True, exist_ok=True)
            # Do not import unrelated partial-body actions from the control center.
            for filename, key in [('saved_joint_actions.yaml', 'actions'), ('saved_joint_poses.yaml', 'poses')]:
                path = self.library / filename
                if not path.exists(): path.write_text('version: 1\n' + key + ': {}\n', encoding='utf-8')
            key = 'CONTROL_CENTER_JOINT_CONTROL_LIBRARY_DIR'; previous = os.environ.get(key)
            os.environ[key] = str(self.library)
            try:
                self.node = JointControlRosNode(); self.panel = make_panel(self.node)
            finally:
                if previous is None: os.environ.pop(key, None)
                else: os.environ[key] = previous
            self.panel.setWindowFlags(C.Qt.Widget); self.layout_.addWidget(self.panel)
        if self.bridge is None:
            class ReceiptBridge(JointBridge):
                outcome = None
                def _advance_motion(inner):
                    if getattr(inner,'task_pause',None) and inner.task_pause.tick():return
                    super()._advance_motion()
                def _record_action_event(inner,kind,reason=''):
                    # Editor playback and task-flow playback share this bridge.
                    # Persist evidence before cancellation clears action state.
                    state=inner._action_state
                    if not state:return
                    try:
                        frame=state['record']['keyframes'][state['frame_index']]
                        entry=dict(time=time.time(),event=kind,reason=str(reason),name=state['name'],
                            frame_index=state['frame_index']+1,loop_index=state['loop_index'],
                            targets=frame['joints_deg'],measured=dict(inner._current),
                            commanded=dict(state['commanded_values']),details=dict(inner._details),
                            feedback_age_sec=time.monotonic()-inner._feedback_monotonic)
                        from .core import atomic_json
                        with (self.owner.run/'action_events.jsonl').open('a',encoding='utf-8') as stream:
                            stream.write(json.dumps(entry,ensure_ascii=False)+'\n')
                        if kind in ('stopped','locked'):atomic_json(self.owner.run/'last_action_error.json',entry)
                    except (OSError,ValueError,KeyError,TypeError,AttributeError):pass
                def _action_target_reached(inner,targets):
                    if super()._action_target_reached(targets):
                        from .action_arrival import remember_opening_arrival
                        remember_opening_arrival(inner,targets)
                        if inner._action_state:
                            inner._action_state.pop('opening_arrival',None)
                            inner._action_state.pop('arm_arrival',None)
                        return True
                    from .action_arrival import settled_opening_arrival
                    if settled_opening_arrival(inner,targets):
                        inner._record_action_event('settled_opening','松开或保持已确认松开状态，夹爪残差≤5°并稳定0.5秒')
                        inner._event('松爪/保持松开已稳定，小于5°残差继续下一帧；身体与手臂容差不变')
                        return True
                    from .action_pose_arrival import settled_arm_arrival,PoseEnvelope
                    def geometry(actual,target):
                        if not hasattr(inner,'_arrival_envelope'):
                            import yaml
                            cfg=yaml.safe_load((self.owner.root/'config/grasp.yaml').read_text('utf-8'))
                            inner._arrival_envelope=PoseEnvelope(cfg['urdf_path'])
                        return inner._arrival_envelope.check(actual,target)
                    receipt=settled_arm_arrival(inner,targets,geometry)
                    if receipt is None:return False
                    # Only already stationary arms are held at their measured
                    # position. Do not change jaws, waist/legs, neck or records.
                    from control_center_joint_control.joint_definitions import GROUPS,group_joint_names
                    affected={side+'_arm' for side in ('left','right') if any(
                        n.startswith('Joint_'+side.title()+'_') for n in receipt['errors_deg'])}
                    state=inner._action_state
                    payload={}
                    for group in ('neck','left_arm','right_arm','leg_waist'):
                        names=group_joint_names(group)
                        values=[float(inner._current[n]) if group in affected else
                                float(state['commanded_values'][n] if n in state['commanded_values'] else
                                      inner._controller_target[n]) for n in names]
                        payload[GROUPS[group]['command_key']]=values
                        if group in affected:state['commanded_values'].update(zip(names,values))
                    inner._publish_vendor(inner._whole_body_pub,payload)
                    state.setdefault('arrival_deviations',[]).append(receipt)
                    inner._record_action_event('settled_arm_tolerance',json.dumps(receipt,ensure_ascii=False))
                    inner._event('手臂小偏差已停稳并通过5mm运动学包络检查，从实测姿态衔接；未跳过关键帧')
                    return True
                def _action_error_summary(inner,frame):
                    errors=sorted(((abs(float(inner._current[n])-float(v)),n,float(v))
                        for n,v in frame['joints_deg'].items()),reverse=True)[:3]
                    return '；'.join(f'{n} 实测{inner._current[n]:.3f}°/目标{v:.3f}°/偏差{e:.3f}°'
                                    for e,n,v in errors)
                def _advance_action_frame(inner, now):
                    inner._record_action_event('frame_completed')
                    name = inner._action_state['name']
                    deviations=len(inner._action_state.get('arrival_deviations',[]))
                    super()._advance_action_frame(now)
                    if inner._action_state is None:
                        inner.outcome = (name, True, f'动作完成（{deviations}帧以验证后的小偏差衔接）' if deviations else '动作完成')
                def _cancel_action(inner, reason):
                    inner._record_action_event('stopped',reason)
                    if inner._action_state: inner.outcome = (inner._action_state['name'], False, reason)
                    super()._cancel_action(reason)
                def _lock(inner, reason):
                    inner._record_action_event('locked',reason)
                    if inner._action_state: inner.outcome = (inner._action_state['name'], False, reason)
                    super()._lock(reason)
            self.bridge = ReceiptBridge()
            from .task_pause import BridgePause
            self.bridge.task_pause=BridgePause(self.bridge,self.owner.run)
        return self.bridge

    def records(self):
        from control_center_joint_control.action_library import load_action_library
        path = self.library / 'saved_joint_actions.yaml'
        return load_action_library(path) if path.exists() else {}

    def moving(self):
        b = self.bridge
        return bool(b and (b._action_state or b._active_goals or b._hardware_prepare)) or self.job is not None

    def release(self):
        if self.moving(): raise ValueError('动作仍在进行，请先停止并等待停稳再切换页面')
        if self.bridge and self.bridge._hardware_enabled and not self.settled():
            raise ValueError('关节尚未停稳，请稍后切换')
        if self.panel:
            self.panel.stop_action_preview(quiet=True); self.panel.preview_timer.stop()
            self.panel._pending_live_groups.clear(); self.panel._pending_live_joint_names.clear()
            self.panel._pending_preview_groups.clear()
            self.panel._set_checked_silently(self.panel.live_toggle, False)
        if self.bridge:
            self.bridge._lock('切换模块，关节下发已锁定，保持力不变')
            handle = self.bridge._instance_lock_handle
            self.bridge.destroy_node()
            if handle: handle.close()
            self.bridge = None

    def settled(self):
        from control_center_joint_control.joint_definitions import BODY_JOINTS
        b = self.bridge
        return bool(b and b._is_feedback_fresh() and all(
            abs(float(b._details.get(n, {}).get('speed', float('inf')))) <= 2. for n in BODY_JOINTS))

    def execute(self, token, record, *, preserve_grippers_at_entry=False):
        record = full_record(record); b = self.ensure()
        if self.moving(): raise ValueError('关节编辑动作未结束')
        self.panel.setEnabled(False)
        self.job = dict(token=token, record=record, phase='feedback', started=time.monotonic(),
                        preserve_grippers_at_entry=preserve_grippers_at_entry)
        b.outcome = None

    def adjust_posture(self,token,setting):
        from .step_posture import targets
        desired=targets(setting)
        if not desired:raise ValueError('没有设置调姿目标')
        b=self.ensure()
        if self.moving():raise ValueError('关节编辑动作未结束')
        self.panel.setEnabled(False);b.outcome=None
        ready=b._hardware_enabled and b._hardware_ready and b._is_feedback_fresh()
        self.job=dict(token=token,posture=desired,phase='enable' if ready else 'feedback',started=time.monotonic())

    def prepare_posture(self,token):
        """Hold-only enable before navigation; keep this controller for the leg."""
        b=self.ensure()
        if self.moving():raise ValueError('关节编辑动作未结束')
        self.panel.setEnabled(False);b.outcome=None
        self.job=dict(token=token,prepare_only=True,phase='feedback',started=time.monotonic())

    def stop(self):
        self.job = None
        if self.bridge:
            if getattr(self.bridge,'task_pause',None):self.bridge.task_pause.cancel()
            if self.bridge._action_state: self.bridge._cancel_action('操作员停止')
            # Hold the measured pose for unfinished manual goals, too.
            if self.bridge._active_goals and self.bridge._is_feedback_fresh():
                from control_center_joint_control.joint_definitions import GROUPS, group_joint_names
                self.bridge._publish_vendor(self.bridge._whole_body_pub, {
                    GROUPS[g]['command_key']: [self.bridge._current[n] for n in group_joint_names(g)]
                    for g in ('neck','left_arm','right_arm','leg_waist')})
                self.bridge._publish_vendor(self.bridge._gripper_pub, {
                    'left_gripper_target_joints_position': [self.bridge._current['Joint_Left_Gripper']],
                    'right_gripper_target_joints_position': [self.bridge._current['Joint_Right_Gripper']]})
            self.bridge._lock('动作停止，禁止继续下发')
        if self.panel: self.panel.setEnabled(True)

    def tick(self):
        if self.bridge is None: return
        import rclpy
        # Feedback, command timers and service replies exceed 100 callbacks/s.
        # Drain a bounded batch without blocking Qt or sharing mutable bridge
        # state across threads; one callback per 10ms accumulates old feedback.
        deadline=time.monotonic()+.003
        for _ in range(16):
            rclpy.spin_once(self.bridge, timeout_sec=0.)
            if time.monotonic()>=deadline:break
        if not self.job: return
        j = self.job; b = self.bridge
        now=time.monotonic();last=j.get('pause_tick',now);j['pause_tick']=now
        if (self.owner.run/'task_pause').exists():
            failed=b.outcome and b.outcome[0]==j['token'] and not b.outcome[1]
            lost=(b.task_pause.started is not None and not b._hardware_ready) or (j['phase']!='feedback' and not b._hardware_enabled)
            if failed or lost or (b._hardware_enabled and not b._is_feedback_fresh()):
                reason=b.outcome[2] if failed else '暂停期间反馈或控制权失效'
                token=j['token'];self.stop();self.completed.emit(token,False,reason);return
            j['started']+=now-last
            if j.get('stable') is not None:j['stable']+=now-last
            return
        try:
            if time.monotonic()-j['started'] > (300 if j['phase']=='posture' else 60 if j['phase'] != 'running' else 3600):
                raise ValueError('动作准备/执行超时')
            from std_srvs.srv import SetBool
            from std_msgs.msg import String
            if j['phase'] == 'feedback':
                if not b._is_feedback_fresh(): return
                response = b._set_hardware_enabled_without_reset(SetBool.Request(data=True), SetBool.Response())
                if not response.success: raise ValueError(response.message)
                j['phase'] = 'enable'
            elif j['phase'] == 'enable':
                if not b._hardware_enabled: raise ValueError(b._last_event)
                if not b._hardware_ready: return
                if j.get('prepare_only'):
                    if not b._is_feedback_fresh() or not self.settled():return
                    self.job=None
                    self.completed.emit(j['token'],True,'保持式准备完成，等待导航接受后调姿')
                    return
                if 'posture' not in j or 'Joint_Ankle' in j['posture']:
                    response = b._set_leg_waist_enabled(SetBool.Request(data=True), SetBool.Response())
                    if not response.success: raise ValueError(response.message)
                if 'posture' in j:
                    from control_center_joint_control.joint_definitions import group_joint_names
                    wanted=j['posture'];groups=[]
                    if 'Joint_Ankle' in wanted:
                        goals={n:float(b._current[n]) for n in group_joint_names('leg_waist')}
                        goals.update({n:v for n,v in wanted.items() if n in goals})
                        b._on_apply(String(data=json.dumps(dict(group='leg_waist',targets_deg=goals,speed_scale=1.,motion_profile='body_height'))))
                        groups.append('leg_waist')
                    if 'Joint_Neck_Pitch' in wanted:
                        b._on_apply(String(data=json.dumps(dict(group='neck',single_joint='Joint_Neck_Pitch',
                            targets_deg={'Joint_Neck_Pitch':wanted['Joint_Neck_Pitch']},speed_scale=1.))))
                        groups.append('neck')
                    if not b._hardware_ready or any(g not in b._active_goals for g in groups):raise ValueError('调姿目标未被控制器接收：'+b._last_event)
                    j.update(phase='posture',started=time.monotonic(),stable=None)
                    return
                if j.get('preserve_grippers_at_entry'):
                    j['record']=entry_gripper_hold(j['record'],b._current)
                    b._event('抓取动作接续：首帧保持当前持物开度；后续帧按示教开合')
                b._on_action_command(String(data=json.dumps(dict(command='start', name=j['token'], action=j['record']))))
                if not b._action_state: raise ValueError(b._last_event)
                j['phase'] = 'running'; j['started'] = time.monotonic()
            elif j['phase']=='posture':
                if not b._hardware_ready or not b._is_feedback_fresh():raise ValueError('身体/头部调姿反馈或控制权失效')
                if b._active_goals or any(abs(b._current[n]-v)>2. for n,v in j['posture'].items()) or not self.settled():j['stable']=None
                elif j['stable'] is None:j['stable']=time.monotonic()
                elif time.monotonic()-j['stable']>=.5:
                    self.job=None;self.panel.setEnabled(True);self.release()
                    self.completed.emit(j['token'],True,'身体和头部已到位并停稳')
            elif j['phase'] == 'running':
                if b.outcome and b.outcome[0] == j['token']:
                    _, ok, text = b.outcome
                    if not ok: raise ValueError(text)
                    j.update(phase='settle', started=time.monotonic(), stable=None)
                elif not b._hardware_ready or not b._is_feedback_fresh(): raise ValueError('动作反馈或控制权失效')
            elif j['phase'] == 'settle':
                if not b._hardware_ready or not b._is_feedback_fresh(): raise ValueError('动作结束反馈失效')
                if not self.settled(): j['stable'] = None
                elif j['stable'] is None: j['stable'] = time.monotonic()
                elif time.monotonic()-j['stable'] >= .3:
                    self.job = None; self.panel.setEnabled(True); self.release()
                    self.completed.emit(j['token'], True, '动作已完成并停稳')
                if self.job and time.monotonic()-j['started'] > 15: raise ValueError('动作结束后未确认停稳')
        except Exception as exc:
            token = j['token']; self.stop(); self.release(); self.completed.emit(token, False, str(exc))

    def shutdown(self):
        self.stop(); self.release(); self.timer.stop()
        if self.panel: self.panel.spin_timer.stop()
        if self.node: self.node.destroy_node(); self.node = None
