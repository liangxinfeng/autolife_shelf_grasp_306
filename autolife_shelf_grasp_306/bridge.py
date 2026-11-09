"""Grasp-only bridge: distinguish settled streaming goals from stalled motors."""
import time
from autolife_handeye_calibration_306.bridge import CalibrationBridge


class GraspBridge(CalibrationBridge):
    last_lock_reason = ''

    def _advance_hardware_prepare(self):
        # 抓取、复位与示教统一使用同一准备握手；短暂使能回包变化时不发动作。
        from .teaching_enable import wait_for_enable_feedback
        if wait_for_enable_feedback(self):return
        super()._advance_hardware_prepare()

    def _record_action_event(self,kind,reason=''):
        # 普通抓取桥没有动作编辑器上下文，仍记录使能阶段的逐轴证据。
        from control_center_joint_control.joint_definitions import BODY_JOINTS
        events=getattr(self,'enable_events',[])
        events.append(dict(time=time.time(),event=kind,reason=str(reason),
            lost_joints=self._communication_lost_joints(BODY_JOINTS)))
        self.enable_events=events[-32:]

    def enable_planned_waist(self,plan):
        """仅普通抓取执行器调用；不开放标定工具的腰腿服务入口。

        使用正式关节桥检查保持使能状态。已检查轨迹限定膝踝固定、
        另一臂和头部固定，仅允许所选手臂及规划中的腰部参与。
        """
        if (plan.get('execution_backend')!='vendor_trajectory_v1' or
                plan.get('kind') in ('prepare','place','observe')):
            raise ValueError('腰部抓取入口只接受已检查的原厂抓取轨迹')
        from .vendor_grasp_trajectory import validate_blocks
        validate_blocks(plan)
        from control_center_joint_control.joint_bridge import JointBridge
        from std_srvs.srv import SetBool
        # 显式调用正式接口；self._set_leg_waist_enabled属于标定限制入口。
        return JointBridge._set_leg_waist_enabled(self,SetBool.Request(data=True),SetBool.Response())

    def _event(self,message,level='info'):
        # Streaming targets arrive at 60Hz. Keep status but avoid synchronous
        # per-frame logging interfering with the control loop.
        if level=='info' and message.startswith('已接收“'):
            now=time.monotonic()
            if now-getattr(self,'_last_target_log',-float('inf'))<1.:
                self._last_event=message;self._last_event_level=level;return
            self._last_target_log=now
        super()._event(message,level)

    def _advance_motion(self):
        if getattr(self,'task_pause',None) and self.task_pause.tick():return
        # CalibrationBridge checks its watchdog BEFORE the parent's arrival
        # cleanup. Streaming can recreate a settled goal between those ticks,
        # leaving its old watchdog alive indefinitely. No motion is expected
        # inside the existing arrival tolerance; genuinely stalled goals remain
        # subject to the original five-second watchdog, freshness and ownership.
        for group in list(self.progress_watch):
            record = self._active_goals.get(group)
            param = 'gripper_goal_tolerance_deg' if group == 'grippers' else 'goal_tolerance_deg'
            tolerance = float(self.get_parameter(param).value)
            if not record or all(abs(self._current[n]-v) <= tolerance
                                 for n,v in record['targets'].items()):
                self.progress_watch.pop(group, None)
        super()._advance_motion()

    def _lock(self, reason):
        if '无跟随' in reason:
            details=[]
            for group,(stamp,previous) in self.progress_watch.items():
                goal=self._active_goals.get(group,{}).get('targets',{})
                if goal and time.monotonic()-stamp>5:
                    error=max(abs(self._current[n]-v) for n,v in goal.items())
                    joint=max(goal,key=lambda n:abs(self._current[n]-goal[n]))
                    details.append(f'{joint} 实测{self._current[joint]:.2f}°/目标{goal[joint]:.2f}°，误差{error:.2f}°')
            if details:reason+='；'+', '.join(details)
            self.last_stop_details={'reason':reason,'time':time.monotonic(),
                'groups':{g:{'targets_deg':dict(r['targets']),
                    'commands_deg':dict(r.get('commands',{})),
                    'measured_deg':{n:self._current[n] for n in r['targets']}}
                    for g,r in self._active_goals.items()}}
        self.last_lock_reason=reason
        super()._lock(reason)
