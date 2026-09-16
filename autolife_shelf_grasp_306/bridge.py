"""Grasp-only bridge: distinguish settled streaming goals from stalled motors."""
import time
from autolife_handeye_calibration_306.bridge import CalibrationBridge


class GraspBridge(CalibrationBridge):
    last_lock_reason = ''

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
