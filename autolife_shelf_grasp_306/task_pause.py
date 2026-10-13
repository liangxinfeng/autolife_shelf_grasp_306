"""Cooperative position hold. Keep feedback/ownership watchdogs alive."""
import time
import math
from pathlib import Path
from .core import atomic_json


class BridgePause:
    def __init__(self, node, run, clock=None):
        self.node=node;self.run=Path(run);self.clock=clock or time.monotonic
        self.started=None;self.total=0.;self.hold=None;self.generation=0

    def now(self):
        now=self.clock()
        return now-self.total-(now-self.started if self.started is not None else 0.)

    def cancel(self):
        self.started=None;self.hold=None

    def tick(self):
        n=self.node;now=self.clock();wanted=(self.run/'task_pause').exists()
        if not wanted and self.started is None:return False
        # Hold is only entered after hold-only enable has actually completed.
        if not n._hardware_ready:
            return False
        if not n._is_feedback_fresh() or n._communication_lost_joints(n._current) or not all(math.isfinite(v) for v in n._current.values()):
            n._lock('暂停期间关节反馈失效');return True
        if hasattr(n,'last_conflict_check') and now-n.last_conflict_check>1:
            from autolife_handeye_calibration_306.bridge import competing_controller
            n.last_conflict_check=now;n.conflict=competing_controller()
            if n.conflict:n._lock(n.conflict);return True
        if self.started is None:
            self.started=now;self.hold=dict(n._current)
            from control_center_joint_control.joint_definitions import GROUPS
            payload={GROUPS[g]['command_key']:[self.hold[name] for name,*_ in GROUPS[g]['joints']]
                     for g in ('left_arm','right_arm','neck','leg_waist')}
            n._publish_vendor(n._whole_body_pub,payload)
            n._publish_vendor(n._gripper_pub,dict(left_gripper_target_joints_position=[self.hold['Joint_Left_Gripper']],
                right_gripper_target_joints_position=[self.hold['Joint_Right_Gripper']]))
            atomic_json(self.run/'pause_motion_status.json',dict(paused=True,time=time.time()))
        if max(abs(n._current[k]-v) for k,v in self.hold.items())>3.:
            n._lock('暂停保持姿态发生明显变化，请重新规划');return True
        n._last_operator_activity=now
        if wanted:return True
        elapsed=now-self.started;self.total+=elapsed;self.started=None;self.generation+=1
        state=getattr(n,'_action_state',None)
        if state:
            for key in ('frame_started','frame_deadline','last_progress_at','hold_started'):
                if key in state:state[key]+=elapsed
            state['commanded_values']={k:n._current[k] for k in state['commanded_values']}
        for goal in n._active_goals.values():
            goal['commands']={k:n._current[k] for k in goal['commands']}
        if hasattr(n,'progress_watch'):
            n.progress_watch={g:(now,dict(n._current)) for g in n.progress_watch}
        atomic_json(self.run/'pause_motion_status.json',dict(paused=False,time=time.time()))
        self.hold=None
        return False


def archive_held(run):
    """Only call at a newly authorized empty-gripper task boundary."""
    run=Path(run);p=run/'held_object.json'
    if not p.exists():return
    import uuid
    directory=run/'held_history';directory.mkdir(exist_ok=True)
    # Atomic rename preserves even an old or malformed receipt without parsing.
    p.replace(directory/(uuid.uuid4().hex+'.json'))


def defer_call(owner, method, *args):
    auto=getattr(owner,'automatic',None)
    if auto and auto.active and auto.paused:
        cursor=owner.workflow.cursor;token=cursor.token if cursor and cursor.active else None
        auto.deferred[(method.__name__,token)]=(method,args,token)
        return True
    return False
