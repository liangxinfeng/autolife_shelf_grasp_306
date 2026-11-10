from types import SimpleNamespace as NS
import pytest
from PyQt5 import QtWidgets as Q
from autolife_shelf_grasp_306.action_studio import ActionExecutor
from autolife_shelf_grasp_306.teaching_motion import compile_action
from test_teaching import pose,draft


@pytest.mark.parametrize('exit_ok',[True,False])
@pytest.mark.parametrize('keep',[True,False])
def test_actual_executor_gates_playback_and_reports_exit(monkeypatch,tmp_path,exit_ok,keep):
    app=Q.QApplication.instance() or Q.QApplication([])
    import rclpy
    monkeypatch.setattr(rclpy,'spin_once',lambda *a,**k:None)
    clock=[100.];monkeypatch.setattr('time.monotonic',lambda:clock[0])
    sessions=[];started=[];receipts=[]
    class Session:
        def __init__(self,b):self.ready=False;self.closed=False;self.active=True;self.verified=False;self.lower_enabled=set();sessions.append(self)
        def begin(self):pass
        def tick(self):pass
        def poll(self):self.verified=self.ready;return self.ready
        def close(self):self.closed=True;return exit_ok
    monkeypatch.setattr('autolife_shelf_grasp_306.pose_manual_session.ManualHoldSession',Session)
    q=pose();b=NS(_current=q,_details={n:dict(speed=0.) for n in q},_is_feedback_fresh=lambda:True,
        _action_state=None,_active_goals={},_hardware_prepare=None,_hardware_enabled=True,_hardware_ready=True,
        outcome=None,_last_event='',_instance_lock_handle=None,_event=lambda *a:None,destroy_node=lambda:None,
        _set_hardware_enabled_without_reset=lambda *a:(_ for _ in ()).throw(AssertionError('must never re-enable separately')),_set_leg_waist_enabled=lambda *a:NS(success=True))
    def command(msg):started.append(msg.data);b._action_state={'name':'integration'}
    b._on_action_command=command
    b._publish_hardware_hold_pose=lambda *a:None;b._joint_enable_pub=None;b._publish_vendor=lambda *a:None
    b._lock=lambda reason:None
    e=ActionExecutor(NS(root=tmp_path,run=tmp_path));e.timer.stop();e.bridge=b;e.ensure=lambda:b
    e.panel=NS(setEnabled=lambda *a:None,stop_action_preview=lambda **k:None,preview_timer=NS(stop=lambda:None),
        _pending_live_groups=set(),_pending_live_joint_names=set(),_pending_preview_groups=set(),
        _set_checked_silently=lambda *a:None,live_toggle=None)
    e.completed.connect(lambda *args:receipts.append(args))
    e.execute('integration',compile_action(draft([pose(),pose(Joint_Right_Elbow=-5)])),keep_session=keep)
    e.tick();assert e.job['phase']=='sync' and not started
    for _ in range(3):e.tick()
    assert not started
    sessions[0].ready=True;e.tick();assert started and e.job['phase']=='running'
    b._action_state=None;b.outcome=('integration',True,'done')
    e.tick();e.tick();clock[0]+=.31;e.tick()
    if keep:
        assert not sessions[0].closed and e.bridge is b and receipts[-1][1]
        e.execute('second',compile_action(draft([pose()])),keep_session=True);e.tick();e.tick()
        assert len(sessions)==1 and len(started)==2  # 连续两次执行只取得一次保持会话。
        e.job=None;b._action_state=None;e.release()
    else:assert sessions[0].closed and e.bridge is None and receipts[-1][1] is exit_ok
    e.deleteLater()
