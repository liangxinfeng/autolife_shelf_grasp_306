"""Exercise the real grasp override without constructing ROS/motor objects."""
from types import SimpleNamespace
from autolife_shelf_grasp_306.bridge import GraspBridge,CalibrationBridge
import autolife_shelf_grasp_306.bridge as module


def test_hold_enable_keeps_measured_pose_without_system_reset(monkeypatch):
    import control_center_joint_control.joint_bridge as factory
    import autolife_handeye_calibration_306.bridge as calibration
    published=[];node=GraspBridge.__new__(GraspBridge)
    node._current={n:0. for n in factory.BODY_JOINTS};node._controller_target=dict(node._current)
    node._current['Joint_Neck_Pitch']=-46.13
    node._details={};node._is_feedback_fresh=lambda:True;node._communication_lost_joints=lambda _:[]
    node._event=lambda *args:None
    node._control_reset_pub=SimpleNamespace(get_subscription_count=lambda:1,publish=lambda msg:published.append(msg.data))
    monkeypatch.setattr(calibration,'competing_controller',lambda:'')
    response=node.unlock()
    assert response.success and not node._hardware_ready
    assert node._hardware_unlock_mode=='without_reset' and not node._leg_waist_enabled
    assert node._hardware_prepare['stage']=='check_modes'
    assert node._hardware_prepare['hold_targets']==node._current and not published
    assert not hasattr(node,'unlock_with_reset') and not node.gui_call
    monkeypatch.setattr(calibration,'competing_controller',lambda:'VR正在运行')
    assert not node.unlock().success and not published


def test_streamed_settled_group_is_not_stalled(monkeypatch):
    now=[0.];locks=[]
    node=GraspBridge.__new__(GraspBridge)
    node._current={'a':0.};node._active_goals={};node.progress_watch={}
    node.get_parameter=lambda _:SimpleNamespace(value=1.)
    # Faithful original watchdog ordering: watch BEFORE parent arrival cleanup.
    def old_tick(n):
        for group,(stamp,_) in list(n.progress_watch.items()):
            if group not in n._active_goals:n.progress_watch.pop(group);continue
            if now[0]-stamp>5:locks.append(group)
        for group,rec in list(n._active_goals.items()):
            if abs(n._current['a']-rec['targets']['a'])<=1:n._active_goals.pop(group)
    monkeypatch.setattr(CalibrationBridge,'_advance_motion',old_tick)
    def reproduce(tick):
        node.progress_watch={};node._active_goals={};locks.clear()
        for i in range(81):
            now[0]=i*.1
            node._active_goals['neck']={'targets':{'a':0.}}
            if 'neck' not in node.progress_watch:node.progress_watch['neck']=(now[0],{'a':0.})
            tick(node)
        return list(locks)
    assert reproduce(old_tick) # reproduce the actual historical false failure
    assert not reproduce(GraspBridge._advance_motion)
    # Outside arrival tolerance, original stalled-motor detection still runs.
    node._active_goals={'left_arm':{'targets':{'a':3.}}}
    node.progress_watch={'left_arm':(0.,{'a':0.})};locks.clear()
    node._advance_motion()
    assert locks==['left_arm']


def test_stall_keeps_joint_targets_commands_and_feedback(monkeypatch):
    node=GraspBridge.__new__(GraspBridge)
    node._current={'shoulder':17.06};node._active_goals={'left_arm':{'targets':{'shoulder':18.},'commands':{'shoulder':18.}}}
    node.progress_watch={'left_arm':(0.,{'shoulder':17.06})}
    monkeypatch.setattr(module.time,'monotonic',lambda:6.)
    monkeypatch.setattr(CalibrationBridge,'_lock',lambda self,reason:self.progress_watch.clear())
    node._lock('实测关节5秒无跟随')
    assert 'shoulder' in node.last_lock_reason and '0.94' in node.last_lock_reason
    assert node.last_stop_details['groups']['left_arm']['measured_deg']=={'shoulder':17.06}
    assert node.last_stop_details['groups']['left_arm']['commands_deg']=={'shoulder':18.}
