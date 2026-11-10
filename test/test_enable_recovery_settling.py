from types import SimpleNamespace as NS
from test_teaching_enable import fixture,lost,tick
from control_center_joint_control.joint_definitions import BODY_JOINTS
from control_center_joint_control.joint_bridge import JointBridge
from autolife_shelf_grasp_306.teaching_enable import wait_for_enable_feedback


def strict_position(b):
    original=b.get_parameter
    b.get_parameter=lambda name:NS(value=.3) if name=='hardware_reset_position_tolerance_deg' else original(name)


def test_small_online_error_waits_without_rebasing_or_publishing(monkeypatch):
    b,clock,events=fixture(monkeypatch);strict_position(b)
    publications=[];b._publish_hardware_hold_pose=lambda p:publications.append(p['hold_targets'].copy())
    source=b._hardware_prepare['hold_targets'].copy()
    lost(b,True);tick(b,clock);lost(b,False)
    name=BODY_JOINTS[0]
    for error in (.59,.4,.35,.31):
        b._current[name]=error
        assert tick(b,clock) and not b.reason and not b._hardware_ready
        assert not publications
    b._current[name]=.1
    for _ in range(15):
        if not tick(b,clock):JointBridge._advance_hardware_prepare(b)
        if b._hardware_ready:break
    assert b._hardware_ready and not b.reason
    assert all(p==source for p in publications)
    assert any(e[0]=='enable_pose_settling' for e in events)


def test_temporary_feedback_gap_waits_then_recovers(monkeypatch):
    b,clock,_=fixture(monkeypatch)
    b._is_feedback_fresh=lambda:False
    for _ in range(5):assert tick(b,clock) and not b.reason
    b._is_feedback_fresh=lambda:True
    for _ in range(15):handled=tick(b,clock)
    assert not handled and not b.reason


def test_large_pose_change_still_stops_without_retarget(monkeypatch):
    b,clock,_=fixture(monkeypatch);lost(b,True);tick(b,clock);lost(b,False)
    b._current[BODY_JOINTS[0]]=10.01
    assert tick(b,clock) and '大幅变化' in b.reason


def test_persistent_small_error_times_out_with_joint_and_error(monkeypatch):
    b,clock,_=fixture(monkeypatch);strict_position(b)
    lost(b,True);tick(b,clock);lost(b,False)
    name=BODY_JOINTS[0];b._current[name]=.35
    for _ in range(70):
        tick(b,clock)
        if b.reason:break
    assert name in b.reason and '0.35' in b.reason and not b._hardware_ready


def test_actual_failure_trace_reaches_original_pose_without_reenable(monkeypatch):
    from enable_recovery_trace import TRACE
    from autolife_handeye_calibration_306.core import BODY,NECK,ARMS
    b,clock,_=fixture(monkeypatch);strict_position(b)
    groups={'leg_waist_joint_state':BODY,'neck_joint_state':NECK,
            'left_arm_joint_state':ARMS['left'],'right_arm_joint_state':ARMS['right']}
    def update(row):
        clock[0]=row['wall'];b._feedback_monotonic=clock[0]
        for key,names in groups.items():
            v=row['raw'][key]
            for i,n in enumerate(names):
                b._current[n]=v['position'][i]
                b._details[n]=dict(speed=v['speed'][i],communication_lost=v['communication_lost'][i])
    update(TRACE[0]);source=b._current.copy()
    b._hardware_prepare.update(started=clock[0],next_publish=clock[0],hold_targets=source.copy())
    publications=[];b._publish_hardware_hold_pose=lambda p:publications.append(p['hold_targets'].copy())
    started=False
    for row in TRACE[1:]:
        update(row)
        if b._communication_lost_joints(BODY_JOINTS):started=True
        if not started:continue
        if not wait_for_enable_feedback(b):JointBridge._advance_hardware_prepare(b)
        assert not b.reason
        if b._hardware_ready:break
    assert started and b._hardware_ready
    assert all(p==source for p in publications)
