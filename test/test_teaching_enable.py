from types import SimpleNamespace as NS
import pytest
from autolife_shelf_grasp_306.teaching_enable import wait_for_enable_feedback
from control_center_joint_control.joint_definitions import BODY_JOINTS
from control_center_joint_control.joint_bridge import JointBridge


def fixture(monkeypatch):
    clock=[100.];monkeypatch.setattr('time.monotonic',lambda:clock[0])
    q={n:0. for n in BODY_JOINTS};events=[]
    b=NS(_hardware_enabled=True,_hardware_ready=False,_hardware_unlock_mode='without_reset',
         _hardware_prepare=dict(stage='verify_hold',started=99.,next_publish=100.,stable_since=None,hold_targets=q.copy()),
         _current=q.copy(),_details={n:dict(speed=0.,communication_lost=False) for n in q},
         _feedback_monotonic=100.,_is_feedback_fresh=lambda:True,reason='',
         get_parameter=lambda n:NS(value=dict(hardware_prepare_timeout_sec=30.,hardware_reset_position_tolerance_deg=1.,hardware_reset_speed_tolerance_deg_s=2.,hardware_reset_stable_sec=.2)[n]),
         _event=lambda *args:events.append(args),_record_action_event=lambda *args:events.append(args))
    b._communication_lost_joints=lambda names:JointBridge._communication_lost_joints(b,names)
    def lock(reason):b.reason=reason;b._hardware_enabled=False;b._hardware_prepare=None
    b._lock=lock
    return b,clock,events


def lost(b,value):
    for n in list(BODY_JOINTS)[:12]:b._details[n]['communication_lost']=value


def tick(b,clock,dt=.05):
    clock[0]+=dt;b._feedback_monotonic=clock[0]
    return wait_for_enable_feedback(b)


def test_healthy_enable_unchanged(monkeypatch):
    b,clock,_=fixture(monkeypatch)
    assert not wait_for_enable_feedback(b)
    assert not b._hardware_ready and not b.reason


def test_twelve_lost_then_stable_online_before_normal_verification(monkeypatch):
    b,clock,events=fixture(monkeypatch);lost(b,True)
    for _ in range(4):assert tick(b,clock)
    lost(b,False)
    for _ in range(8):assert tick(b,clock)
    for _ in range(3):handled=tick(b,clock)
    assert not handled and not b.reason
    assert not b._hardware_ready  # Only native verification may mark ready.
    assert any(e[0]=='enable_feedback_recovered' for e in events)
    assert all(v==0. for v in b._hardware_prepare['hold_targets'].values())


@pytest.mark.parametrize('fault',['persistent','stale','drift','speed','nan','frozen','overall_timeout'])
def test_unreliable_recovery_never_authorizes_playback(monkeypatch,fault):
    b,clock,_=fixture(monkeypatch);lost(b,True);assert tick(b,clock)
    lost(b,False);n=BODY_JOINTS[0]
    if fault=='persistent':lost(b,True)
    elif fault=='stale':b._is_feedback_fresh=lambda:False
    elif fault=='drift':b._current[n]=1.01
    elif fault=='speed':b._details[n]['speed']=2.01
    elif fault=='nan':b._current[n]=float('nan')
    elif fault=='overall_timeout':b._hardware_prepare['started']=60.
    for i in range(70):
        clock[0]+=.05
        if fault!='frozen':b._feedback_monotonic=clock[0]
        assert wait_for_enable_feedback(b)
        if b.reason:break
    assert b.reason and not b._hardware_ready and not b._hardware_enabled


def test_second_dropout_restarts_online_window(monkeypatch):
    b,clock,_=fixture(monkeypatch);lost(b,True);assert tick(b,clock);lost(b,False)
    for _ in range(12):tick(b,clock)
    assert b._hardware_prepare['enable_feedback_recovery']['recovered']
    lost(b,True);assert tick(b,clock);lost(b,False)
    for _ in range(8):assert tick(b,clock)
    for _ in range(3):handled=tick(b,clock)
    assert not handled


def test_transient_speed_waits_for_stability_instead_of_false_pose_error(monkeypatch):
    b,clock,_=fixture(monkeypatch);lost(b,True);assert tick(b,clock);lost(b,False)
    b._details[BODY_JOINTS[0]]['speed']=3.
    assert tick(b,clock) and not b.reason
    b._details[BODY_JOINTS[0]]['speed']=0.
    for _ in range(12):handled=tick(b,clock)
    assert not handled and not b.reason


def test_native_preparation_still_must_finish_and_wait_sends_nothing(monkeypatch):
    b,clock,_=fixture(monkeypatch);publications=[]
    b._publish_hardware_hold_pose=lambda p:publications.append(clock[0])
    lost(b,True)
    for _ in range(5):
        assert tick(b,clock)
        assert not publications and not b._hardware_ready
    lost(b,False);recovered_at=clock[0]
    for _ in range(25):
        if not tick(b,clock):JointBridge._advance_hardware_prepare(b)
        if b._hardware_ready:break
    assert b._hardware_ready and not b.reason
    assert min(publications)-recovered_at>=.4


@pytest.mark.parametrize('mode',['reset','running','disabled'])
def test_only_nonreset_preparation_is_affected(monkeypatch,mode):
    b,clock,_=fixture(monkeypatch);lost(b,True)
    if mode=='reset':b._hardware_unlock_mode='reset'
    elif mode=='running':b._hardware_ready=True;b._hardware_prepare=None
    else:b._hardware_enabled=False
    assert not wait_for_enable_feedback(b)
