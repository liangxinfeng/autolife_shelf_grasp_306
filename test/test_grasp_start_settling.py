import pytest
from autolife_shelf_grasp_306.pose_handoff import await_start_pose
from autolife_shelf_grasp_306.feedback_state import FeedbackNotReady

SOURCE={'Joint_Right_Shoulder_Inner':43.68780517578125,'Joint_Right_Gripper':12.5}


def run(values,*,frozen=False,stale=False,missing_until=0,cancel_at=0):
    count=[0];reports=[]
    def poll():
        count[0]+=1
        if count[0]==cancel_at:raise ValueError('用户停止')
    def clock():return 10+count[0]*.025
    def stamp():return 9. if stale else 10.025 if frozen else clock()
    def snapshot():
        if count[0]<=missing_until:raise FeedbackNotReady('offline',reason='offline',joints=['Joint_Right_Shoulder_Inner'])
        return {**SOURCE,'Joint_Right_Shoulder_Inner':SOURCE['Joint_Right_Shoulder_Inner']+values[min(count[0]-1,len(values)-1)]}
    result=await_start_pose(snapshot,poll,stamp,SOURCE,'使能期间',clock,reports.append,arm='right')
    return result,count[0],reports


def test_recorded_point_seven_four_transition_can_settle_without_rebasing():
    q,count,reports=run([-.7360992431640625,-.65,-.51,-.38,-.2]+[-.1]*12)
    assert count>=10 and abs(q['Joint_Right_Shoulder_Inner']-SOURCE['Joint_Right_Shoulder_Inner'])<.5
    assert any(r.get('waiting_settle') and not r['ok'] for r in reports)
    assert reports[-1]['ok'] and reports[-1]['stable']
    assert SOURCE['Joint_Right_Shoulder_Inner']==43.68780517578125


def test_healthy_start_adds_only_short_stable_window():
    _,count,reports=run([0.])
    assert 7<=count<=9 and reports[-1]['stable']


def test_persistent_small_offset_is_not_accepted_as_new_origin():
    with pytest.raises(ValueError,match='未在准备期限内稳定'):run([-.74])


def test_large_displacement_stops_immediately():
    with pytest.raises(ValueError,match='姿态改变'):run([4.])


def test_nonfinite_data_never_settles():
    with pytest.raises(ValueError,match='无效'):run([float('nan')])


@pytest.mark.parametrize('options',[dict(frozen=True),dict(stale=True)])
def test_old_or_repeated_feedback_cannot_count_as_stable(options):
    with pytest.raises(ValueError,match='未在准备期限内稳定'):run([0.],**options)


def test_short_offline_gap_then_fresh_stability():
    _,count,reports=run([0.],missing_until=8)
    assert count>=15 and reports[-1]['stable']


def test_persistent_offline_and_cancel_are_not_ignored():
    with pytest.raises(ValueError,match='未在准备期限内稳定'):run([0.],missing_until=1000)
    with pytest.raises(ValueError,match='用户停止'):run([-.74],cancel_at=2)


def test_changing_pose_inside_position_tolerance_is_not_stable():
    with pytest.raises(ValueError,match='未在准备期限内稳定'):run([.4,-.4]*100)


def test_real_bridge_does_not_report_ready_at_recorded_shoulder_error(monkeypatch):
    from types import SimpleNamespace as NS
    from control_center_joint_control.joint_bridge import JointBridge
    from control_center_joint_control.joint_definitions import BODY_JOINTS
    clock=[100.];monkeypatch.setattr('time.monotonic',lambda:clock[0])
    q={n:0. for n in BODY_JOINTS};q['Joint_Right_Shoulder_Inner']=-.7360992431640625
    params=dict(hardware_prepare_timeout_sec=30.,hardware_reset_position_tolerance_deg=.3,
                hardware_reset_speed_tolerance_deg_s=2.,hardware_reset_stable_sec=.2)
    b=NS(_hardware_enabled=True,_hardware_ready=False,_current=q,
         _details={n:dict(speed=0.) for n in q},
         _hardware_prepare=dict(started=99.,stage='verify_hold',next_publish=100.,
             stable_since=99.,hold_targets={n:0. for n in q}),
         get_parameter=lambda n:NS(value=params[n]),_communication_lost_joints=lambda names:[],
         _publish_hardware_hold_pose=lambda prepare:None,_event=lambda *a:None)
    JointBridge._advance_hardware_prepare(b)
    assert not b._hardware_ready and b._hardware_prepare['stable_since'] is None
    q['Joint_Right_Shoulder_Inner']=-.1
    clock[0]=100.21;JointBridge._advance_hardware_prepare(b)
    assert not b._hardware_ready
    clock[0]=100.43;JointBridge._advance_hardware_prepare(b)
    assert b._hardware_ready and b._hardware_prepare is None
