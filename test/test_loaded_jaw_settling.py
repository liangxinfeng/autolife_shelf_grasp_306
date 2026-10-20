import pytest
from autolife_shelf_grasp_306.pose_handoff import LoadedJawSettling,await_loaded_pose


def test_window_requires_fresh_span_not_same_frame():
    w=LoadedJawSettling()
    for _ in range(30):assert not w.add(1.,1.,162.)
    for t in (1.04,1.08,1.12,1.16):assert not w.add(t,t,162.)
    assert w.add(1.21,1.21,162.1)


@pytest.mark.parametrize('stamp,now,angle',[(1.,1.31,160.),(2.,1.,160.),(1.,1.,float('nan'))])
def test_stale_or_nonfinite(stamp,now,angle):
    with pytest.raises(ValueError):LoadedJawSettling().add(stamp,now,angle)


def scenario(jaw,arm=lambda i:0.,stale=False):
    index=[0];reports=[]
    def now():return 10.+index[0]*.025
    def poll():index[0]+=1
    def snap():return {'Joint_Left_Gripper':jaw(index[0]),'Joint_Left_Elbow':arm(index[0])}
    return await_loaded_pose(snap,poll,lambda:10. if stale else now(),
        {'Joint_Left_Gripper':165.08545916773892,'Joint_Left_Elbow':0.},
        'SYNC切换期间','place',now,reports.append),index[0]


def test_recorded_2_34_deg_offset_needs_stability():
    q,n=scenario(lambda i:162.74676671031168)
    assert q['Joint_Left_Gripper']==162.74676671031168 and n>=8


def test_transient_then_settle():
    q,n=scenario(lambda i:165-i*.22 if i<12 else 162.4)
    assert n>=16 and q['Joint_Left_Gripper']==162.4


def test_oscillation_rejected_even_inside_angle_bound():
    with pytest.raises(ValueError,match='未稳定'):scenario(lambda i:163.+(i%2)*1.2)


def test_arm_drift_not_accepted_as_jaw_settling():
    with pytest.raises(ValueError,match='姿态改变'):scenario(lambda i:163.,lambda i:.51)


def test_cumulative_drift_cannot_rebase_reference():
    with pytest.raises(ValueError):scenario(lambda i:165.-i*.5)


def test_frozen_feedback_cannot_pass():
    with pytest.raises(ValueError,match='反馈'):scenario(lambda i:163.,stale=True)


def test_final_receipt_after_sync_exit_in_executor():
    from pathlib import Path
    text=(Path(__file__).parents[1]/'autolife_shelf_grasp_306/execution.py').read_text()
    assert text.index('handoff_ok=session.close()')<text.index("record['q']=await_loaded_pose")<text.index("atomic_json(run/'held_object.json',record)")
    assert "'hold_feedback_confirmed'" in text
