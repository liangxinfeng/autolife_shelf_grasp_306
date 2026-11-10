import pytest
from autolife_shelf_grasp_306.pose_handoff import await_loaded_pose


def exercise(fresh_after=1,drift=0.,frozen=False):
    count=[0];reports=[];source={'Joint_Right_Gripper':160.,'Joint_Right_Elbow':-120.}
    def poll():count[0]+=1
    def now():return 10.+count[0]*.025
    def stamp():return 9.5 if frozen or count[0]<fresh_after else now()
    def snapshot():return {**source,'Joint_Right_Elbow':source['Joint_Right_Elbow']+drift}
    value=await_loaded_pose(snapshot,poll,stamp,source,'使能期间','place',now,reports.append,arm='right')
    return value,count[0],reports


def test_drains_callback_before_checking_enable_transition():
    q,n,records=exercise()
    assert q['Joint_Right_Gripper']==160 and n>=9
    assert all(r['feedback_age_sec']<=.3 for r in records)


def test_short_stale_gap_waits_for_new_frames():
    q,n,records=exercise(fresh_after=12)
    assert n>=20 and len(records)<n/2


def test_missing_feedback_never_authorizes_movement():
    with pytest.raises(ValueError,match='新鲜持物反馈超时'):exercise(frozen=True)


def test_real_motion_not_hidden_by_refresh_wait():
    with pytest.raises(ValueError,match='姿态改变'):exercise(fresh_after=8,drift=.6)


def test_same_fresh_frame_does_not_count_repeatedly():
    count=[0];source={'Joint_Left_Gripper':160.}
    def poll():count[0]+=1
    with pytest.raises(ValueError,match='反馈'):
        await_loaded_pose(lambda:source,poll,lambda:10.,source,'切换','place',
            lambda:10.+count[0]*.025,lambda value:None)


def test_reversed_timestamp_never_passes():
    count=[0];source={'Joint_Left_Gripper':160.}
    def poll():count[0]+=1
    with pytest.raises(ValueError,match='新鲜持物反馈超时'):
        await_loaded_pose(lambda:source,poll,lambda:100.,source,'切换','place',
            lambda:10.+count[0]*.025,lambda value:None)
