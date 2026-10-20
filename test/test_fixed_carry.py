"""Offline fixed endpoint, collision, height and protocol regression tests."""
import copy
import time
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from autolife_shelf_grasp_306.carry import carry_goal,validate_carry_plan
from autolife_shelf_grasp_306.planning import Planner,IKFailure,CollisionFailure


def planner():
    p=Planner.__new__(Planner);p.names=['arm']
    p.cfg={'carry_pose_deg':[10.],'fixed_carry_enabled':True,'up_torso':[0,0,1],
           'max_joint_speed_deg_s':120,'carry_max_tilt_deg':45.}
    p.pose=lambda q,tcp:(np.array([0.,0.,.3+q['arm']/100]),np.eye(3))
    p.check=lambda *args:None
    return p


def test_only_holding_arm_changes_and_all_states_checked():
    p=planner();source={'arm':0.,'neck':-20.,'other_arm':33.,'gripper':270.};seen=[]
    p.check=lambda q,tcp,carrying,product:seen.append((q.copy(),carrying))
    segment,count=p._carry_segment(source,np.zeros(3),{},np.eye(3),time.monotonic())
    assert segment['phase']=='carry' and segment['waypoints'][-1]=={**source,'arm':10.}
    assert source['arm']==0 and count==len(seen)>20
    assert all(c and q['gripper']==270 and q['neck']==-20 and q['other_arm']==33 for q,c in seen)


@pytest.mark.parametrize('failure',['collision','dip','tilt'])
def test_entire_carry_curve_is_checked(failure):
    p=planner()
    if failure=='collision':
        def check(q,*args):
            if 4<q['arm']<6:raise CollisionFailure('携物碰撞')
        p.check=check
    else:
        def pose(q,tcp):
            mid=4<q['arm']<6
            return (np.array([0.,0.,.1 if mid and failure=='dip' else .4]),
                    Rotation.from_euler('x',60 if mid and failure=='tilt' else 0,degrees=True).as_matrix())
        p.pose=pose
    with pytest.raises((IKFailure,CollisionFailure)):
        p._carry_segment({'arm':0.},np.zeros(3),{},np.eye(3),time.monotonic())


def test_all_sources_share_one_goal():
    p=planner()
    for start in [-20,0,30]:
        assert carry_goal(p.cfg,{'arm':start,'jaw':100},p.names)=={'arm':10.,'jaw':100}


def test_protocol_rejects_old_endpoint_and_discontinuous_paths():
    cfg=planner().cfg
    plan={'config':cfg,'source_q':{'arm':0},'carry_protocol_version':1,
          'segments':[{'phase':phase,'waypoints':[{'arm':0},{'arm':0}]} for phase in ['pregrasp','grasp','lift','withdraw','carry']]}
    plan['segments'][-1]['waypoints'][-1]['arm']=10
    validate_carry_plan(plan,['arm'])
    for mutation in ['missing','endpoint','continuity']:
        bad=copy.deepcopy(plan)
        if mutation=='missing':bad['segments'].pop()
        if mutation=='endpoint':bad['segments'][-1]['waypoints'][-1]['arm']=9
        if mutation=='continuity':bad['segments'][-1]['waypoints'][0]['arm']=2
        with pytest.raises(ValueError):validate_carry_plan(bad,['arm'])


@pytest.mark.parametrize('values',[[0,1],[float('nan')]])
def test_invalid_configuration(values):
    with pytest.raises(ValueError):carry_goal({'carry_pose_deg':values},{'arm':0},['arm'])
