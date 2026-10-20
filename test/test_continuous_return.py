"""No hardware: continuous return contract and bounded stage speed policy."""
import copy,time
import numpy as np
import pytest
from autolife_shelf_grasp_306.carry import carry_goal,validate_carry_plan
from autolife_shelf_grasp_306.motion import feedback_lead,Follower,configure_grasp_path
from autolife_shelf_grasp_306.planning import Planner,IKFailure,CollisionFailure
from autolife_handeye_calibration_306.verification_trajectory import JointPath


def config():
    return dict(arm='left',fixed_carry_enabled=True,carry_return_mode='continuous_reset',
                initial_pose_deg={'left_arm':[10.]},grasp_transit_feedback_lead_deg=5.)


def test_one_reset_source_and_only_holding_arm_changes():
    cfg=config();source={'a':0.,'jaw':150.,'neck':-30.,'right_arm':-40.}
    assert carry_goal(cfg,source,['a'])=={**source,'a':10.}
    cfg['initial_pose_deg']['left_arm']=[20.]
    assert carry_goal(cfg,source,['a'])['a']==20.


def test_new_plan_requires_three_connected_stages_and_exact_reset_endpoint():
    plan={'config':config(),'source_q':{'a':0},'carry_protocol_version':2,'segments':[
        {'phase':'pregrasp','waypoints':[{'a':0},{'a':2}]},
        {'phase':'grasp','waypoints':[{'a':2},{'a':4}]},
        {'phase':'carry','waypoints':[{'a':4},{'a':6},{'a':10}]}]}
    validate_carry_plan(plan,['a'])
    for mutation in ['version','endpoint','discontinuous','old_stages']:
        bad=copy.deepcopy(plan)
        if mutation=='version':bad['carry_protocol_version']=1
        if mutation=='endpoint':bad['segments'][-1]['waypoints'][-1]['a']=9
        if mutation=='discontinuous':bad['segments'][-1]['waypoints'][0]['a']=3
        if mutation=='old_stages':bad['segments'].insert(2,{'phase':'lift'})
        with pytest.raises(ValueError):validate_carry_plan(bad,['a'])


def test_merged_curve_rechecked_not_just_clearance_vertices():
    p=Planner.__new__(Planner);p.names=['a'];p.cfg={**config(),'up_torso':[0,0,1]}
    p.pose=lambda q,tcp:(np.array([0,0,.4+q['a']*.01]),np.eye(3))
    seen=[];p.check=lambda q,*_:seen.append(q.copy())
    start={'a':0.,'jaw':150.};clearance=[start,{**start,'a':2.}]
    segment,count=p._carry_segment(start,None,{},np.eye(3),time.monotonic(),clearance)
    assert segment['phase']=='carry' and segment['waypoints']==[start,{**start,'a':10.}]
    assert segment['return_strategy']=='direct'
    assert count==len(seen)>90 and all(q['jaw']==150 for q in seen)
    def blocked(q,*_):
        if 5<q['a']<6:raise CollisionFailure('obstacle')
    p.check=blocked
    with pytest.raises(CollisionFailure):p._carry_segment(start,None,{},np.eye(3),time.monotonic(),clearance)


def test_lift_region_must_not_descend_even_if_final_reset_is_lower():
    p=Planner.__new__(Planner);p.names=['a'];p.cfg={**config(),'up_torso':[0,0,1]}
    p.check=lambda *args:None
    p.pose=lambda q,tcp:(np.array([0,0,.4-q['a']*.01]),np.eye(3))
    with pytest.raises(IKFailure,match='起瓶区域'):
        p._carry_segment({'a':0},None,{},np.eye(3),time.monotonic(),[{'a':0},{'a':2}])


def test_speed_scope_and_limits():
    cfg=config()
    assert feedback_lead(cfg,'carry')==5 and feedback_lead(cfg,'pregrasp')==5
    assert feedback_lead(cfg,'grasp')==3
    assert feedback_lead(cfg,'carry','place')==3 and feedback_lead(cfg,'carry','observe')==3
    for value in [0,5.1,float('nan')]:
        with pytest.raises(ValueError):feedback_lead({**cfg,'grasp_transit_feedback_lead_deg':value},'carry')
    f=Follower(configure_grasp_path(JointPath(['a'],[{'a':0},{'a':90}]),{}),0,5)
    for i in range(1,301):
        command=f.step({'a':0.},i/60)
        if command:assert command['a']<=5.+1e-6
    assert not f.final and f.progress<=5.+1e-6
    with pytest.raises(ValueError,match='偏离'):f.step({'a':-3.},6.)
