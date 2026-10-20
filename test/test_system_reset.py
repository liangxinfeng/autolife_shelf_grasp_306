import copy,json
import pytest
from control_center_joint_control.joint_definitions import GROUPS
from autolife_shelf_grasp_306.system_reset import apply_system_reset
from autolife_shelf_grasp_306.carry import carry_goal

@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.setattr('socket.gethostname',lambda:'autolife-robot-300')
    pose={'left_arm':[47.,.4,-.1,140.,.2,-5.,0.], 'right_arm':[-47.,-.1,0.,-140.,0.,.1,-.1], 'neck':[0.,-30.,-2.6]}
    vendor={'LEFT_ARM_MOTOR_HOME_ANGLE':{'standing':[0.]*4+pose['left_arm']},
        'RIGHT_ARM_MOTOR_HOME_ANGLE':{'standing':[0.]*4+pose['right_arm']},
        'NECK_MOTOR_HOME_ANGLE':{'zero':[0.]*4+pose['neck']}}
    joints={j[0]:v for g,values in pose.items() for j,v in zip(GROUPS[g]['joints'],values)}
    state={'version':1,'robot_id':'300','mode':'custom','custom':{'joints_deg':joints},'original':{'configs':{'vendor':vendor}}}
    sp=tmp_path/'state.json';vp=tmp_path/'vendor.json'
    sp.write_text(json.dumps(state));vp.write_text(json.dumps(vendor))
    cfg={'robot_id':300,'arm':'left','reset_pose_source':'robot_system','initial_pose_deg':{},'carry_return_mode':'continuous_reset','gripper_open_deg':10.,'hardware_execution':False}
    return cfg,sp,vp,state,pose

def test_local_pose_shared_with_carry_without_mutation(setup):
    cfg,sp,vp,state,pose=setup;original=copy.deepcopy(cfg)
    result=apply_system_reset(cfg,sp,[vp])
    assert result['initial_pose_deg']==pose and cfg==original
    assert result['hardware_execution'] is False and result['gripper_open_deg']==10.
    assert result['system_reset_snapshot']['body_policy']=='keep_measured'
    names=[j[0] for j in GROUPS['left_arm']['joints']]
    assert list(carry_goal(result,{},names).values())==pose['left_arm']
    result['arm']='right';names=[j[0] for j in GROUPS['right_arm']['joints']]
    assert list(carry_goal(result,{},names).values())==pose['right_arm']

def test_original_mode_ignores_old_custom(setup):
    cfg,sp,vp,state,pose=setup;state['mode']='original';state['custom']['joints_deg']={}
    sp.write_text(json.dumps(state));assert apply_system_reset(cfg,sp,[vp])['initial_pose_deg']==pose

@pytest.mark.parametrize('robot',[283,306,307,345])
def test_foreign_robot_rejected(setup,robot):
    cfg,sp,vp,state,pose=setup;state['robot_id']=robot;sp.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='编号'):apply_system_reset(cfg,sp,[vp])

@pytest.mark.parametrize('value',[float('nan'),151.])
def test_invalid_joint_rejected(setup,value):
    cfg,sp,vp,state,pose=setup;state['custom']['joints_deg']['Joint_Left_Elbow']=value;sp.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='越界'):apply_system_reset(cfg,sp,[vp])

def test_vendor_mismatch_rejected(setup):
    cfg,sp,vp,state,pose=setup;data=json.loads(vp.read_text());data['LEFT_ARM_MOTOR_HOME_ANGLE']['standing'][7]=110.;vp.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='不一致'):apply_system_reset(cfg,sp,[vp])

def test_missing_state_never_falls_back(setup):
    cfg,sp,vp,state,pose=setup;sp.unlink()
    with pytest.raises(FileNotFoundError):apply_system_reset(cfg,sp,[vp])

def test_previous_session_snapshot_stays_stable(setup):
    cfg,sp,vp,state,pose=setup;first=apply_system_reset(cfg,sp,[vp])
    state['custom']['joints_deg']['Joint_Left_Elbow']=130.;sp.write_text(json.dumps(state))
    data=json.loads(vp.read_text());data['LEFT_ARM_MOTOR_HOME_ANGLE']['standing'][7]=130.;vp.write_text(json.dumps(data))
    second=apply_system_reset(cfg,sp,[vp])
    assert first['initial_pose_deg']['left_arm'][3]==140.
    assert second['initial_pose_deg']['left_arm'][3]==130.

def test_legacy_configuration_keeps_explicit_package_mode():
    cfg={'initial_pose_deg':{'neck':[0,-20,0]}}
    assert apply_system_reset(cfg) is cfg
