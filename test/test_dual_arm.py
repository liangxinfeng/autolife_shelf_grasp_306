import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from autolife_shelf_grasp_306 import arm_selection as a
from autolife_shelf_grasp_306.pose_handoff import start_pose_diagnostic
from autolife_shelf_grasp_306.placement_request import held_grasp,check_held_gripper,new_standalone_request
from autolife_shelf_grasp_306.demo_protocol import validate_place


def fake_fk():
    from autolife_handeye_calibration_306.core import TORSO
    joints={}
    for s,y in [('left',-.23),('right',.23)]:
        t=np.eye(4);t[1,3]=y
        joints['Link_'+s.title()+'_Wrist_Lower_to_Gripper']=(s,'fixed',TORSO,t,None)
    return SimpleNamespace(joints=joints)


@pytest.mark.parametrize('y,expected',[(-.3,'left'),(.3,'right'),(0.,'left')])
def test_rank_robot_not_camera_axis(y,expected):
    assert a.rank_arms(fake_fk(),[.4,y,0.])[0]==expected


@pytest.mark.parametrize('bad',['auto','LEFT',None,'../right'])
def test_reject_invalid_arm(bad):
    with pytest.raises(ValueError):a.gripper_name(bad)


@pytest.mark.parametrize('side',['left','right'])
def test_jaw_command_preserves_other_measured_jaw(side):
    q={'Joint_Left_Gripper':73.,'Joint_Right_Gripper':121.}
    targets=a.gripper_targets(side,330.,q)
    other='right' if side=='left' else 'left'
    assert targets[a.gripper_name(side)]==330.
    assert targets[a.gripper_name(other)]==q[a.gripper_name(other)]
    assert set(a.gripper_targets(side,10.,q,both=True).values())=={10.}


def test_loaded_right_jaw_only_relaxed():
    q={'Joint_Left_Gripper':10.,'Joint_Right_Gripper':130.}
    assert start_pose_diagnostic({**q,'Joint_Right_Gripper':134.},q,'test','place',arm='right')['ok']
    assert not start_pose_diagnostic({**q,'Joint_Left_Gripper':14.},q,'test','place',arm='right')['ok']


def test_legacy_left_receipts_and_cross_arm_rejection():
    assert a.require_same_arm({}, {})=='left'
    with pytest.raises(ValueError):a.require_same_arm({'config':{'arm':'right'}},{})
    with pytest.raises(ValueError):a.plan_arm({'arm':'left','config':{'arm':'right'}})


def test_right_held_gripper_guard():
    record={'arm':'right','q':{'Joint_Right_Gripper':120.}}
    check_held_gripper({'Joint_Right_Gripper':125.},record)
    with pytest.raises(ValueError):check_held_gripper({'Joint_Left_Gripper':120.},record)
    with pytest.raises(ValueError):check_held_gripper({'Joint_Right_Gripper':130.},record)


def test_right_standalone_place_end_to_end_provenance(tmp_path):
    from autolife_shelf_grasp_306.core import atomic_json
    from autolife_shelf_grasp_306.demo_config import validate
    pid='b'*32
    product=dict(id='water',name='water',geometry='cylinder',height_m=.2,diameter_m=.07)
    grasp=dict(plan_id=pid,run_id=tmp_path.name,product=product,config={'arm':'right'})
    record=dict(status='completed',plan_id=pid,arm='right',q={'Joint_Left_Gripper':10.,'Joint_Right_Gripper':120.})
    atomic_json(tmp_path/('plan_'+pid+'.json'),grasp);atomic_json(tmp_path/'held_object.json',record)
    assert held_grasp(tmp_path)==(grasp,record)
    context=new_standalone_request(tmp_path,validate({}))
    plan=dict(kind='place',config={'arm':'right'},placement_mode='standalone',placement_request_id=context['run_id'],
        place_protocol_version=1,table_height_checked=True,parent_plan_id=pid,product=product,
        demo_settings=context['settings'],source_q=record['q'],
        segments=[dict(phase=p) for p in ('place_approach','place_lower','place_release_retreat','place_return')])
    validate_place(plan,tmp_path)
    plan['config']['arm']='left'
    with pytest.raises(ValueError):validate_place(plan,tmp_path)


@pytest.mark.parametrize('failing',[[],['right'],['right','left']])
def test_pool_prefers_near_hand_and_bounded_fallback(monkeypatch,failing):
    calls=[]
    def geometry(cfg,cal,fk,side):return {**cfg,'arm':side,'tool_geometry':{'provisional':side=='right'}},np.array([.23,0.,0.])
    monkeypatch.setattr(a,'tool_geometry',geometry)
    class Planner:
        def __init__(self,cfg):self.cfg=cfg
        def plan(self,q,tcp,point,product):
            calls.append(self.cfg['arm'])
            if self.cfg['arm'] in failing:raise ValueError('IK unavailable')
            return {}
    cfg={'arm':'left','up_torso':[0.,0.,1.]}
    pool=a.ArmPlannerPool(cfg,{},fake_fk(),Planner)
    if len(failing)==2:
        with pytest.raises(ValueError,match='双臂'):pool.plan({},[.4,.3,0.],{},cfg)
        assert calls==['right','left']
    else:
        p,planner,c,tcp,center,correction=pool.plan({},[.4,.3,0.],{},cfg)
        assert p['arm']==('left' if failing else 'right')
        assert calls==(['right','left'] if failing else ['right'])
        assert cfg=={'arm':'left','up_torso':[0.,0.,1.]}


def test_cancel_never_enters_next_arm(monkeypatch):
    pool=a.ArmPlannerPool({'arm':'left'}, {},fake_fk(),lambda cfg:None)
    with pytest.raises(ValueError,match='停止'):pool.plan({},[.4,.3,0.],{}, {},stopped=lambda:True)


def test_tool_transfer_keeps_real_mount_not_torso_mirror(monkeypatch):
    from autolife_handeye_calibration_306 import verification_core as v
    monkeypatch.setattr(v,'load_geometry',lambda *args,**kwargs:np.array([.23,.007,-.016]))
    fk=SimpleNamespace(joints={},fk=lambda *args:np.eye(4))
    cfg={'tcp_offset_mm':15.,'robot_id':300,'T_wrist_tool_quat_xyzw':[-.70710678,0,0,.70710678],
         'target_offset_torso_mm':[0,15,0]}
    right,tcp=a.tool_geometry(cfg,{'config':{'arm':'left'}},fk,'right')
    assert np.allclose(tcp,[.23,.007,-.016])
    assert right['tool_geometry']['provisional']
    assert right['target_offset_torso_mm']==[0,0,0]
    left,tcp=a.tool_geometry(cfg,{'config':{'arm':'left'}},fk,'left')
    assert left['target_offset_torso_mm']==[0,15,0]
    assert not left['tool_geometry']['provisional']
