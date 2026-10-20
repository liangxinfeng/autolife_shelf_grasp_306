from types import SimpleNamespace
import pytest
from autolife_shelf_grasp_306 import prepare


def test_worker_checks_original_vr_pose_before_executor(monkeypatch):
    cfg={'urdf_path':'fake','initial_pose_deg':{
        'left_arm':[0,0,0,110,0,13,0],
        'right_arm':[0,0,0,-110,0,-13,0], 'neck':[0,-20,0]}}
    monkeypatch.setattr(prepare,'digest',lambda _:'hash')
    source={'Joint_Neck_Pitch':-20.};seen=[]
    def plan(q):
        seen.append(q.copy())
        return {'kind':'prepare','source_q':q.copy(),'config':cfg,
                'self_collision_checked':True,'segments':[{'test':'checked'}]}
    result=prepare.checked_reset_plan(SimpleNamespace(reset_plan=plan),cfg,source)
    assert seen==[source] and result['config']['initial_pose_deg']==cfg['initial_pose_deg']
    assert result['prepare_mode']=='hold_then_vr' and result['reset_protocol_version']==2
    assert result['self_collision_checked'] and result['segments']
    with pytest.raises(ValueError,match='未通过检查'):
        prepare.checked_reset_plan(SimpleNamespace(reset_plan=lambda q:{'self_collision_checked':False}),cfg,source)
