from types import SimpleNamespace
import numpy as np
import pytest
from autolife_shelf_grasp_306 import planning


def test_local_contacts_after_center_path_failure(monkeypatch):
    monkeypatch.setattr(planning,'directional_yaws',lambda *a:iter([0.]))
    monkeypatch.setattr(planning,'approach_candidates',lambda *a:iter([(0.,False,'direct')]))
    calls=[]
    def route(*a):raise planning.CollisionFailure('机器人自碰撞')
    def local(*a):calls.append(a);return dict(candidate_failures=[],grasp_contact_mode='front_local')
    p=SimpleNamespace(cfg={},_route=route,_local_contact_plan=local)
    result=planning.Planner.plan(p,{},np.zeros(3),np.ones(3),dict(bbox_axes_torso=True,measured_footprint_torso=[]))
    assert calls and result['contact_fallback_reason']=='center_paths_exhausted'
    assert result['candidate_failures'][0]['reason']=='机器人自碰撞'


def test_successful_center_path_unchanged(monkeypatch):
    monkeypatch.setattr(planning,'directional_yaws',lambda *a:iter([0.]))
    monkeypatch.setattr(planning,'approach_candidates',lambda *a:iter([(0.,False,'direct')]))
    def forbidden(*a):raise AssertionError('unnecessary fallback')
    p=SimpleNamespace(cfg={},_route=lambda *a:dict(mode='center'),_local_contact_plan=forbidden)
    assert planning.Planner.plan(p,{},np.zeros(3),np.ones(3),dict(bbox_axes_torso=True))['mode']=='center'


def test_error_reports_deepest_stage_not_last_pose_error():
    failures=[dict(stage_index=1,reason='指间开度不足'),dict(stage_index=0,reason='位置误差200mm')]
    message=planning.failure_summary(failures)
    assert '伸入夹取' in message and '指间开度不足' in message and '200mm' not in message
