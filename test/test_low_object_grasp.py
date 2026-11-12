from types import SimpleNamespace as NS
from copy import deepcopy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from autolife_shelf_grasp_306.low_object_grasp import enabled,basis,support,fit_support,orientation,check_clearance
from autolife_shelf_grasp_306.core import stages


@pytest.mark.parametrize('height,expected',[(.03,True),(.0999,True),(.1,False),(.15,False),(0,False)])
def test_threshold(height,expected):assert enabled(dict(height_m=height),{}) is expected


def test_partial_tall_bottle_and_bag_not_misclassified():
    assert not enabled(dict(height_m=.04,collision_height_m=.25),{})
    assert not enabled(dict(height_m=.04,geometry='bag_edge'),{})


@pytest.mark.parametrize('up',[[0,0,1],[.4,0,.916515],[0,0,-1]])
def test_pitch_and_pregrasp_respect_gravity(up):
    up=np.array(up,float);up/=np.linalg.norm(up)
    B=basis(up,.2,35);assert np.allclose(B.T@B,np.eye(3));assert np.linalg.det(B)==pytest.approx(1)
    assert B[:,0]@up==pytest.approx(-np.sin(np.deg2rad(35)))
    center=np.array([.4,.1,.2]);cfg=dict(up_torso=up,pregrasp_m=.12,lift_m=.04,withdraw_m=.2,grasp_approach_direction=B[:,0])
    path=stages(center,.2,cfg)
    assert (path[0][1]-center)@up==pytest.approx(.12*np.sin(np.deg2rad(35)))
    assert np.allclose(path[1][1],center) and np.allclose(path[2][1]-center,up*.04)


def test_local_plane_preserves_world_coordinates_and_rejects_outliers():
    rng=np.random.default_rng(38);xy=rng.uniform([.2,-.2],[.6,.2],(2500,2))
    z=.02*xy[:,0]-.015*xy[:,1]+.70+rng.normal(0,.0005,len(xy))
    p=np.c_[xy,z];center=np.array([.4,0,.723]);prod=dict(height_m=.03)
    result=fit_support(np.r_[p,[[.4,0,.85]]*30],prod,center,[0,0,1])['low_object_support']
    assert result['source']=='local_rgbd_plane' and result['plane_points']>2000
    assert abs(np.dot([.4,0,.708],result['normal'])-result['offset'])<.001
    assert 'low_object_support' not in prod


def test_insufficient_depth_is_explicit_estimate_not_measured_table():
    p=fit_support(np.empty((0,3)),dict(height_m=.03),np.array([.4,0,.7]),[0,0,1])
    assert p['low_object_support']['source']=='object_bottom_estimate'
    assert p['low_object_support']['offset']==pytest.approx(.685)


def planner(side='right'):
    # 真实包围形状：手掌向后，指尖位于TCP处。单位米。
    cfg=dict(arm=side,T_wrist_tool_quat_xyzw=[0,0,0,1],up_torso=[0,0,1])
    v=np.array([[-.15,-.04,-.03],[-.15,.04,.03],[0,-.04,-.01],[0,.04,.01]])
    return NS(_low_object_grasp_active=True,cfg=cfg,_low_object_vertices=(side.title(),v),pose=lambda q,t:(np.array(q.get('p',[0,0,0])),np.array(q.get('R',np.eye(3)))))


@pytest.mark.parametrize('side',['left','right'])
def test_low_pitch_and_virtual_tcp_keep_object_center(side):
    p=planner(side);obj=dict(height_m=.03);center=np.array([.4,0,.015]);obj=support(obj,center,[0,0,1])
    R,d,pitch,gap,shift=orientation(p,{},np.zeros(3),center,obj,0,adjust_height=True)
    assert pitch>0 and gap>=.006-1e-9 and 0<=shift<=.013
    virtual=-R.T@np.array([0,0,shift])
    wrist=center-R@virtual
    assert np.allclose(wrist,center+[0,0,shift])
    check_clearance(p,dict(p=center,R=R),virtual,obj)
    with pytest.raises(ValueError,match='局部桌面'):
        check_clearance(p,dict(p=center-[0,0,.04],R=R),virtual,obj)


def test_high_object_keeps_old_horizontal_without_model_lookup():
    p=planner();del p._low_object_vertices
    R,d,pitch,gap,shift=orientation(p,{},np.zeros(3),np.array([.4,0,.1]),dict(height_m=.2),0)
    assert pitch==0 and shift==0 and gap is None and np.allclose(d,[1,0,0])


def test_manual_correction_does_not_move_support_plane():
    product=support(dict(height_m=.04),np.array([.4,0,.02]),[0,0,1]);original=deepcopy(product)
    assert support(product,[.4,.04,.04],[0,0,1])['low_object_support']==original['low_object_support']


def test_empty_angle_candidates_fail_without_horizontal_fallback():
    p=planner();p.cfg['low_object_pitch_candidates_deg']=[]
    with pytest.raises(ValueError,match='俯角'):orientation(p,{},np.zeros(3),np.array([.4,0,.02]),dict(height_m=.04),0)


def test_grasp_scope_cannot_leak_to_placement_or_reset():
    from autolife_shelf_grasp_306.low_object_grasp import grasp_scope
    class P:
        @grasp_scope
        def run(self,fail=False):
            assert self._low_object_grasp_active
            if fail:raise ValueError('test')
    p=P();p.run();assert not p._low_object_grasp_active
    with pytest.raises(ValueError):p.run(True)
    assert not p._low_object_grasp_active
    # 同一保存物品携带旧桌面元数据，放置时不继续施加抓取桌面。
    p=planner();p._low_object_grasp_active=False
    obj=support(dict(height_m=.03),np.array([.4,0,.015]),[0,0,1])
    check_clearance(p,dict(p=[.4,0,-1]),np.zeros(3),obj)
