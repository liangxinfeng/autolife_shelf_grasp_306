import numpy as np
import pytest
from types import SimpleNamespace
from autolife_shelf_grasp_306.demo_surface import surfaces,obstacle_boxes
from autolife_shelf_grasp_306.demo_config import DEFAULTS
from autolife_shelf_grasp_306.demo_planning import placement_candidates,rank_place_ik,IKFailure


def patch(height=.9):
    x,y=np.meshgrid(np.arange(.4,.9,.0025),np.arange(-.25,.25,.0025))
    return np.c_[x.ravel(),y.ravel(),np.full(x.size,height)]


def test_semantic_height_is_measured_not_reference_height():
    p=patch(.96);diag={}
    candidates,table=surfaces(p,DEFAULTS,.035,diag,support_points=p)
    assert len(candidates)>0 and abs(table['coef'][2]-.96)<1e-8
    assert diag['semantic_support'] and not diag['unknown_is_free']


def test_only_visible_patch_required_not_whole_table():
    p=patch();p=p[(p[:,0]<.58)&(p[:,1]<-.05)]
    candidates,_=surfaces(p,DEFAULTS,.035,support_points=p)
    assert len(candidates)>0 and candidates[:,0].max()<.58-.035


def test_large_distant_patch_does_not_hide_near_narrow_patch():
    near=patch();near=near[(near[:,0]<.515)&(near[:,1]<-.05)]
    far=patch()+[.7,0,0]
    pts,_=surfaces(np.vstack([near,far]),DEFAULTS,.0325,support_points=np.vstack([near,far]))
    assert np.any(pts[:,0]<.55) and np.any(pts[:,0]>1.1)


def test_unknown_hole_and_obstacle_outside_semantic_mask_remain_blocked():
    p=patch();p=p[np.linalg.norm(p[:,:2]-[.55,0],axis=1)>.035]
    obstacle=np.array([[.75+i*.002,j*.002,1.0] for i in range(-8,9) for j in range(-8,9)])
    pts,_=surfaces(np.vstack([p,obstacle]),DEFAULTS,.035,support_points=p)
    assert np.linalg.norm(pts[:,:2]-[.55,0],axis=1).min()>.075
    assert np.linalg.norm(pts[:,:2]-[.75,0],axis=1).min()>.06


def test_mask_never_connects_distinct_support_islands():
    p=patch();p=p[np.abs(p[:,0]-.65)>.06]
    pts,_=surfaces(p,DEFAULTS,.035,support_points=p)
    assert np.all(np.abs(pts[:,0]-.65)>.06+.035)


def test_only_vertical_surface_or_floor_rejected():
    p=patch();vertical=p.copy();vertical[:,[0,2]]=p[:,[2,0]]
    for seeds in (vertical,patch(.1),np.empty((0,3))):
        with pytest.raises(ValueError):surfaces(seeds,DEFAULTS,.035,support_points=seeds)


def test_candidates_bounded_diverse_and_near_current():
    points=patch()[:,:];current=np.array([.8,-.15,1.])
    chosen=placement_candidates(points,current)
    assert len(chosen)==10
    assert np.linalg.norm(chosen[0][:2]-current[:2])<.04
    assert len({tuple(p) for p in chosen})==len(chosen)


def test_concave_clutter_does_not_turn_empty_inside_into_a_box():
    x,y=np.meshgrid(np.arange(.4,.9,.005),np.arange(-.25,.25,.005))
    keep=(x<.46)|(y>.19)
    p=np.c_[x[keep],y[keep],np.full(keep.sum(),1.05)]
    table=dict(coef=[0,0,.9],minimum=[.4,-.25,.9],maximum=[.9,.25,.9],support_method='observed_semantic_plane')
    boxes=obstacle_boxes(p,table)
    center=np.array([.7,0.,1.])
    assert boxes and not any(np.all(center>=b['minimum']) and np.all(center<=b['maximum']) for b in boxes)
    for point in p[::37]:assert any(np.all(point>=b['minimum']) and np.all(point<=b['maximum']) for b in boxes)
    assert len(obstacle_boxes(None,table,frames=[p]*5))==len(boxes)


def test_endpoint_screen_ranks_without_sending_motion_or_path_search():
    import time
    calls=[]
    class Planner:
        names=['a']
        def solve(self,source,tcp,point,rotation,*args,**kwargs):
            calls.append('ik')
            if point[0]>.8:raise IKFailure('unreachable')
            return {'a':point[0]*30}
        def check(self,*args):calls.append('collision')
    guard=SimpleNamespace(check=lambda *args:calls.append('table'))
    diagnostics=[]
    ranked=rank_place_ik(Planner(),{'a':0},np.zeros(3),[np.array([x,0,.9]) for x in (.9,.7,.5)],
        np.eye(3),np.array([0,0,1]),{},DEFAULTS,guard,time.monotonic()+8,diagnostics)
    assert ranked and all(v['target'][0]<=.8 for v in ranked)
    assert ranked[0]['target'][0]==.5 and len(ranked)<=6
    assert diagnostics[-1]['phase']=='ik_screen_summary'
    assert set(calls)=={'ik','collision','table'}
