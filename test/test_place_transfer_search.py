import time
from types import SimpleNamespace
import numpy as np
import pytest
from autolife_shelf_grasp_306.demo_planning import (
    placement_candidates,rank_place_ik,checked_transfer_approach,CollisionFailure)
from autolife_shelf_grasp_306.demo_surface import obstacle_boxes
from autolife_handeye_calibration_306.verification_trajectory import JointPath


def test_surface_coverage_keeps_near_point_and_other_empty_pockets():
    near=np.c_[np.linspace(0,.2,100),np.zeros(100),np.ones(100)]
    far=np.array([[.5,.25,1.],[.6,-.25,1.]])
    points=np.vstack([near,far]);current=np.array([0.,0.,1.])
    chosen=placement_candidates(points,current,limit=3,cover_surface=True)
    np.testing.assert_allclose(chosen[0],near[0])
    assert all(any(np.allclose(c,p) for c in chosen) for p in far)


def test_blocked_connector_is_only_a_candidate_not_an_executable_path():
    class Planner:
        names=['a']
        def solve(self,*a,**kw):return {'a':12.}
        def check(self,q,*a):
            if 4<q['a']<8:raise CollisionFailure('blocked intermediate')
    args=(Planner(),{'a':0.},np.zeros(3),[np.array([.5,0,.8])],np.eye(3),
          np.array([0,0,1.]),{},dict(approach_height_m=.1),SimpleNamespace(check=lambda *a:None))
    assert not rank_place_ik(*args,time.monotonic()+8,[])
    choices=rank_place_ik(*args,time.monotonic()+8,[],allow_detour=True)
    assert len(choices)==1 and choices[0]['needs_detour']
    assert 'segments' not in choices[0]


class TransferPlanner:
    names=['a','b']
    def pose(self,q,tcp):return np.array([q['a']/100,q['b']/100,.5]),np.eye(3)
    def solve(self,q,tcp,target,*a,**kw):return {**q,'a':target[0]*100,'b':target[1]*100}


def test_side_transfer_checks_curve_and_preserves_non_arm_state():
    p=TransferPlanner();source={'a':0.,'b':0.,'neck':17.};goal={'a':30.,'b':20.,'neck':17.}
    checked=[]
    def check(q,carrying):
        assert carrying and q['neck']==17.
        if 10<q['a']<20 and q['b']<8:raise CollisionFailure('clutter')
        checked.append(q)
    points=checked_transfer_approach(p,source,goal,np.zeros(3),np.array([0,0,1.]),check)
    assert points[0]==source and points[-1]==goal and len(points)>=3
    assert len(checked)>20
    for values in JointPath(p.names,points).checks():check({**source,**values},True)


def test_transfer_never_accepts_colliding_waypoints():
    def check(*a):raise CollisionFailure('solid obstacle')
    with pytest.raises(CollisionFailure,match='solid obstacle'):
        checked_transfer_approach(TransferPlanner(),{'a':0.,'b':0.},{'a':30.,'b':20.},
                                 np.zeros(3),np.array([0,0,1.]),check)


def test_transfer_propagates_deadline_not_fallback_motion():
    def check(*a):raise TimeoutError('budget')
    with pytest.raises(TimeoutError):
        checked_transfer_approach(TransferPlanner(),{'a':0.,'b':0.},{'a':30.,'b':20.},
                                 np.zeros(3),np.array([0,0,1.]),check)


@pytest.mark.parametrize('semantic',[False,True])
def test_reduced_padding_still_encloses_measured_obstacle_and_table_column(semantic):
    x,y=np.meshgrid(np.linspace(.411,.431,6),np.linspace(.011,.031,5))
    points=np.c_[x.ravel(),y.ravel(),np.full(x.size,1.)]
    table=dict(coef=[0,0,.8],minimum=[.3,-.2,.8],maximum=[.8,.2,.8])
    if semantic:table['support_method']='observed_semantic_plane'
    boxes=obstacle_boxes(points,table)
    assert len(boxes)==1
    box=boxes[0];lo=np.array(box['minimum']);hi=np.array(box['maximum'])
    assert box['padding_m']==.010
    np.testing.assert_allclose(lo[:2],points.min(axis=0)[:2]-.010)
    np.testing.assert_allclose(hi,points.max(axis=0)+.010)
    assert lo[2]<=.8 and np.all(points>=lo) and np.all(points<=hi)
