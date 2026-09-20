"""Observed support, not a convex hull or blind plane extension."""
import numpy as np
import pytest
from autolife_shelf_grasp_306.demo_config import DEFAULTS,validate
from autolife_shelf_grasp_306.demo_surface import surfaces,obstacle_boxes,retryable_surface_error


def strip(width=.11):
    x,y=np.meshgrid(np.arange(.55,.55+width,.0025),np.arange(-.25,.25,.0025))
    return np.c_[x.ravel(),y.ravel(),np.full(x.size,.74)]


def test_narrow_but_full_bottle_supported_strip():
    diag={};p=strip();radius=.0325
    pts,table=surfaces(p,DEFAULTS,radius,diag)
    assert len(pts)>0 and diag['grid_resolution_m']==.005
    assert .01<=diag['edge_extra_m']<.03
    assert table['margin_m']>=radius+.01
    assert np.min(pts[:,0]-p[:,0].min())>=radius+.01
    assert np.min(p[:,0].max()-pts[:,0])>=radius+.01


def test_small_margin_is_user_adjustable_and_not_ignored():
    with pytest.raises(ValueError,match='支撑半径'):
        surfaces(strip(),{**DEFAULTS,'minimum_edge_margin_m':.03},.0325)


def test_gap_and_unknown_space_not_filled():
    p=strip(.6);p=p[np.abs(p[:,0]-.85)>.10]
    pts,_=surfaces(p,DEFAULTS,.035)
    assert np.all(np.abs(pts[:,0]-.85)>.10+.035+.01)


def test_bottle_must_still_fit_and_obstacles_remain_blocked():
    with pytest.raises(ValueError):surfaces(strip(.065),DEFAULTS,.0325)
    p=strip(.4);obstacle=np.array([[.75+i*.003,j*.003,.82] for i in range(-5,6) for j in range(-5,6)])
    pts,_=surfaces(np.vstack([p,obstacle]),DEFAULTS,.0325)
    assert np.min(np.linalg.norm(pts[:,:2]-[.75,0],axis=1))>.0325+.025


def test_config_lower_bounds_and_faster_lowering():
    assert DEFAULTS['place_speed_m_s']==.03
    for v in [0.,-.01,.005,float('nan'),float('inf')]:
        with pytest.raises(ValueError):validate({'minimum_edge_margin_m':v})


def test_temporal_concatenation_does_not_invent_spatial_neighbours():
    table=dict(coef=[0,0,.74],minimum=[.4,-.2,.74],maximum=[.9,.2,.74])
    # Two points per cell per frame are not independent spatial support;
    # merging five frames first used to turn them into 10 neighbours.
    specks=np.array([[x,y,.765] for x in [.61,.625,.64] for y in [0,.002]])
    assert obstacle_boxes(None,table,frames=[specks]*5)==[]


def test_real_or_latest_frame_obstacle_is_not_voted_away():
    table=dict(coef=[0,0,.74],minimum=[.4,-.2,.74],maximum=[.9,.2,.74])
    for height in [.765,.82]:
        obj=np.array([[.65+i*.001,-.01+j*.001,height] for i in range(10) for j in range(10)])
        empty=np.empty((0,3))
        assert len(obstacle_boxes(None,table,frames=[empty]*4+[obj]))==1


def test_old_low_noise_requires_positive_latest_clear_evidence():
    table=dict(coef=[0,0,.74],minimum=[.4,-.2,.74],maximum=[.9,.2,.74])
    obj=np.array([[.65+i*.001,-.01+j*.001,.765] for i in range(10) for j in range(10)])
    clear=obj.copy();clear[:,2]=.74
    assert obstacle_boxes(None,table,frames=[obj]*4+[clear])==[]
    assert obstacle_boxes(None,table,frames=[obj]*4+[np.empty((0,3))])
    tall=obj.copy();tall[:,2]=.82
    assert obstacle_boxes(None,table,frames=[tall]*4+[clear])
    assert obstacle_boxes(None,table,frames=[clear]*4+[obj])


def test_refresh_bounded_and_not_used_for_motion_or_ik_faults():
    for count in range(2):assert retryable_surface_error('放置轨迹接近桌面已有物品#1',count)
    assert not retryable_surface_error('放置轨迹接近桌面已有物品#1',2)
    for message in ['用户停止演示','关节发生移动','位置误差20mm','SYNC运行期间硬件或反馈失效']:
        assert not retryable_surface_error(message,0)
