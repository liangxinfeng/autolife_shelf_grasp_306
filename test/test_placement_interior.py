import numpy as np
from scipy.spatial.transform import Rotation
from autolife_shelf_grasp_306.placement_support import pack_support
from autolife_shelf_grasp_306.placement_interior import InteriorPreference,interior_rank
from autolife_shelf_grasp_306.demo_planning import placement_candidates


def table(grid=None):
    result=dict(minimum=[0,0,0],maximum=[1,1,0])
    if grid is not None:result['footprint_support']=pack_support(grid,[0,0],.005)
    return result


def test_prefers_inside_over_nearest_edge_without_deleting_fallback():
    points=np.array([[.04,.5,0],[.13,.5,0],[.5,.5,0]])
    pref=InteriorPreference(table(),.03)
    selected=placement_candidates(points,np.array([0,.5,0]),limit=3,
        edge_margins=pref.margins(points,np.eye(4)))
    np.testing.assert_allclose(selected[0],points[1])
    assert len(selected)==3


def test_comfort_met_prefers_reachable_nearer_site_not_table_centre():
    points=np.array([[.13,.5,0],[.5,.5,0]])
    pref=InteriorPreference(table(),.03)
    selected=placement_candidates(points,np.zeros(3),edge_margins=pref.margins(points,np.eye(4)))
    np.testing.assert_allclose(selected[0],points[0])


def test_support_holes_and_outer_border_are_not_filled():
    grid=np.ones((200,200),bool);grid[80:120,80:120]=False
    pref=InteriorPreference(table(grid),.03)
    margins=pref.margins([[.02,.5,0],[.5,.5,0],[.25,.25,0]],np.eye(4))
    assert margins[0]<0 and margins[1]<0 and margins[2]>.08


def test_waist_anchor_translation_and_rotation_do_not_change_margin():
    pref=InteriorPreference(table(np.ones((200,200),bool)),.03)
    points=np.array([[.13,.51,.6],[.21,.71,.6]])
    T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',[0,-29,17],degrees=True).as_matrix();T[:3,3]=[.1,0,1]
    torso=(points-T[:3,3])@T[:3,:3]
    np.testing.assert_allclose(pref.margins(torso,T),pref.margins(points,np.eye(4)),atol=.011)


def test_ik_motion_cost_does_not_reverse_inside_preference():
    edge=dict(score=5.,edge_shortfall_m=.06)
    inside=dict(score=60.,edge_shortfall_m=0.)
    assert sorted([edge,inside],key=interior_rank)[0] is inside
    assert interior_rank(dict(score=3.))==interior_rank(dict(score=3.,edge_shortfall_m=0.))


def test_larger_objects_need_more_room_and_narrow_table_still_has_candidates():
    points=np.array([[.04,.5,0],[.07,.5,0]])
    small=InteriorPreference(table(),.02);big=InteriorPreference(table(),.06)
    np.testing.assert_allclose(small.margins(points,np.eye(4))-big.margins(points,np.eye(4)),.04)
    selected=placement_candidates(points,np.zeros(3),edge_margins=big.margins(points,np.eye(4)))
    assert len(selected)==2
