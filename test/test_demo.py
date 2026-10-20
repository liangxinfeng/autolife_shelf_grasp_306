import json,math
import numpy as np
import pytest
from autolife_shelf_grasp_306.demo_config import DEFAULTS,validate,require_drink
from autolife_shelf_grasp_306.demo_surface import surfaces,stable_tables,obstacle_boxes,surface_obstacles
from autolife_shelf_grasp_306.demo_protocol import validate_place


def test_parameter_limits_and_non_drink():
    for change in [{'table_height_m':float('nan')},{'place_speed_m_s':2.},{'edge_margin_m':0}]:
        with pytest.raises(ValueError):validate(change)
    with pytest.raises(ValueError):require_drink({'geometry':'bag_edge','height_m':.2})


def cloud():
    x,y=np.meshgrid(np.arange(.35,1.05,.008),np.arange(-.4,.4,.008))
    return np.c_[x.ravel(),y.ravel(),np.full(x.size,.74)]


def test_table_edge_and_obstacle_rejection():
    p=cloud();b=np.array([[.7,0,z] for z in np.arange(.77,.95,.01)])
    pts,table=surfaces(np.vstack([p,b]),DEFAULTS,.035)
    assert len(pts)>10
    assert pts[:,0].min()>.44 and pts[:,0].max()<.94
    assert np.min(np.linalg.norm(pts[:,:2]-[.7,0],axis=1))>.060
    assert abs(table['coef'][2]-.74)<.002
    stable,_=stable_tables([(pts,table)]*3);assert len(stable)==len(pts)


def test_no_fabricated_support_and_multiframe_disagreement():
    with pytest.raises(ValueError):surfaces(cloud()+[0,0,.4],DEFAULTS,.035)
    pts,table=surfaces(cloud(),DEFAULTS,.035)
    wrong={**table,'coef':[0,0,.77]}
    with pytest.raises(ValueError):stable_tables([(pts,table),(pts,wrong),(pts,table)])


def test_detected_table_obstacles_are_not_ignored():
    p=cloud();_,table=surfaces(p,DEFAULTS,.035)
    assert obstacle_boxes(p,table)==[]
    obstacle=np.array([[.65+i*.004,-.1+j*.004,.80+k*.015] for i in range(6) for j in range(6) for k in range(8)])
    boxes=obstacle_boxes(np.vstack([p,obstacle]),table)
    assert len(boxes)==1 and boxes[0]['minimum'][2]<=.74+1e-8


def test_place_requires_same_completed_run(tmp_path):
    for name,data in [('demo_run.json',dict(grasp_plan_id='one',product_id='drink',settings=DEFAULTS)),
                      ('demo_grasp_result.json',dict(status='completed',plan_id='one')),
                      ('demo_base_result.json',dict(status='completed'))]:
        (tmp_path/name).write_text(json.dumps(data))
    plan=dict(place_protocol_version=1,table_height_checked=True,parent_plan_id='one',
              product={'id':'drink'},demo_settings=DEFAULTS,segments=[{'phase':s} for s in
              ['place_approach','place_lower','place_release_retreat','place_return']])
    validate_place(plan,tmp_path)
    plan['parent_plan_id']='old'
    with pytest.raises(ValueError):validate_place(plan,tmp_path)


def test_adaptive_margin_keeps_full_bottle_support():
    x,y=np.meshgrid(np.arange(.4,.58,.003),np.arange(-.2,.2,.003))
    p=np.c_[x.ravel(),y.ravel(),np.full(x.size,.74)]
    diag={};pts,table=surfaces(p,DEFAULTS,.035,diag)
    assert diag['edge_extra_m'] in (.03,.05)
    assert pts[:,0].min()>.4+.035+.025
    assert pts[:,0].max()<.58-.035-.025
    with pytest.raises(ValueError):surfaces(p[p[:,0]<.48],DEFAULTS,.035)


def test_large_hole_is_not_fabricated_support():
    p=cloud();p=p[np.linalg.norm(p[:,:2]-[.7,0],axis=1)>.10]
    pts,_=surfaces(p,DEFAULTS,.035)
    assert np.linalg.norm(pts[:,:2]-[.7,0],axis=1).min()>.13


def test_depth_speckles_vs_repeated_low_obstacle():
    base=np.tile([.603,.003,.74],(100,1));speck=np.tile([.603,.003,.765],(2,1))
    assert len(surface_obstacles(np.vstack([base,speck]),[0,0,.74]))==0
    low=np.tile([.603,.003,.765],(30,1))
    assert len(surface_obstacles(np.vstack([base,low]),[0,0,.74]))==30
    high=np.array([[.603,.003,.81]])
    assert len(surface_obstacles(np.vstack([base,high]),[0,0,.74]))==1


def test_stability_checks_local_height_not_extrapolated_intercept():
    pts=np.array([[.6,0,.74],[.615,0,.74]])
    a=dict(coef=[0,0,.74],margin_m=.10)
    b=dict(coef=[.02,0,.728],margin_m=.09)
    result,_=stable_tables([(pts,a),(pts,b),(pts,a)])
    assert len(result)==2


def test_noisy_tilted_plane_remains_detectable():
    p=cloud();rng=np.random.default_rng(28)
    p[:,2]+=.10*(p[:,0]-.7)+rng.normal(0,.003,len(p))
    pts,table=surfaces(p,DEFAULTS,.035)
    assert len(pts)>10 and abs(table['coef'][0]-.10)<.01
