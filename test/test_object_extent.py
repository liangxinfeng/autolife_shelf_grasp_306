import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from autolife_shelf_grasp_306.object_extent import coherent_component,oriented_bounds,projected_width,directional_yaws,bbox
from autolife_shelf_grasp_306.bottle import DepthQualityError
from autolife_shelf_grasp_306.demo_config import footprint_radius,carried_height


def product():
    return dict(geometry='solid',diameter_m=.07,depth_m=.18,height_m=.15,
        bbox_axes_torso=[[0,1,0],[1,0,0],[0,0,1]],bbox_extents_m=[.074,.184,.15])


def test_long_object_can_fit_across_narrow_axis():
    p=product();assert np.hypot(.074,.184)>.095
    assert projected_width(p,[0,1,0])==pytest.approx(.074)
    assert projected_width(p,[1,0,0])==pytest.approx(.184)
    cfg=dict(approach_yaw_deg=0,up_torso=[0,0,1],gripper_max_width_m=.095)
    yaws=list(directional_yaws(p,cfg));assert yaws[0]==0
    assert 90 not in yaws and 30 not in yaws
    assert footprint_radius(p)>.09 and carried_height(p)==.15


def test_rotated_object_tries_narrow_direction_not_forced_frontal():
    p=product();rotation=Rotation.from_euler('z',50,degrees=True).as_matrix()
    p['bbox_axes_torso']=(rotation@np.array(p['bbox_axes_torso'])).tolist()
    cfg=dict(approach_yaw_deg=0,up_torso=[0,0,1],gripper_max_width_m=.095)
    yaws=list(directional_yaws(p,cfg));assert yaws and 0 not in yaws
    assert min(abs(y-50) for y in yaws)<1e-8


def test_flexible_ik_tilt_can_require_more_aperture():
    p=product();closing=Rotation.from_euler('x',30,degrees=True).apply([0,1,0])
    assert projected_width(p,closing)>.095


def test_wide_in_every_direction_still_rejected():
    p=product();p['bbox_extents_m']=[.12,.18,.15]
    assert not list(directional_yaws(p,dict(approach_yaw_deg=0,up_torso=[0,0,1],gripper_max_width_m=.095)))


def test_density_removes_detached_tail_not_main_geometry():
    x,y=np.meshgrid(np.linspace(0,.05,30),np.linspace(0,.07,40));main=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    tail=main[:50]+[.3,0,0];p=np.r_[main,tail];keep=coherent_component(p)
    assert keep[:len(main)].all() and not keep[len(main):].any()
    with pytest.raises(DepthQualityError):coherent_component(np.r_[main,main+[.3,0,0]])


def test_oriented_dimensions_follow_rotated_box():
    rng=np.random.default_rng(4);p=rng.uniform([-.03,-.09,-.07],[.03,.09,.07],(5000,3))
    rotation=Rotation.from_euler('z',38,degrees=True).as_matrix();p=p@rotation.T+[.6,.1,.2]
    _,lo,hi,axes=oriented_bounds(p,[0,0,1]);np.testing.assert_allclose(sorted((hi-lo)[:2]),[.06,.18],atol=.007)
    np.testing.assert_allclose(axes.T@axes,np.eye(3),atol=1e-7)


@pytest.mark.parametrize('key,value',[('bbox_extents_m',[float('nan'),.1,.1]),('bbox_axes_torso',np.zeros((3,3)).tolist())])
def test_invalid_extent_rejected(key,value):
    with pytest.raises(ValueError):bbox({**product(),key:value})


def test_multiframe_hulls_not_rotated_empty_box_corners():
    from autolife_shelf_grasp_306.unregistered import combine_product
    from autolife_shelf_grasp_306.demo_config import require_placeable
    template=dict(id='temporary_test',geometry='solid',temporary=True,dimension_source='multi_frame_rgbd')
    vertices=np.array([[x,y,0.] for x in (-.08,.08) for y in (-.03,.03)])
    samples=[]
    for angle,shift in zip((10.,0.,0.,0.,0.),(-.001,0.,.001,0.,0.)):
        axes=Rotation.from_euler('z',angle,degrees=True).as_matrix()@np.array([[0.,1,0],[1,0,0],[0,0,1]])
        p={**template,'diameter_m':.06,'depth_m':.16,'height_m':.14,
           'bbox_axes_torso':axes.tolist(),'bbox_extents_m':[.064,.164,.14],
           'footprint_vertices_torso':vertices.tolist()}
        samples.append((np.array([.6,shift,.2]),dict(temporary_product=p)))
    result=combine_product(template,samples);require_placeable(result)
    assert projected_width(result,[0,1,0])==pytest.approx(.066)
    assert projected_width(result,[1,0,0])==pytest.approx(.164)
    assert 'footprint_vertices_torso' not in result


def test_yaw_for_negative_up_and_tilted_torso():
    from autolife_shelf_grasp_306.bottle import horizontal_basis
    up=np.array([.25,.02,-.96]);up/=np.linalg.norm(up)
    forward,side,up=horizontal_basis(up,np.deg2rad(37.))
    p=product();p['bbox_axes_torso']=np.column_stack([side,-forward,up]).tolist()
    cfg=dict(approach_yaw_deg=0,up_torso=up,gripper_max_width_m=.095)
    assert any(abs(y-37.)<1e-7 for y in directional_yaws(p,cfg))
