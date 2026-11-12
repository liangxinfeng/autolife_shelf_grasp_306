import json
import numpy as np
import pytest
from autolife_shelf_grasp_306.unregistered import target_id,same_target,temporary_product,estimate_temporary,combine_product
from autolife_shelf_grasp_306.bottle import DepthQualityError
from autolife_shelf_grasp_306.demo_config import require_placeable,footprint_radius,validate
from autolife_shelf_grasp_306.placement_request import new_standalone_request,standalone_source
from autolife_shelf_grasp_306.core import atomic_json


def item(label='bottle'):
    return dict(id=3,name='未录入 · '+label,product=None,semantic_label=label,
                mask=np.ones((20,20),bool),feature=np.array([1.,0.]))


def test_identity_scoped_to_session_and_instance():
    a=item();assert target_id(a,'a')==target_id(a,'a')
    assert target_id(a,'a')!=target_id(a,'b')
    assert target_id(a,'a')!=target_id({**a,'id':4},'a')
    assert target_id({**a,'product':{'id':'saved'}},'a')=='saved'


def test_unknown_identity_rechecks_mask_and_appearance():
    a=item();assert same_target(a,a,'a')
    assert not same_target(a,None,'a')
    assert not same_target(a,{**a,'feature':np.array([0.,1.])},'a')
    assert not same_target(a,{**a,'mask':np.zeros((20,20),bool)},'a')
    assert not same_target(a,{**a,'product':{'id':'new-registration'}},'a')


@pytest.mark.parametrize('label',['table','white bag','person','hand','rope','towel'])
def test_not_all_semantic_masks_are_graspable(label):
    with pytest.raises(ValueError):temporary_product(item(label),'s')


def test_temporary_product_never_supplies_guessed_dimensions():
    p=temporary_product(item(),'s');assert p['geometry']=='cylinder' and p['temporary']
    assert 'diameter_m' not in p and 'height_m' not in p
    assert temporary_product(item('box'),'s')['geometry']=='solid'


def test_multiframe_geometry_and_identity():
    template=temporary_product(item(),'s');p={**template,'diameter_m':.06,'height_m':.18,'depth_m':.06}
    samples=[(np.zeros(3),{'temporary_product':dict(p)}) for _ in range(5)]
    product=combine_product(template,samples);require_placeable(product)
    assert product['dimension_frames']==5 and 'height_m' not in template
    samples[-1][1]['temporary_product']['height_m']=.24
    with pytest.raises(ValueError,match='尺寸不稳定'):combine_product(template,samples)


def test_temporary_geometry_must_be_confirmed_for_place():
    p=dict(geometry='solid',diameter_m=.05,height_m=.08,depth_m=.04,temporary=True)
    with pytest.raises(ValueError):require_placeable(p)
    p.update(dimension_source='multi_frame_rgbd',dimension_frames=5);require_placeable(p)
    assert footprint_radius(p)==pytest.approx(np.hypot(.05,.04)/2)
    assert footprint_radius({**p,'geometry':'cylinder'})==.025


@pytest.mark.parametrize('field,value',[('diameter_m',.2),('height_m',float('nan')),('geometry','bag_edge'),('depth_m',.3)])
def test_bad_geometry_cannot_place(field,value):
    p=dict(geometry='solid',diameter_m=.05,height_m=.08,depth_m=.04)
    with pytest.raises(ValueError):require_placeable({**p,field:value})


def test_unknown_solid_uses_observed_top_not_fake_thickness(monkeypatch):
    import autolife_shelf_grasp_306.unregistered as module
    y,x=np.meshgrid(np.linspace(.075,.125,40),np.linspace(.58,.62,40))
    top=np.c_[x.ravel(),y.ravel(),np.full(x.size,.18)]
    z,y=np.meshgrid(np.linspace(.10,.18,40),np.linspace(.075,.125,40))
    front=np.c_[np.full(z.size,.58),y.ravel(),z.ravel()];points=np.r_[top,front]
    pixels=np.c_[np.zeros(len(points)),np.linspace(1,99,len(points))]
    monkeypatch.setattr(module,'object_points',lambda *a:(points,pixels,(0,100)))
    mask=np.zeros((120,120),bool);mask[10:110,10:110]=True
    T=np.eye(4);T[:3,3]=[0,.1,.4]
    template=temporary_product(item('box'),'s')
    c,stats=estimate_temporary({},mask,T,[0,0,1],template,{})
    np.testing.assert_allclose(c,[.6,.1,.14],atol=.002)
    product=stats['temporary_product']
    np.testing.assert_allclose(sorted([product['diameter_m'],product['depth_m']]),[.04,.05],atol=.003)
    monkeypatch.setattr(module,'object_points',lambda *a:(front,pixels[:len(front)],(0,100)))
    with pytest.raises(DepthQualityError):estimate_temporary({},mask,T,[0,0,1],template,{})


def test_unregistered_cropped_target_rejected_without_guessing():
    with pytest.raises(ValueError,match='完整入镜'):
        estimate_temporary({},np.ones((120,120),bool),np.eye(4),[0,0,1],temporary_product(item(),'s'),{})


def test_temporary_held_product_survives_without_catalog(tmp_path):
    pid='a'*32;product=dict(id='temporary_x',name='未录入物品',temporary=True,
        dimension_source='multi_frame_rgbd',dimension_frames=5,geometry='solid',diameter_m=.05,height_m=.08,depth_m=.04)
    atomic_json(tmp_path/('plan_'+pid+'.json'),dict(plan_id=pid,run_id=tmp_path.name,product=product))
    atomic_json(tmp_path/'held_object.json',dict(status='completed',plan_id=pid,q={'Joint_Left_Gripper':110}))
    context=new_standalone_request(tmp_path,validate({}))
    assert context['product_id']=='temporary_x'
    assert standalone_source(tmp_path)[1]['product']==product
    assert not (tmp_path/'products.json').exists()
