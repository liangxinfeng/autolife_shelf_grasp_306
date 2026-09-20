import numpy as np
import pytest
from autolife_shelf_grasp_306.bottle import center_from_points,combine,horizontal_basis,DepthQualityError,estimate,planning_product
from autolife_shelf_grasp_306.core import stages,reset_blockers,execution_blockers


def cylinder():
    t,z=np.meshgrid(np.linspace(-1.1,1.1,40),np.linspace(-.109,.109,80))
    p=np.c_[.6-.035*np.cos(t.ravel()),.1+.035*np.sin(t.ravel()),.2+z.ravel()]
    return p,{'diameter_m':.07,'height_m':.22}


@pytest.mark.parametrize('sign',[1,-1])
def test_center_does_not_assume_z_up(sign):
    p,product=cylinder();p[:,2]*=sign
    center,stats,_=center_from_points(p,[0,0,sign],product,[0,.1,.2*sign])
    np.testing.assert_allclose(center,[.6,.1,.2*sign],atol=.002)
    assert stats['height_source']=='visible_bottle_midpoint'


def test_incomplete_height_uses_visible_center():
    p,product=cylinder();p=p[p[:,2]>.2]
    c,stats,_=center_from_points(p,[0,0,1],product,[0,.1,.2])
    np.testing.assert_allclose(c,[.6,.1,.255],atol=.002)
    assert 105<stats['observed_height_mm']<112


@pytest.mark.parametrize('reference',[.04,.10,.22,.40])
def test_reference_height_does_not_change_target(reference):
    p,product=cylinder();product['height_m']=reference
    c,stats,_=center_from_points(p,[0,0,1],product,[0,.1,.2])
    np.testing.assert_allclose(c,[.6,.1,.2],atol=.002)


def test_only_tiny_fragment_still_rejected():
    p,product=cylinder();p=p[np.abs(p[:,2]-.2)<.01]
    with pytest.raises(DepthQualityError,match='25mm'):
        center_from_points(p,[0,0,1],product,[0,.1,.2])


def cropped_frame(monkeypatch):
    import autolife_handeye_calibration_306.depth_target as depth_target
    monkeypatch.setattr(depth_target,'aligned_depth',lambda *args:None)
    h,w=160,180;K=np.array([[600.,0,90],[0,600.,80],[0,0,1.]])
    ray=(np.arange(w)-90)/600.;a=1+ray**2;discriminant=.6**2-a*(.6**2-.035**2)
    valid=discriminant>0;z=(.6-np.sqrt(np.maximum(0,discriminant)))/a
    mask=np.broadcast_to(valid,(h,w)).copy();mask[120:]=False
    frame={'image':np.zeros((h,w,3),np.uint8),'depth':np.broadcast_to(z*1000,(h,w)).copy(),
           'K':K,'D':np.zeros(5),'frame_id':1,'depth_id':1,
           'intrinsics':{'model_id':0},'depth_intrinsics':{'model_id':0}}
    return frame,mask,{'depth_scale_m':.001,'depth_range_m':[.15,2.]}


def test_image_border_crop_plans_visible_midpoint(monkeypatch):
    frame,mask,cfg=cropped_frame(monkeypatch)
    c,stats=estimate(frame,mask,np.eye(4),[0,-1,0],{'diameter_m':.07,'height_m':.22},cfg)
    assert stats['image_cropped'] and .09<stats['observed_height_mm']/1000<.13
    assert abs(c[0])<.002 and abs(c[2]-.6)<.003 and -.025<c[1]<-.015


def test_cropped_object_still_requires_depth_and_stable_center(monkeypatch):
    frame,mask,cfg=cropped_frame(monkeypatch);frame['depth'][:]=0
    with pytest.raises(DepthQualityError,match='深度不足'):
        estimate(frame,mask,np.eye(4),[0,-1,0],{'diameter_m':.07,'height_m':.22},cfg)


def test_measured_height_and_hidden_collision_extent_do_not_mutate_catalog():
    p={'id':'sprite','geometry':'cylinder','diameter_m':.07,'height_m':.22}
    geometry={'observed_height_mm':100.,'image_cropped':True}
    result=planning_product(p,geometry)
    assert result['height_m']==.1 and result['collision_height_m']==pytest.approx(.34)
    assert p['height_m']==.22 and 'collision_height_m' not in p
    geometry.update(observed_height_mm=280.,image_cropped=False)
    result=planning_product(p,geometry)
    assert result['height_m']==.28 and result['collision_height_m']==.28


def test_invalid_dimensions_are_not_retryable_depth_error():
    p,product=cylinder();product['diameter_m']=.9
    with pytest.raises(ValueError,match='尺寸') as error:center_from_points(p,[0,0,1],product,[0,.1,.2])
    assert not isinstance(error.value,DepthQualityError)


def test_lift_uses_real_up():
    cfg={'up_torso':[0,0,-1],'pregrasp_m':.12,'lift_m':.035,'withdraw_m':.2}
    out=stages([.6,.1,-.2],0,cfg)
    np.testing.assert_allclose(out[2][1],[.6,.1,-.235])
    f,s,u=horizontal_basis([.1,0,-1]);assert abs(f@u)<1e-10
    assert np.linalg.det(np.column_stack([f,s,u]))==pytest.approx(1)


def test_no_silent_default_up():
    with pytest.raises(KeyError):stages([0,0,0],0,{'lift_m':.03})


def test_reset_does_not_unlock_grasp():
    cfg={'reset_hardware_enabled':True}
    assert reset_blockers(cfg)==[] and len(execution_blockers(cfg))==5
    assert reset_blockers({'reset_hardware_enabled':'true'})


def test_multiframe():
    samples=[(np.array([.6,.1,-.2]),{'observed_height_mm':219}) for _ in range(5)]
    c,stats=combine(samples);assert stats['support_required'] is False
    samples[-1]=(c+[.03,0,0],samples[-1][1])
    with pytest.raises(ValueError,match='10mm'):combine(samples)
