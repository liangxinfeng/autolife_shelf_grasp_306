"""已识别物体的深度位置纠偏：真实投影、错误输入及调用模式隔离。"""
from copy import deepcopy
import numpy as np
import pytest
from autolife_shelf_grasp_306.visual_shape_position import position_from_shape

CFG=dict(depth_scale_m=.001,depth_range_m=[.1,2.])


def scene(dx=0,dy=0,z=600.):
    mask=np.zeros((240,320),bool);mask[60+dy:180+dy,130+dx:190+dx]=True
    depth=np.full(mask.shape,900.,float);depth[mask]=z
    return dict(mask=mask,depth=depth,K=np.array([[400.,0,160],[0,400.,120],[0,0,1]]),D=np.zeros(5))


@pytest.mark.parametrize('dx,dy',[(0,0),(15,-10),(-12,10)])
def test_shape_translation_with_no_descriptors(dx,dy):
    a=scene();b=scene(dx,dy);d,s=position_from_shape(a,np.eye(4),b,np.eye(4),CFG)
    assert np.allclose(d[:3,3],[dx*.6/400,dy*.6/400,0],atol=1e-8)
    assert np.array_equal(d[:3,:3],np.eye(3)) and not s['orientation_observed']
    assert s['angle_resolution_deg']==180 and s['inliers']>100


def test_uses_camera_transform_not_camera_pixel_difference():
    a=scene();T=np.eye(4);T[:3,3]=[.03,-.02,.01]
    d,_=position_from_shape(a,np.eye(4),a,T,CFG)
    assert np.allclose(d[:3,3],T[:3,3])


def test_partial_transparent_background_does_not_bias_translation():
    a=scene();b=scene(10);b['depth'][65:95,145:175]=1300
    d,s=position_from_shape(a,np.eye(4),b,np.eye(4),CFG)
    assert np.allclose(d[:3,3],[.015,0,0],atol=.0001) and s['inliers']<s['matches']


def test_partial_object_or_wrong_shape_rejected():
    a=scene();b=scene();b['mask'][120:]=False
    with pytest.raises(ValueError,match='深度'):position_from_shape(a,np.eye(4),b,np.eye(4),CFG)


def test_small_label_patch_depth_not_enough_for_whole_object():
    a=scene();b=scene();b['depth'][:]=0;b['depth'][105:130,140:180]=600
    with pytest.raises(ValueError,match='局部|不足'):position_from_shape(a,np.eye(4),b,np.eye(4),CFG)


def test_bad_camera_direction_or_offscreen_cannot_fake_translation():
    from scipy.spatial.transform import Rotation
    a=scene();T=np.eye(4);T[:3,:3]=Rotation.from_euler('y',25,degrees=True).as_matrix()
    with pytest.raises(ValueError,match='观察方向'):position_from_shape(a,np.eye(4),a,T,CFG)
    a['mask'][:65,130:190]=True
    with pytest.raises(ValueError,match='超出画面'):position_from_shape(a,np.eye(4),a,np.eye(4),CFG)


def test_appearance_supplement_can_win_over_weak_original(monkeypatch):
    from autolife_shelf_grasp_306.visual_multiview import seed_match
    a=(dict(metric_registered=True),{'tag':'old'})
    b=(dict(metric_registered=False),{'tag':'new'})
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_multiview.register_views',lambda *args:(np.eye(4),dict(inliers=8,rms_mm=3),*a))
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_registration.register',lambda *args:(np.eye(4),dict(inliers=40,rms_mm=1)))
    _,_,meta,arrays=seed_match([a,b],{})
    assert arrays['tag']=='new' and not meta['metric_registered']


def test_observe_fallback_uses_initial_not_supplement_position(monkeypatch):
    from types import SimpleNamespace as NS
    from autolife_shelf_grasp_306.visual_action_worker import VisualActions
    from autolife_shelf_grasp_306.visual_reference import matrix
    a=scene();a.update(image=np.zeros((240,320,3),np.uint8),points=np.zeros((10,3)),pixels=np.zeros((10,2)),descriptors=np.zeros((10,128)),cloud=np.zeros((40,3)))
    b=deepcopy(a);b['mask']=scene(10)['mask']
    meta=dict(camera_pose=np.eye(4).tolist(),object_pose=np.eye(4).tolist(),created=1)
    supplemental=dict(meta,metric_registered=False,created=2,object_pose=(np.eye(4)+np.diag([0,0,0,0])).tolist())
    supplemental['object_pose'][0][3]=.7
    v=VisualActions.__new__(VisualActions);v.cfg=CFG;v.perception=None;v.reference_cache={}
    v.store=NS(views=lambda _: [(meta,a),(supplemental,a)],view_ids=lambda _:['first','supplement'])
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_multiview.resolve_views',lambda x:x)
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_registration.image_features',lambda *a,**k:(_ for _ in ()).throw(ValueError('use stored')))
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_multiview.seed_match',lambda *args:(np.eye(4),dict(inliers=20),supplemental,a))
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_box_selection.projected_box',lambda *a:[.1,.1,.9,.9])
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_box_selection.segment_box',lambda *a:dict(mask=b['mask']))
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_action_worker.features',lambda *a,**k:{})
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_action_worker.register_views',lambda *a:(_ for _ in ()).throw(ValueError('different surface')))
    monkeypatch.setattr('autolife_shelf_grasp_306.bottle.object_points',lambda *a:(np.zeros((40,3)),None,None))
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_action_worker.validate_surface',lambda *a:1.)
    d,s,*_=v.observe('id',b,np.eye(4),position_only=True)
    assert np.allclose(d[:3,3],[.015,0,0],atol=1e-8) and s['method']=='shape_position'
    with pytest.raises(ValueError,match='朝向'):v.observe('id',b,np.eye(4),position_only=False)
