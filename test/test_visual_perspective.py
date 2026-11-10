"""视点变化的轮廓方向对齐、一次再分割，以及失败帧诊断。"""
from types import SimpleNamespace as NS
import cv2
import numpy as np
import pytest
from test_visual_shape_position import scene,CFG
from autolife_shelf_grasp_306.visual_shape_position import position_from_shape,_estimate


@pytest.mark.parametrize('angle',[-11.,-10.,10.,11.])
def test_same_object_with_camera_roll_is_not_a_different_shape(angle):
    from scipy.spatial.transform import Rotation
    a=scene();b=scene();warp=cv2.getRotationMatrix2D((160,120),angle,1.)
    b['mask']=cv2.warpAffine(a['mask'].astype('uint8'),warp,(320,240),flags=cv2.INTER_NEAREST)>0
    b['depth']=np.where(b['mask'],600.,900.)
    T=np.eye(4);T[:3,:3]=Rotation.from_euler('z',angle,degrees=True).as_matrix()
    with pytest.raises(ValueError,match='轮廓|深度'):_estimate(a,np.eye(4),b,T,CFG)
    delta,stats=position_from_shape(a,np.eye(4),b,T,CFG)
    assert np.linalg.norm(delta[:3,3])<.002 and stats['silhouette_alignment']=='principal_axes'


def test_camera_roll_and_translation_preserve_known_world_offset():
    from scipy.spatial.transform import Rotation
    a=scene();b=scene();warp=cv2.getRotationMatrix2D((160,120),10.,1.)
    b['mask']=cv2.warpAffine(a['mask'].astype('uint8'),warp,(320,240),flags=cv2.INTER_NEAREST)>0
    b['depth']=np.where(b['mask'],600.,900.)
    T=np.eye(4);T[:3,:3]=Rotation.from_euler('z',10,degrees=True).as_matrix();T[:3,3]=[.04,.03,.02]
    d,_=position_from_shape(a,np.eye(4),b,T,CFG)
    assert np.allclose(d[:3,3],T[:3,3],atol=.002)


def test_single_mask_speckle_does_not_change_whole_object_bbox():
    a=scene();b=scene(10);b['mask'][20,20]=True
    d,_=position_from_shape(a,np.eye(4),b,np.eye(4),CFG)
    assert np.allclose(d[:3,3],[.015,0,0],atol=.001)


def test_shape_alignment_does_not_allow_occluded_half_object():
    a=scene();b=scene();b['mask'][120:]=False
    with pytest.raises(ValueError):position_from_shape(a,np.eye(4),b,np.eye(4),CFG)


def test_local_segmentation_recovers_without_recording_new_reference(tmp_path,monkeypatch):
    from autolife_shelf_grasp_306.visual_action_worker import VisualActions
    a=scene();a.update(image=np.zeros((240,320,3),np.uint8),points=np.zeros((10,3)),pixels=np.zeros((10,2)),descriptors=np.zeros((10,128)),cloud=np.zeros((40,3)))
    meta=dict(camera_pose=np.eye(4).tolist(),object_pose=np.eye(4).tolist(),created=1)
    v=VisualActions.__new__(VisualActions);v.cfg=CFG;v.perception=None;v.reference_cache={};v.run=tmp_path
    v.store=NS(views=lambda _: [(meta,a)],view_ids=lambda _:['first'])
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_multiview.resolve_views',lambda x:x)
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_registration.image_features',lambda *a,**k:(_ for _ in ()).throw(ValueError('use stored')))
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_multiview.seed_match',lambda *args:(np.eye(4),dict(inliers=20),meta,a))
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_box_selection.projected_box',lambda *a:[.3,.2,.7,.8])
    calls=[]
    def segment(p,frame,box):
        calls.append((id(frame),box));mask=a['mask'].copy()
        if len(calls)==1:mask[120:]=False
        return dict(mask=mask,box=[130,60,190,180])
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_box_selection.segment_box',segment)
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_action_worker.features',lambda *a,**k:{})
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_action_worker.register_views',lambda *a:(_ for _ in ()).throw(ValueError('different surface')))
    monkeypatch.setattr('autolife_shelf_grasp_306.bottle.object_points',lambda *a:(np.zeros((40,3)),None,None))
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_action_worker.validate_surface',lambda *a:1.)
    delta,stats,*_=v.observe('id',a,np.eye(4),position_only=True)
    assert stats['segmentation_attempts']==2 and np.allclose(delta,np.eye(4))
    assert calls[0][0]==calls[1][0] and calls[1][1][0]<calls[0][1][0]
    assert list(tmp_path.iterdir())==[]
    # 两次都缺失下半部分时仍不输出纠偏，不把失败当成零位移。
    calls.clear()
    def partial(*args):
        calls.append(1);mask=a['mask'].copy();mask[120:]=False;return dict(mask=mask)
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_box_selection.segment_box',partial)
    with pytest.raises(ValueError):v.observe('id',a,np.eye(4),position_only=True)
    assert len(calls)==2
    report=tmp_path/'visual_observation_failed.npz'
    assert report.is_file()
    with np.load(report,allow_pickle=False) as z:assert str(z['reference'])=='id' and z['mask'].sum()<a['mask'].sum()
    assert list(tmp_path.iterdir())==[report]
