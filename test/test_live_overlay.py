import json
import cv2
import numpy as np
import pytest
from autolife_shelf_grasp_306.live_overlay import LiveOverlay,encode_packet,decode_packet,transform_points
from autolife_shelf_grasp_306.grounded_sam2 import detection_query,normalize_prompt,category_label_matches


@pytest.mark.parametrize('name',['盒子','瓶子','纸巾','袋子','鼠标'])
def test_bare_name_expands_without_color(name):
    text,threshold=detection_query(name,{'grounded_box_threshold':.35})
    assert threshold==.25 and len(text.split('. '))>=2
    assert not any(c in text for c in ('green','white','red','blue'))


@pytest.mark.parametrize('name',['白色盒子','绿色盒子','white box','green bottle'])
def test_explicit_color_is_never_removed(name):
    text,threshold=detection_query(name,{'grounded_box_threshold':.35})
    assert text==normalize_prompt(name) and threshold==.35


def fixture():
    rng=np.random.default_rng(23);im=np.full((240,400,3),40,np.uint8)
    im[60:150,100:210]=rng.integers(25,240,(90,110,3),dtype=np.uint8)
    item=dict(id=3,name='盒子',box=[100,60,210,150],mask_contours=[[[100,60],[210,60],[210,150],[100,150]]])
    frame=dict(image=im,frame_id=1,received=100.)
    return frame,item


def test_tracks_mask_box_and_click_inverse_without_mutating_reference():
    frame,item=fixture();saved=json.dumps(item);overlay=LiveOverlay();overlay.set(frame,[item],8)
    first=overlay.render(frame);assert len(first['items'])==1
    move=np.float32([[1,0,9],[0,1,5]])
    next_frame=dict(image=cv2.warpAffine(frame['image'],move,(400,240)),frame_id=2,received=100.08)
    out=overlay.render(next_frame);assert out['revision']==8 and out['frame_id']==2
    box=out['items'][0];assert np.allclose(box['box'],[109,65,219,155],atol=.5)
    assert np.allclose(transform_points([[159,105]],np.array(box['display_to_reference'])),[[150,100]],atol=.5)
    assert json.dumps(item)==saved


def test_failed_or_stale_track_hidden_without_losing_rgb_packet():
    f,i=fixture();o=LiveOverlay();o.set(f,[i],1);o.render(f)
    f2={**f,'frame_id':2,'received':102.}
    out=o.render(f2);assert out['lost']==1 and out['items']==[]
    ok,jpeg=cv2.imencode('.jpg',f2['image']);assert ok
    raw,meta=decode_packet(encode_packet(jpeg,out))
    assert meta['frame_id']==2 and cv2.imdecode(np.frombuffer(raw,np.uint8),1).shape==(240,400,3)


def test_path_projection_independent_of_object_flow_and_feedback_loss():
    f,i=fixture();o=LiveOverlay();o.set(f,[i],4)
    o.set_path(lambda frame:[[float(frame['frame_id']),2.]],4)
    assert o.render(f)['path']==[[1.,2.]]
    def stale(frame):raise ValueError('feedback expired')
    o.set_path(stale,4);out=o.render({**f,'frame_id':2,'received':100.08})
    assert out['path']==[] and out['path_status']=='unavailable' and out['items']


def test_rename_keeps_propagation_and_new_scene_clears_path():
    f,i=fixture();o=LiveOverlay();o.set(f,[i],1);o.render(f)
    o.rename([{**i,'name':'商品盒'}],2);out=o.render({**f,'frame_id':2,'received':100.08})
    assert out['revision']==2 and out['items'][0]['name']=='商品盒'
    o.set_path(lambda frame:[[1.,2.]],2);o.set(f,[i],3)
    assert o.render(f)['path']==[]


def test_category_expansion_rejects_ambiguous_fragments():
    assert not category_label_matches('纸巾','pack')
    assert not category_label_matches('盒子','packaged')
    assert category_label_matches('盒子','carton packaged box')
    assert category_label_matches('瓶子','water bottle beverage')
    assert category_label_matches('白色盒子','white box')


def test_failed_backward_flow_hides_only_overlay(monkeypatch):
    f,i=fixture();o=LiveOverlay();o.set(f,[i],1);o.render(f)
    calls=[]
    def flow(a,b,pts,*args,**kwargs):
        calls.append(1)
        return (pts.copy(),np.ones((len(pts),1),np.uint8),None) if len(calls)==1 else (None,None,None)
    monkeypatch.setattr(cv2,'calcOpticalFlowPyrLK',flow)
    out=o.render({**f,'received':100.08,'frame_id':2})
    assert out['items']==[] and out['lost']==1
