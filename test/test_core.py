import json
import numpy as np
import pytest
from autolife_shelf_grasp_306.core import match, ProductStore, unit, stages, execution_blockers, crop_object,corrected_target


def test_target_correction_moves_toward_robot_right_shoulder():
    center=np.array([.5,.2,-.1]);original=center.copy()
    result,details=corrected_target(center,[0,15,0])
    np.testing.assert_allclose(result,center+[0,.015,0])
    np.testing.assert_equal(center,original)
    # Actual URDF shoulder origins in the planning torso frame.
    left=np.array([0.,-.14269,-.27507]);right=np.array([0.,.1426,-.27507])
    assert np.dot(result-center,right-left)>0
    assert details['target_offset_torso_mm']==[0.,15.,0.]
    assert details['target_offset_frame']=='Link_Waist_Yaw_to_Shoulder_Inner'
    np.testing.assert_allclose(corrected_target(center,[0,0,0])[0],center)
    for bad in ([51,0,0],[float('nan'),0,0],[30,0]):
        with pytest.raises(ValueError):corrected_target(center,bad)


def test_match_unknown_empty():assert match([1,0],[])['id'] is None
def test_match_ambiguous():
    p=[{'id':'a','features':[[1,0]]},{'id':'b','features':[[.999,.01]]}]
    assert match([1,0],p)['id'] is None
def test_match_distinct():
    p=[{'id':'a','features':[[1,0],[.99,.01]]},{'id':'b','features':[[0,1]]}]
    assert match([1,0],p)['id']=='a'
def test_low_similarity():assert match([0,1],[{'id':'a','features':[[1,0]]}])['id'] is None
@pytest.mark.parametrize('v',[[0,0],[float('nan'),0],[float('inf'),1]])
def test_invalid_feature(v):
    with pytest.raises(ValueError):unit(v)

def test_store_roundtrip(tmp_path):
    s=ProductStore(tmp_path);im=np.zeros((40,20,3),np.uint8)
    p=s.add('测试/水',70,220,[1,0],im);s.add('测试/水',70,220,[.9,.1],im)
    r=ProductStore(tmp_path)
    assert len(r.products)==1 and len(r.products[0]['features'])==2
    assert all((tmp_path/x).exists() for x in p['examples'])
    updated=s.add('测试/水',60,220,[1,0],im)
    assert updated['id']==p['id'] and updated['diameter_m']==.06 and len(updated['features'])==3

@pytest.mark.parametrize('d,h',[(0,200),(100,200),(70,0),(70,500)])
def test_invalid_dimensions(tmp_path,d,h):
    with pytest.raises(ValueError):ProductStore(tmp_path).add('水',d,h,[1,0],np.zeros((30,30,3),np.uint8))

def test_phases():
    out=stages([.5,.1,.2],0,{'up_torso':[0,0,1],'pregrasp_m':.12,'lift_m':.035,'withdraw_m':.2})
    assert [v[0] for v in out]==['预抓取','抓取','抬起','收回']
    np.testing.assert_allclose(out[0][1],[.38,.1,.2]);np.testing.assert_allclose(out[-1][1],[.3,.1,.235])
def test_stages_follow_yaw():
    out=stages([.5,.1,.2],np.pi/2,{'up_torso':[0,0,1],'pregrasp_m':.12,'lift_m':.035,'withdraw_m':.2})
    np.testing.assert_allclose(out[0][1],[.5,-.02,.2])
def test_default_fail_closed():assert len(execution_blockers({}))==5
def test_truthy_string_cannot_unlock():assert len(execution_blockers({'hardware_execution':'true'}))==5
def test_crop_masks_background():
    im=np.zeros((30,30,3),np.uint8);mask=np.zeros((30,30),bool);mask[5:25,5:25]=True
    crop=crop_object(im,mask);assert crop[0,0,0]==127 and crop[6,6,0]==0
