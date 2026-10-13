import numpy as np
import pytest
from autolife_shelf_grasp_306.geometry_memory import GeometryMemory
from autolife_shelf_grasp_306.unregistered import estimate_temporary,combine_product
from autolife_shelf_grasp_306.bottle import planning_product


def item():return dict(feature=np.array([1.,0]),semantic_label='white tissue')
def product():return dict(id='x',name='tissue',geometry='solid',temporary=True,dimension_frames=5,
                          diameter_m=.06,height_m=.12,depth_m=.05,dimension_source='multi_frame_rgbd')


def test_only_full_multiframe_geometry_is_remembered(tmp_path):
    m=GeometryMemory(tmp_path/'memory.json');p=product()
    m.remember(item(),p,dict(image_cropped=True),100)
    assert m.find(item(),101) is None
    m.remember(item(),{**p,'dimension_frames':1},{},100)
    assert m.find(item(),101) is None
    m.remember(item(),p,{},100)
    assert m.find(item(),101)['height_m']==.12
    assert m.find(item(),1901) is None
    assert m.find(item(),99) is None
    assert m.find({**item(),'semantic_label':'yellow tissue'},101) is None
    assert m.find({**item(),'feature':np.array([0.,1.])},101) is None


def test_conflicting_sizes_do_not_supply_a_hidden_extent(tmp_path):
    m=GeometryMemory(tmp_path/'memory.json')
    m.remember(item(),product(),{},100)
    m.remember({**item(),'feature':np.array([.95,.31225])},{**product(),'height_m':.25},{},101)
    assert m.find({**item(),'feature':np.array([.986,.166])},102) is None


def test_partial_appearance_matches_without_lowering_identity_threshold(tmp_path):
    m=GeometryMemory(tmp_path/'memory.json')
    m.remember({**item(),'partial_features':[[.9,.43589]]},product(),{},100)
    assert m.find({**item(),'feature':np.array([.9,.43589])},101) is not None
    assert m.find({**item(),'feature':np.array([0.,1.])},101) is None


def test_partial_views_crop_actual_pixels_not_filled_unknown():
    from autolife_shelf_grasp_306.geometry_memory import partial_views
    image=np.full((120,120,3),100,np.uint8);mask=np.zeros((120,120),bool);mask[20:100,30:90]=True
    views=partial_views(image,mask)
    assert len(views)==8 and all(v.shape[0]<88 for v in views)
    assert all(set(np.unique(v))<={100,127} for v in views)


def test_partial_uses_live_depth_not_cached_coordinates(monkeypatch):
    from autolife_shelf_grasp_306 import object_geometry
    mask=np.zeros((120,120),bool);mask[20:,30:90]=True
    reference={**product(),'reference_age_s':12}
    def estimate(*args):return np.array([.55,-.1,.2]),dict(observed_height_mm=60.,geometry='solid',points=800)
    monkeypatch.setattr(object_geometry,'estimate_object',estimate)
    p=product();center,stats=estimate_temporary({},mask,np.eye(4),[0,0,1],p,{},reference)
    np.testing.assert_allclose(center,[.55,-.1,.2])
    assert stats['image_cropped'] and stats['dimension_source']=='recent_full_view_with_live_depth'
    result=combine_product(p,[(center,stats)]*5)
    collision=planning_product(result,dict(observed_height_mm=60.,image_cropped=True))
    assert collision['height_m']==.06 and collision['collision_height_m']==pytest.approx(.18)
    assert result['geometry_reference']=='recent_full_view'


def test_unknown_or_side_cropped_object_still_requires_evidence():
    mask=np.zeros((120,120),bool);mask[20:,30:90]=True
    with pytest.raises(ValueError,match='完整尺寸记录'):
        estimate_temporary({},mask,np.eye(4),[0,0,1],product(),{})
    mask[:,0]=True
    with pytest.raises(ValueError,match='左右边界'):
        estimate_temporary({},mask,np.eye(4),[0,0,1],product(),{},{**product(),'reference_age_s':1})


def test_a_larger_visible_object_does_not_reuse_smaller_reference(monkeypatch):
    from autolife_shelf_grasp_306 import object_geometry
    monkeypatch.setattr(object_geometry,'estimate_object',lambda *a:(np.zeros(3),dict(observed_height_mm=200.)))
    mask=np.zeros((120,120),bool);mask[20:,30:90]=True
    with pytest.raises(ValueError,match='尺寸超过'):
        estimate_temporary({},mask,np.eye(4),[0,0,1],product(),{},{**product(),'reference_age_s':1})
