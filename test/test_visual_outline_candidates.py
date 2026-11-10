from types import SimpleNamespace as NS
import numpy as np
import pytest
from test_visual_shape_position import scene,CFG
from autolife_shelf_grasp_306.visual_shape_position import position_from_shape
from autolife_shelf_grasp_306.visual_box_selection import dominant_component


def test_low_silhouette_score_alone_cannot_reject_valid_depth():
    a=scene();b=scene(10)
    # 分割漏掉内部一片，但上下左右的实际深度对应仍充分且一致。
    b['mask'][90:150,150:190]=False
    delta,stats=position_from_shape(a,np.eye(4),b,np.eye(4),CFG)
    assert stats['silhouette_overlap']<.78
    assert np.allclose(delta[:3,3],[.015,0,0],atol=.001)


def test_disconnected_pen_is_not_part_of_bottle_bbox():
    mask=scene()['mask'];mask[20:25,50:90]=True
    clean=dominant_component(mask)
    assert clean.sum()==7200 and not clean[22,55] and clean[100,150]


def test_comparable_components_not_arbitrarily_discarded():
    mask=np.zeros((200,200),bool);mask[20:60,20:60]=True;mask[100:140,100:140]=True
    assert np.array_equal(mask,dominant_component(mask))


def test_geometry_selects_alternative_instead_of_largest_box(monkeypatch):
    from autolife_shelf_grasp_306.visual_action_worker import VisualActions
    v=VisualActions.__new__(VisualActions)
    good=dict(mask=np.ones((4,4)),tag='bottle')
    main=dict(mask=np.zeros((4,4)),tag='bottle+pen',_alternatives=[good])
    v.perception=None
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_box_selection.segment_box',lambda *a:main)
    def locate(*args):
        item=args[4]
        if item['tag']!='bottle':raise ValueError('depth inconsistent')
        return np.eye(4),dict(rms_mm=1.),item,{},np.zeros((40,3))
    v.locate_mask=locate
    r=v.observe_box([(dict(object_pose=np.eye(4).tolist()),{})],[],{},np.eye(4),[],True,{})
    assert r[2]['tag']=='bottle' and r[1]['segmentation_candidates']==2 and r[1]['geometry_candidates']==1


def test_conflicting_valid_candidates_cannot_choose_convenient_action(monkeypatch):
    from autolife_shelf_grasp_306.visual_action_worker import VisualActions
    v=VisualActions.__new__(VisualActions);v.perception=None
    other=dict(mask=np.ones((4,4)),offset=.05)
    main=dict(mask=np.ones((4,4)),offset=0.,_alternatives=[other])
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_box_selection.segment_box',lambda *a:main)
    def locate(*args):
        item=args[4];delta=np.eye(4);delta[0,3]=item['offset']
        return delta,dict(rms_mm=1.),item,{},np.zeros((40,3))
    v.locate_mask=locate
    with pytest.raises(ValueError,match='不同位置'):
        v.observe_box([(dict(object_pose=np.eye(4).tolist()),{})],[],{},np.eye(4),[],True,{})
