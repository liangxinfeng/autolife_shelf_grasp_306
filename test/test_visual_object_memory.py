import numpy as np
import pytest
from types import SimpleNamespace
from autolife_shelf_grasp_306.visual_object_memory import select_match,ObjectAppearanceMemory,AmbiguousObject


def test_back_view_does_not_get_averaged_with_front():
    result=select_match([[0,1]],[[1,0],[0,1]])
    assert result==(0,1,1.)


def test_two_instances_ambiguous_but_two_views_are_not():
    with pytest.raises(AmbiguousObject):select_match([[1,0],[.99,.01]],[[1,0]])
    assert select_match([[1,0]],[[1,0],[.99,.01]]) is not None


def test_other_object_and_invalid_features_not_accepted():
    assert select_match([[0,1]],[[1,0]]) is None
    for value in ([[0,0]],[[np.nan,0]]):
        with pytest.raises(ValueError):select_match(value,[[1,0]])


def test_active_view_delete_invalidates_features_without_mutating_anchor():
    calls=[]
    def embed(crops):calls.append(len(crops));return np.tile([[1.,0]],(len(crops),1))
    p=SimpleNamespace(embed=embed);store=SimpleNamespace(view_ids=lambda ident:ids)
    image=np.zeros((20,20,3),np.uint8);mask=np.ones((20,20),bool)
    views=[({'object_pose':np.eye(4).tolist()},{'image':image,'mask':mask})]*2
    ids=['root','view'];m=ObjectAppearanceMemory(p,{})
    m.reference_features(store,'root',views);m.reference_features(store,'root',views)
    ids=['root'];m.reference_features(store,'root',views[:1])
    assert calls==[2,1]
    assert np.array_equal(views[0][0]['object_pose'],np.eye(4))


def test_appearance_only_supplement_never_supplies_metric_pose():
    image=np.zeros((20,20,3),np.uint8);mask=np.ones((20,20),bool)
    item={'image':image,'mask':mask};calls=[]
    def embed(crops):
        calls.append(len(crops))
        return np.array([[1.,0],[0,1]]) if len(crops)==2 else np.array([[0.,1]])
    p=SimpleNamespace(embed=embed,segmenter=SimpleNamespace(proposals=lambda *args,**kwargs:[dict(mask=mask,box=[0,0,20,20])]))
    views=[({'object_pose':np.eye(4).tolist()},item),({'metric_registered':False,'camera_pose':'must not read'},item)]
    store=SimpleNamespace(view_ids=lambda ident:['root','view'])
    match=ObjectAppearanceMemory(p,{}).recognize(store,'root',views,{'image':image})
    assert match[1]['recognition_view']=='补录视角1'
    assert 'camera_pose' not in match[1]


def test_dino_path_does_not_require_full_scene_sift(monkeypatch):
    from autolife_shelf_grasp_306.visual_action_worker import VisualActions
    import autolife_shelf_grasp_306.visual_registration as registration
    monkeypatch.setattr(registration,'features',lambda *a,**k:pytest.fail('No full scene SIFT'))
    worker=VisualActions.__new__(VisualActions);worker.cfg={};worker.perception=SimpleNamespace(embed=lambda x:x)
    worker.store=SimpleNamespace(views=lambda ident:[])
    worker.appearance_memory=SimpleNamespace(recognize=lambda *a:({'mask':np.ones((4,4))},{'recognition_method':'dinov2_object'}))
    worker.locate_mask=lambda *a:('located',a[-1])
    assert worker.observe('root',{},np.eye(4),position_only=True)[0]=='located'


def test_ambiguous_identity_never_falls_back_to_one_texture_match():
    from autolife_shelf_grasp_306.visual_action_worker import VisualActions
    def ambiguous(*a):raise AmbiguousObject('two objects')
    worker=VisualActions.__new__(VisualActions);worker.cfg={};worker.perception=SimpleNamespace(embed=lambda x:x)
    worker.store=SimpleNamespace(views=lambda ident:[]);worker.appearance_memory=SimpleNamespace(recognize=ambiguous)
    worker.observe_texture=lambda *a:pytest.fail('Ambiguity bypassed')
    with pytest.raises(AmbiguousObject):worker.observe('root',{},np.eye(4))
