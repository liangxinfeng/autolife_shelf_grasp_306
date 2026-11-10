from types import SimpleNamespace as NS
import numpy as np
import pytest
from PyQt5 import QtWidgets as Q,QtCore as C,QtGui as G,QtTest
from autolife_shelf_grasp_306.visual_box_selection import pixel_box,choose_box_mask


def test_box_coordinates_support_reverse_drag_and_edges():
    assert np.allclose(pixel_box([.8,.9,.2,.1],(100,200,3)),[40,10,160,90])
    assert np.allclose(pixel_box([0,0,1,1],(100,200,3)),[0,0,200,100])
    for bad in ([.1,.1,.101,.102],[-.1,0,1,1],[0,0,float('nan'),1]):
        with pytest.raises(ValueError):pixel_box(bad,(100,200,3))


def test_box_prefers_whole_bottle_over_small_label_and_rejects_background():
    full=np.zeros((100,100),bool);full[15:85,35:65]=1
    label=np.zeros_like(full);label[40:60,35:65]=1
    background=np.ones_like(full)
    mask,_=choose_box_mask([label,background,full],[.97,.99,.80],np.array([30,10,70,90]))
    assert np.array_equal(mask,full)
    with pytest.raises(ValueError):choose_box_mask([background],[.99],np.array([30,10,70,90]))


def test_drag_letterbox_coordinates_no_point_click():
    from autolife_shelf_grasp_306.visual_box_view import BoxImageView
    app=Q.QApplication.instance() or Q.QApplication([]);w=BoxImageView();w.resize(400,400);w.pix=G.QPixmap(400,200);w.pix.fill(C.Qt.white);w.shape=[200,400,3];w.selection_enabled=True;w.show();app.processEvents()
    boxes=[];clicks=[];w.boxSelected.connect(boxes.append);w.clicked.connect(lambda *a:clicks.append(a))
    QtTest.QTest.mousePress(w,C.Qt.LeftButton,pos=C.QPoint(300,250));QtTest.QTest.mouseRelease(w,C.Qt.LeftButton,pos=C.QPoint(100,150))
    assert np.allclose(boxes[-1],[.25,.25,.75,.75]) and not clicks
    w.selection_enabled=False;QtTest.QTest.mouseClick(w,C.Qt.LeftButton,pos=C.QPoint(200,200));assert len(boxes)==1
    w.close();app.processEvents()


def test_preview_does_not_save_until_confirm_and_cannot_duplicate(tmp_path,monkeypatch):
    from autolife_shelf_grasp_306.visual_action_worker import VisualActions
    from autolife_shelf_grasp_306.visual_reference import ReferenceStore
    from test_visual_followup import fixture
    m,a=fixture();image=np.zeros((100,100,3),np.uint8);mask=np.zeros((100,100),bool);mask[10:90,30:70]=1
    frame=dict(image=image,depth=np.ones((100,100)),K=np.eye(3),D=np.zeros(5))
    v=object.__new__(VisualActions);v.run=tmp_path;v.store=ReferenceStore(tmp_path);v.cfg={};v.geometry=NS(root='base');v.frozen=None;v.pending_view=None;v.reference_cache={};v.perception=None;v.products=None
    v.robot=NS(refresh_measured=lambda:None,base_sample=lambda:[0,0,0]);v.sample=lambda:(frame,{},np.eye(4));v.stable=lambda *a:None;v.model_hash=lambda:'test';v.mount=lambda *a:np.eye(4);v.check_mount=lambda *a:None
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_box_selection.segment_box',lambda *args:dict(mask=mask,box=[30,10,70,90]))
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_action_worker.features',lambda *args,**kw:{k:a[k] for k in ('points','pixels','descriptors')})
    monkeypatch.setattr('autolife_shelf_grasp_306.bottle.object_points',lambda *args:(a['cloud'],None,None))
    frozen=v.handle(dict(action='visual_freeze',visual_id='freeze'))
    req=dict(action='visual_remember',visual_id='box',token=frozen['token'],box=[.3,.1,.7,.9],preview_only=True,name='物体')
    preview=v.handle(req);assert not v.store.list()
    old=preview['preview_token'];preview=v.handle(req)
    with pytest.raises(ValueError):v.handle(dict(action='visual_commit_view',visual_id='save',preview_token=old))
    command=dict(action='visual_commit_view',visual_id='save',preview_token=preview['preview_token'])
    saved=v.handle(command);assert len(v.store.list())==1 and saved['item']['mask_contours']
    with pytest.raises(ValueError):v.handle(command)
    root=saved['reference'];before=(v.store.directory/(root+'.npz')).read_bytes()
    frozen=v.handle(dict(action='visual_freeze',visual_id='more'))
    preview=v.handle(dict(req,action='visual_add_view',reference=root,token=frozen['token']))
    assert len(v.store.views(root))==1
    saved=v.handle(dict(command,preview_token=preview['preview_token']))
    assert saved['reference']==root and len(v.store.views(root))==2
    assert (v.store.directory/(root+'.npz')).read_bytes()==before
