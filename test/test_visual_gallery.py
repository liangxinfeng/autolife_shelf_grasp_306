from types import SimpleNamespace as NS
import numpy as np
import pytest
from PyQt5 import QtWidgets as Q,QtCore as C
from autolife_shelf_grasp_306.visual_reference import ReferenceStore
from autolife_shelf_grasp_306.visual_reference_gallery import ReferenceGallery


def memory(tmp_path):
    store=ReferenceStore(tmp_path);meta=dict(name='测试物体',object_pose=np.eye(4).tolist(),created=1794300000.)
    image=np.zeros((100,160,3),np.uint8);image[20:80,60:100]=[30,190,90];mask=np.zeros((100,160),bool);mask[20:80,60:100]=True
    arrays=dict(image=image,mask=mask,cloud=np.ones((30,3)));root=store.save(meta,arrays)
    child=store.add_view(root,dict(meta,created=1794300100.,metric_registered=False),arrays)
    return store,root,child


def test_remove_supplement_preserves_initial_and_poses(tmp_path):
    store,root,child=memory(tmp_path);original=(store.directory/(root+'.npz')).read_bytes()
    assert store.view_ids(root)==[root,child]
    store.remove_view(root,child)
    assert store.view_ids(root)==[root] and len(store.views(root))==1
    assert (store.directory/(root+'.npz')).read_bytes()==original
    assert store.load(child)[0]['parent_reference']==root  # immutable history retained
    with pytest.raises(ValueError):store.remove_view(root,root)
    with pytest.raises(ValueError):store.remove_view(root,child)


def test_wrong_object_view_cannot_be_removed(tmp_path):
    store,root,child=memory(tmp_path);other=store.save(dict(name='其他',object_pose=np.eye(4).tolist()),{})
    with pytest.raises(ValueError):store.remove_view(other,child)
    assert store.view_ids(root)==[root,child]


def panel(store):
    p=Q.QWidget();p.store=store;p.workspace=NS(available=lambda:None,studio=NS(visual=NS(busy=False)));p.token='stale';p.adding_view='stale';return p


def test_gallery_initial_marker_preview_toggle_and_delete(tmp_path,monkeypatch):
    app=Q.QApplication.instance() or Q.QApplication([]);store,root,child=memory(tmp_path);p=panel(store);g=ReferenceGallery(p,root)
    try:
        assert g.list.count()==2 and '初始帧' in g.list.item(0).text() and not g.remove.isEnabled()
        assert not g.pix.isNull();g.overlay.setChecked(False);assert not g.pix.isNull()
        g.list.setCurrentRow(1);assert g.remove.isEnabled() and '已用于识别' in g.info.text()
        monkeypatch.setattr(Q.QMessageBox,'question',lambda *a:Q.QMessageBox.Yes)
        g.delete_selected();assert g.list.count()==1 and not g.remove.isEnabled() and g.changed
        assert p.token is None and p.adding_view is None and len(store.views(root))==1
    finally:g.close();p.close();app.processEvents()


def test_cancel_or_inflight_visual_never_deletes(tmp_path,monkeypatch):
    app=Q.QApplication.instance() or Q.QApplication([]);store,root,child=memory(tmp_path);p=panel(store);g=ReferenceGallery(p,root)
    try:
        g.list.setCurrentRow(1);monkeypatch.setattr(Q.QMessageBox,'question',lambda *a:Q.QMessageBox.No)
        g.delete_selected();assert len(store.views(root))==2
        p.workspace.studio.visual.busy=True;monkeypatch.setattr(Q.QMessageBox,'question',lambda *a:Q.QMessageBox.Yes)
        g.delete_selected();assert len(store.views(root))==2 and '等待' in g.note.text()
    finally:g.close();p.close();app.processEvents()


def test_broken_supplement_can_be_removed_without_deleting_anchor(tmp_path,monkeypatch):
    app=Q.QApplication.instance() or Q.QApplication([]);store,root,child=memory(tmp_path)
    (store.directory/(child+'.npz')).write_bytes(b'broken')
    p=panel(store);g=ReferenceGallery(p,root)
    try:
        g.list.setCurrentRow(1);assert '读取失败' in g.list.item(1).text()
        monkeypatch.setattr(Q.QMessageBox,'question',lambda *a:Q.QMessageBox.Yes)
        g.delete_selected();assert len(store.views(root))==1
    finally:g.close();p.close();app.processEvents()


def test_worker_reloads_views_after_delete_even_with_cached_features(tmp_path,monkeypatch):
    from test_visual_followup import fixture
    from autolife_shelf_grasp_306.visual_action_worker import VisualActions
    from autolife_shelf_grasp_306 import visual_registration as vr
    m,a=fixture();a.update(image=np.zeros((20,20,3),np.uint8),mask=np.ones((20,20),bool))
    store=ReferenceStore(tmp_path);root=store.save(m,a);child=store.add_view(root,m,a)
    v=object.__new__(VisualActions);v.store=store;v.reference_cache={};v.cfg={};v.perception=None;v.products=None
    monkeypatch.setattr(vr,'image_features',lambda *a,**k:(_ for _ in ()).throw(ValueError('use stored')))
    stats=dict(rms_mm=0.,inliers=10,pixels=[[10,10]],angle_resolution_deg=0.)
    sizes=[]
    def seed(banks,current):sizes.append(len(banks));return np.eye(4),dict(stats),banks[0][0],banks[0][1]
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_multiview.seed_match',seed)
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_action_worker.features',lambda *args,**kw:a)
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_action_worker.register_views',lambda banks,current:seed(banks,current))
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_action_worker.validate_surface',lambda *args:1.)
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_box_selection.projected_box',lambda *args:[0,0,1,1])
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_box_selection.segment_box',lambda *args,**kw:dict(mask=a['mask']))
    monkeypatch.setattr('autolife_shelf_grasp_306.bottle.object_points',lambda *args:(a['cloud'],None,None))
    v.observe(root,dict(image=a['image']),np.eye(4));assert sizes[-1]==2
    store.remove_view(root,child)
    v.observe(root,dict(image=a['image']),np.eye(4));assert sizes[-1]==1
    assert v.reference_cache[('resolved',root)][0]==(root,)
