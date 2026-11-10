"""保存姿态只关联初始记录；相机不可用/物体被遮挡都不触发识别。"""
import hashlib
from types import SimpleNamespace as NS
import numpy as np
import pytest
from autolife_shelf_grasp_306.visual_reference import ReferenceStore,make_binding,corrected_tcp
from control_center_joint_control.joint_definitions import JOINT_LIMITS


def joints():return {n:max(lo,min(hi,0.)) for n,(lo,hi) in JOINT_LIMITS.items()}


def fixture(tmp_path):
    model=tmp_path/'robot.urdf';model.write_text('model')
    old=np.eye(4);old[:3,3]=[.5,.2,.8]
    tcp={}
    for side,y in [('left',.3),('right',-.3)]:
        t=np.eye(4);t[:3,3]=[.6,y,.9];tcp['Link_'+side.title()+'_Wrist_Lower_to_Gripper']=t
    geo=NS(path=model,frames=lambda q:tcp,tool_transform=lambda side:np.eye(4))
    store=ReferenceStore(tmp_path/'library')
    meta=dict(name='农夫山泉',object_pose=old.tolist(),model_sha256=hashlib.sha256(model.read_bytes()).hexdigest())
    ident=store.save(meta,dict(image=np.zeros((20,20,3),np.uint8)))
    return store,ident,geo,old,tcp


@pytest.mark.parametrize('side',['left','right'])
@pytest.mark.parametrize('mode',['position','rigid'])
def test_saved_initial_anchor_and_relative_tcp_are_exact(tmp_path,side,mode):
    store,ident,geo,old,tcp=fixture(tmp_path);raw=(store.directory/(ident+'.npz')).read_bytes()
    b=make_binding(store,ident,side,mode,joints(),geo)
    assert np.allclose(old@np.array(b['relative_tcp']),tcp['Link_'+side.title()+'_Wrist_Lower_to_Gripper'])
    assert b['taught_object_pose']==old.tolist()
    moved=old.copy();moved[:3,3]+=[.03,.01,0]
    target=corrected_tcp(old,moved,b['relative_tcp'],mode)
    assert np.allclose(target[:3,3],tcp['Link_'+side.title()+'_Wrist_Lower_to_Gripper'][:3,3]+[.03,.01,0])
    assert raw==(store.directory/(ident+'.npz')).read_bytes()


def test_supplement_position_never_replaces_saved_initial_anchor(tmp_path):
    store,ident,geo,old,_=fixture(tmp_path)
    meta,_=store.load(ident);meta['object_pose']=np.eye(4).tolist();meta['object_pose'][0][3]=1.2
    child=store.add_view(ident,meta,{})
    assert make_binding(store,ident,'right','position',joints(),geo)['taught_object_pose']==old.tolist()
    with pytest.raises(ValueError,match='物体编号'):make_binding(store,child,'right','position',joints(),geo)


def test_worker_compatibility_does_not_capture_recognize_or_refresh(tmp_path):
    from autolife_shelf_grasp_306.visual_action_worker import VisualActions
    store,ident,geo,old,_=fixture(tmp_path)
    class Forbidden:
        def __getattr__(self,name):pytest.fail('Unexpected hardware/perception access: '+name)
    v=VisualActions.__new__(VisualActions);v.store=store;v.geometry=geo;v.run=tmp_path;v.robot=Forbidden();v.perception=Forbidden()
    v.snapshot=v.sample=v.observe=v.stable=lambda *a,**kw:pytest.fail('Saving must not observe the scene')
    r=v.handle(dict(action='visual_bind',visual_id='save',reference=ident,side='right',mode='position',pose=joints()))
    assert r['binding']['taught_object_pose']==old.tolist() and r['diagnostics']['method']=='stored_initial_reference'


def test_ui_binding_no_worker_request_and_survives_camera_unavailable(tmp_path,monkeypatch):
    from PyQt5 import QtWidgets as Q
    from autolife_shelf_grasp_306.visual_action_panel import VisualPanel
    from autolife_shelf_grasp_306.pose_workspace_data import PoseStore,frame,compose
    from autolife_shelf_grasp_306.teaching_library import TeachingLibrary
    app=Q.QApplication.instance() or Q.QApplication([])
    store,ident,geo,old,_=fixture(tmp_path)
    monkeypatch.setattr('autolife_shelf_grasp_306.pose_geometry.ModelGeometry',lambda _:geo)
    workspace=NS(owner=NS(),available=lambda:None,studio=NS(cfg={'urdf_path':str(geo.path)},library=NS(directory=store.directory.parent),
        visual=NS(busy=False,request=lambda *a,**kw:pytest.fail('No vision request when saving'))))
    p=VisualPanel(workspace);poses=PoseStore(store.directory.parent);results=[]
    try:
        p.reference.setCurrentIndex(p.reference.findData(ident))
        def saved(bound):poses.save('持瓶',joints(),visual_binding=bound);results.append(bound)
        p.bind(joints(),saved)
        assert len(results)==1 and poses.binding('持瓶')['reference']==ident
        row=frame('持瓶',poses.read()['持瓶']);row['visual_binding']=poses.binding('持瓶')
        library=TeachingLibrary(store.directory.parent);library.save(compose('拿水瓶',[row],70))
        assert library.records()['拿水瓶']['visual_bindings']['0']==results[0]
        p.reference.setCurrentIndex(0);p.bind(joints(),results.append);assert results[-1] is None
    finally:p.close();app.processEvents()


def test_invalid_pose_or_changed_model_does_not_save_wrong_transform(tmp_path):
    store,ident,geo,_,_=fixture(tmp_path)
    with pytest.raises(ValueError):make_binding(store,ident,'right','position',{},geo)
    with pytest.raises(ValueError):make_binding(store,ident,'bad','position',joints(),geo)
    geo.path.write_text('another model')
    with pytest.raises(ValueError,match='模型'):make_binding(store,ident,'right','position',joints(),geo)
