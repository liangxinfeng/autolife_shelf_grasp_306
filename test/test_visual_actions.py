"""参照隔离、真实坐标变换、动作兼容和取消链路；不连接机器人。"""
import json
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from autolife_shelf_grasp_306.visual_reference import *
from autolife_shelf_grasp_306.visual_registration import fit,consensus,register
from autolife_shelf_grasp_306.teaching_motion import normalize_draft,compile_action
from autolife_shelf_grasp_306.teaching_library import TeachingLibrary
from autolife_shelf_grasp_306.pose_workspace_data import PoseStore,frame,compose
from control_center_joint_control.joint_definitions import JOINT_LIMITS

def pose():return {n:max(lo,min(hi,0.)) for n,(lo,hi) in JOINT_LIMITS.items()}
def bound():return dict(reference='a'*64,side='right',mode='position',relative_tcp=np.eye(4).tolist(),taught_object_pose=np.eye(4).tolist())

@pytest.mark.parametrize('mode',['position','rigid'])
def test_transform_preserves_relative_pose_and_does_not_mutate(mode):
    old=np.eye(4);old[:3,3]=[.3,.2,.6]
    delta=np.eye(4);delta[:3,:3]=Rotation.from_euler('z',12,degrees=True).as_matrix();delta[:3,3]=[.02,-.01,.003]
    local=np.eye(4);local[:3,3]=[.1,0,0]
    saved=local.copy();new=delta@old;r=corrected_tcp(old,new,local,mode)
    assert np.array_equal(local,saved)
    if mode=='rigid':assert np.allclose(np.linalg.inv(new)@r,local)
    else:
        assert np.allclose(r[:3,:3],(old@local)[:3,:3])
        assert np.allclose(r[:3,3]-(old@local)[:3,3],new[:3,3]-old[:3,3])

@pytest.mark.parametrize('offset',[[.01,.02,0],[-.04,0,.03],[0,0,0]])
def test_ransac_rejects_outliers_and_recovers_rigid_transform(offset):
    rng=np.random.default_rng(12);a=rng.uniform(-.15,.15,(80,3));r=Rotation.from_euler('xyz',[3,-2,8],degrees=True).as_matrix()
    b=a@r.T+offset+rng.normal(0,.0003,a.shape);b[:15]=rng.uniform(1,2,(15,3))
    t,stats,ok=consensus(a,b)
    assert stats['inliers']>=60 and stats['rms_mm']<1
    assert np.allclose(t[:3,:3],r,atol=.003) and np.allclose(t[:3,3],offset,atol=.001)

def test_collinear_or_wrong_object_does_not_make_zero_correction():
    a=np.column_stack([np.linspace(0,.2,20),np.zeros((20,2))])
    with pytest.raises(ValueError):consensus(a,a)
    rng=np.random.default_rng(4)
    with pytest.raises(ValueError):consensus(rng.random((20,3)),rng.random((20,3)))

def test_descriptor_registration_and_duplicate_ambiguity():
    rng=np.random.default_rng(6);d=rng.normal(size=(50,128)).astype('f');p=rng.uniform(-.1,.1,(50,3));pixels=rng.random((50,2))
    a=dict(descriptors=d,points=p,pixels=pixels);b=dict(descriptors=d+.001,points=p+[.02,0,0],pixels=pixels)
    t,s=register(a,b);assert s['inliers']==50 and np.allclose(t[:3,3],[.02,0,0])
    b={k:np.concatenate([v,v]) for k,v in b.items()}
    b['points'][50:]+=np.array([.1,0,0])  # 两个不同位置同外观的物体；同一实体多描述不算歧义
    with pytest.raises(ValueError,match='唯一'):register(a,b)

def test_immutable_reference_hash_and_traversal(tmp_path):
    store=ReferenceStore(tmp_path);meta=dict(name='柜子',object_pose=np.eye(4).tolist())
    ident=store.save(meta,dict(points=np.ones((10,3))));assert store.load(ident)[0]==meta
    assert store.list()[0]['name']=='柜子'
    with pytest.raises(ValueError):store.load('../bad')
    (store.directory/(ident+'.npz')).write_bytes(b'bad')
    with pytest.raises(ValueError,match='改变'):store.load(ident)
    assert store.list()==[] and len(store.errors)==1

def test_library_roundtrip_binding_and_old_action(tmp_path):
    q=pose();f=frame('接近',q);f['visual_binding']=bound();d=compose('纠偏动作',[frame('准备',q),f,frame('收回',q)],70)
    lib=TeachingLibrary(tmp_path);record=lib.save(d)
    assert record['visual_bindings']=={'1':bound()}
    assert lib.records()['纠偏动作']['visual_bindings']=={'1':bound()}
    assert lib.load('纠偏动作')==d
    old=compose('旧动作',[frame('旧姿态',q)],100);lib.save(old)
    assert 'visual_bindings' not in lib.records()['旧动作'] and lib.load('旧动作')==old
    runtime=clone_without_bindings(record);runtime['keyframes'][1]['joints_deg']['Joint_Right_Gripper']=12
    assert lib.records()['纠偏动作']==record

def test_pose_snapshot_binding_atomic_and_replacement(tmp_path):
    store=PoseStore(tmp_path);q=pose();store.save('A',q,visual_binding=bound())
    assert store.binding('A')==bound() and store.read()['A']==q
    store.save('A',q,True);assert store.binding('A') is None
    store.save('A',q,True,visual_binding=bound());store.delete('A');assert store.binding('A') is None

@pytest.mark.parametrize('key',['-1','2','abc','01'])
def test_invalid_frame_binding_is_not_silently_ignored(key):
    r=compile_action(compose('a',[frame('a',pose())],100));r['visual_bindings']={key:bound()}
    with pytest.raises((ValueError,TypeError)):record_bindings(r)

def test_select_single_frame_keeps_only_its_reference():
    from autolife_shelf_grasp_306.action_studio import selected_frame_action
    f=frame('a',pose());f['visual_binding']=bound();r=compile_action(compose('a',[frame('b',pose()),f],100))
    assert selected_frame_action(r,1)['visual_bindings']=={'0':bound()}
    assert not selected_frame_action(r,0).get('visual_bindings')

def test_edited_metadata_never_silently_drops_binding(tmp_path):
    lib=TeachingLibrary(tmp_path);f=frame('a',pose());f['visual_binding']=bound();lib.save(compose('a',[f],100))
    import yaml
    doc=yaml.safe_load(lib.path.read_text());doc['teaching_editor']['a']['record_digest']='invalid';lib.path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError,match='不一致'):lib.load('a')

def test_ui_client_cancel_ignores_success_and_stale_reply(tmp_path):
    from PyQt5 import QtWidgets as Q
    from autolife_shelf_grasp_306.visual_action_client import VisualClient
    app=Q.QApplication.instance() or Q.QApplication([])
    studio=Q.QWidget();calls=[]
    studio.owner=SimpleNamespace(run=tmp_path,busy=False,motion=None,navigation=SimpleNamespace(inflight=False),ready=True,
        process=SimpleNamespace(write=lambda b:len(b)),status=SimpleNamespace(setText=lambda t:None),controls=lambda:None)
    studio.runtime=SimpleNamespace(busy=False);studio.editor=SimpleNamespace(remote_locked=lambda:False)
    studio.engine=SimpleNamespace(bridge=None);studio._acquire=lambda:calls.append('lock');studio._unlock=lambda:calls.append('unlock')
    client=VisualClient(studio);replies=[];client.request('visual_correct',replies.append,record={});ident=client.pending['id']
    client.receive(dict(visual_id='old',result={}));assert client.busy and not replies
    client.cancel();client.receive(dict(visual_id=ident,result={'record':'must not execute'}))
    assert not client.busy and 'error' in replies[0] and calls==['lock','unlock']
    client.timer.stop();studio.close()


def test_shared_task_entry_corrects_before_execution_and_keeps_gripper_flag():
    from autolife_shelf_grasp_306.teaching_studio import TeachingStudio
    calls=[];pending=[]
    fake=SimpleNamespace(_acquire=lambda:calls.append('lease'),ensure=lambda:None,_unlock=lambda:None,
        _completed=lambda *v:calls.append(v),engine=SimpleNamespace(bridge=None,moving=lambda:False,execute=lambda *a,**kw:calls.append((a,kw))),
        visual=SimpleNamespace(request=lambda action,callback,**kw:pending.append(callback)))
    f=frame('a',pose());f['visual_binding']=bound();record=compile_action(compose('a',[f],100))
    TeachingStudio.execute(fake,'task:2',record,preserve_grippers_at_entry=True)
    assert calls==['lease'] and len(pending)==1
    corrected=clone_without_bindings(record)
    pending[0](dict(result=dict(record=corrected,source=pose(),base_anchor=[0,0,0])))
    assert calls[-1][0]==('task:2',corrected) and calls[-1][1]['preserve_grippers_at_entry'] is True
    assert calls[-1][1]['visual_source']==pose()


def test_shared_task_failure_does_not_execute_uncorrected_record():
    from autolife_shelf_grasp_306.teaching_studio import TeachingStudio
    calls=[]
    fake=SimpleNamespace(_acquire=lambda:None,ensure=lambda:None,_unlock=lambda:None,
        _completed=lambda *v:calls.append(v),engine=SimpleNamespace(bridge=None,moving=lambda:False,execute=lambda *a,**kw:pytest.fail('unexpected motion')),
        visual=SimpleNamespace(request=lambda action,callback,**kw:callback(dict(error='参照被遮挡'))))
    TeachingStudio.execute(fake,'task:2',dict(visual_bindings={'0':bound()}))
    assert calls==[('task:2',False,'物体纠偏未执行：参照被遮挡')]


def test_pose_ui_binding_added_to_sequence_and_task_library(tmp_path):
    from PyQt5 import QtWidgets as Q
    from autolife_shelf_grasp_306.app import Window
    app=Q.QApplication.instance() or Q.QApplication([])
    root=tmp_path/'root';root.mkdir();(root/'data').mkdir();run=tmp_path/'run';run.mkdir()
    w=Window(root,run,start_worker=False);w.timer.stop();w.busy=False;page=w.pose_workspace
    try:
        page.store.save('交互位',pose(),visual_binding=bound());page.reload_poses();page.pose_list.setCurrentText('交互位')
        page.add_pose();assert page.rows[0]['visual_binding']==bound()
        page.sequence_name.setText('测试纠偏');page.save_action()
        assert w.studio.records()['测试纠偏']['visual_bindings']=={'0':bound()}
        # The original VR editor must not silently replace this action's anchor.
        w.studio.editor.draft=dict(name='测试纠偏',frames=[frame('new',pose())]);w.studio.editor.name.setText('测试纠偏')
        with pytest.raises(ValueError,match='物体纠偏'):w.studio.editor.save()
        w.studio.editor.dirty=False
        page.clear_visual();assert page.rows[0]['visual_binding']==bound() and not page.rows[0]['visual_enabled']
        assert page.visual_toggle.text()=='恢复选中姿态纠偏'
        page.save_action();assert not w.studio.records()['测试纠偏'].get('visual_bindings')
        page.dirty=False;page.load_action();assert page.rows[0]['visual_binding']==bound() and not page.rows[0]['visual_enabled']
        page.clear_visual();page.save_action();assert w.studio.records()['测试纠偏']['visual_bindings']=={'0':bound()}
        page.dirty=False
    finally:
        w.studio.editor.dirty=False;page.dirty=False;w.close();app.processEvents()


def test_model_camera_mount_change_rejected_without_movement():
    from autolife_shelf_grasp_306.visual_action_worker import VisualActions
    v=VisualActions.__new__(VisualActions);v.geometry=SimpleNamespace(frames=lambda q:{'Link_Camera_Head_Forehead':np.eye(4)})
    meta=dict(camera_mount=np.eye(4).tolist());v.check_mount(meta,{},np.eye(4))
    shifted=np.eye(4);shifted[0,3]=.01
    with pytest.raises(ValueError,match='外参'):v.check_mount(meta,{},shifted)
