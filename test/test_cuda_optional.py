"""No ROS node, hardware publishers or real navigation goals are created."""
import json
import time
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from PyQt5 import QtWidgets as Q, QtCore as C
from autolife_shelf_grasp_306.app import Window
from autolife_shelf_grasp_306.task_flow_ui import FlowDialog
from autolife_shelf_grasp_306.task_flow import Cursor


@pytest.fixture(scope='session')
def qt():
    return Q.QApplication.instance() or Q.QApplication([])


@pytest.fixture
def window(qt,tmp_path,monkeypatch):
    (tmp_path/'data').mkdir();run=tmp_path/'run';run.mkdir()
    w=Window(tmp_path,run,start_worker=False)
    for timer in (w.timer,w.automatic.timer,w.navigation.product_timer,w.studio.timer):timer.stop()
    w.receive_event(dict(event='vision_unavailable',text='CUDA不可用'))
    w.launch_motion=Mock(side_effect=AssertionError('Unexpected physical motion'))
    w.studio.execute=Mock();w.studio.adjust_posture=Mock();w.studio.prepare_posture=Mock()
    w.navigation.command=Mock();w.navigation.proc=SimpleNamespace(state=lambda:C.QProcess.Running)
    w.navigation.data=dict(snapshot_time=time.time(),ready=True,pose=[0,0,0],active_map='map',names=['前台','货架'])
    monkeypatch.setattr(Q.QMessageBox,'question',lambda *a,**k:Q.QMessageBox.Yes)
    monkeypatch.setattr(Q.QMessageBox,'warning',lambda *a,**k:None)
    yield w
    w.workflow.cursor=None;w.automatic.active=False;w.automatic.shutdown()
    w.navigation.proc=None;w.hide();w.deleteLater();qt.processEvents()


def nav():return dict(kind='navigate',target='前台')
def action():return dict(kind='action',target='丢垃圾')


def set_action(w,monkeypatch):
    # Only action validation is stubbed here; actual playback regression runs separately.
    monkeypatch.setattr('autolife_shelf_grasp_306.action_studio.full_record',lambda x:x)
    w.studio.records=lambda:{'丢垃圾':{'test_record':True}}


def test_no_cuda_ui_and_editor_available_vision_disabled(window):
    w=window;w.controls()
    assert w.flow_button.isEnabled() and not w.scan.isEnabled() and not w.prepare.isEnabled()
    assert not w.semantic_apply.isEnabled() and not w.place.isEnabled()
    assert 'CUDA不可用' in w.capability_note.text()
    w.navigation.data={};w.controls();assert w.navigation.start_button.isEnabled()
    d=FlowDialog(w);d.add('navigate','前台');d.add('action','丢垃圾');d.name.setText('无GPU流程');d.save()
    assert len(json.loads((w.root/'data/task_flows.json').read_text())['无GPU流程']['steps'])==2
    d.close()


def test_navigation_action_navigation_full_dispatch_without_worker(window,qt,monkeypatch):
    w=window;set_action(w,monkeypatch);w.send=Mock(side_effect=AssertionError('Visual worker used'))
    w.workflow.start(dict(steps=[nav(),action(),nav()]),confirmed=True)
    f=w.workflow;assert f.phase=='navigate';assert w.navigation.command.call_count==1
    f.navigation_done(dict(run_id=f.cursor.run_id,leg=f.cursor.token,success=True,map='map'))
    qt.processEvents();assert f.phase=='action';assert w.studio.execute.call_count==1
    f.action_done(f.cursor.token,True,'完成');qt.processEvents()
    assert f.phase=='navigate' and w.navigation.command.call_count==2
    f.navigation_done(dict(run_id=f.cursor.run_id,leg=f.cursor.token,success=True,map='map'))
    assert not f.active and '已完成' in w.flow_label.text()
    assert not w.send.called and not w.launch_motion.called


@pytest.mark.parametrize('kind',['grasp','place'])
def test_mixed_flow_blocked_before_any_navigation(window,kind):
    w=window
    with pytest.raises(ValueError,match='第2步.*需要视觉能力'):
        w.workflow.start(dict(steps=[nav(),dict(kind=kind,target='瓶子')]),confirmed=True)
    assert not w.navigation.command.called and not w.workflow.active


@pytest.mark.parametrize('fault',['stale','no_pose','map_fault','not_ready','missing_waypoint'])
def test_no_cuda_preserves_navigation_guards(window,fault):
    d=window.navigation.data
    if fault=='stale':d['snapshot_time']=0
    elif fault=='no_pose':d['pose']=None
    elif fault=='missing_waypoint':d['names']=[]
    elif fault=='not_ready':d['ready']=False
    else:d['map_fault']=True
    with pytest.raises(ValueError):window.workflow.start(dict(steps=[nav()]),confirmed=True)
    assert not window.navigation.command.called


def test_worker_exit_does_not_interrupt_nonvisual_flow(window):
    w=window;w.workflow.start(dict(steps=[nav()]),confirmed=True)
    w.finished();assert w.workflow.active
    w.receive_event(dict(event='fatal',text='视觉进程退出'))
    assert w.workflow.active and not w.ready


def test_retry_navigation_without_perception(window):
    w=window;w.workflow.start(dict(steps=[nav()]),confirmed=True)
    w.workflow.stopped('导航暂停');assert w.workflow.retry_available
    w.controls();assert w.flow_retry.isEnabled()
    w.workflow.retry_current();assert w.workflow.active
    assert w.navigation.command.call_count==2


def test_auto_pure_flow_no_reset_no_archive_and_repeats(window,qt,monkeypatch):
    w=window;set_action(w,monkeypatch);p=w.automatic
    w.send=Mock(side_effect=AssertionError('Unexpected reset/vision request'))
    p.show_display=lambda:None
    p.library={'动作导航':dict(steps=[action(),nav()])};p.names.addItem('动作导航');p.cycles.setValue(2)
    (w.run/'held_object.json').write_text('{"sentinel":true}')
    # Avoid actual stage dispatch for this test: test startup and loop preparation semantics.
    w.workflow.next=lambda:None
    p.start('loop',confirmed=True)
    assert p.active and p.phase=='running'
    assert (w.run/'held_object.json').exists() and not w.send.called
    for _ in range(2):w.workflow.advance();qt.processEvents()
    assert p.completed==1;p.deadline=0;p.tick()
    assert p.phase=='running' and not w.send.called
    for _ in range(2):w.workflow.advance();qt.processEvents()
    assert p.completed==2 and not p.active


def test_vision_recovery_restores_vision_buttons(window):
    w=window;w.receive_event(dict(event='ready',blockers=[]));w.controls()
    assert w.ready and w.scan.isEnabled() and w.vision_error==''


def test_visual_request_cannot_bypass_disabled_button(window):
    window.process=Mock();window.send('plan',id=1)
    assert not window.process.write.called and not window.busy
    window.process=None


def test_nav_posture_uses_independent_action_executor(window):
    w=window;r=nav();r['posture']=dict(body_mode='forward',lowering_cm=10,head_pitch_deg=0)
    w.send=Mock(side_effect=AssertionError('Visual worker used'))
    w.workflow.start(dict(steps=[r]),confirmed=True)
    assert w.studio.prepare_posture.call_count==1 and not w.send.called
    f=w.workflow;f.action_done(f.cursor.token+':nav_prepare',True,'ready')
    f.navigation_started(dict(run_id=f.cursor.run_id,leg=f.cursor.token))
    assert w.studio.adjust_posture.call_count==1 and w.navigation.command.call_count==1
