import json
import time
from unittest.mock import patch
import pytest
from PyQt5 import QtWidgets as Q, QtCore as C, QtGui as G
from autolife_shelf_grasp_306.app import Window
from autolife_shelf_grasp_306.task_flow import Cursor
from autolife_shelf_grasp_306.grip_width import checked_margin, closing_command


@pytest.fixture(scope='session')
def qt():
    return Q.QApplication.instance() or Q.QApplication([])


@pytest.fixture
def window(qt, tmp_path):
    (tmp_path/'data').mkdir(); run=tmp_path/'run'; run.mkdir()
    (tmp_path/'data/task_flows.json').write_text(json.dumps({'自动递送':dict(steps=[dict(kind='grasp',target='盒子')])}),encoding='utf-8')
    w=Window(tmp_path,run,start_worker=False)
    w.timer.stop();w.automatic.timer.stop();w.navigation.product_timer.stop();w.studio.timer.stop()
    w.ready=True;w.busy=False;w.blockers=[];w.module_tabs.setCurrentIndex(2)
    w.launch_motion=lambda *a,**k:pytest.fail('Tests must not issue motion')
    w.send=lambda action,**k:w.automatic.prepared({'status':'prepared'}) if action=='auto_prepare' else pytest.fail('Unexpected worker command')
    w.workflow.next=lambda:None
    yield w
    w.automatic.abort('test cleanup');w.workflow.cursor=None;w.automatic.shutdown()
    w.hide();w.deleteLater();qt.processEvents()


def start(w, mode='single'):
    w.automatic.start(mode,confirmed=True)
    assert w.automatic.phase=='running',w.automatic.state.text()
    assert w.automatic.run_id==w.workflow.cursor.run_id


def test_single_completed_only_explicit_repeat(window):
    w=window;p=w.automatic;start(w)
    assert p.display.isFullScreen()
    assert not p.display.again.isEnabled()
    assert w.automatic_grasp_enabled() and w.grasp_page_allowed()
    assert not w.auto_grasp.isChecked()  # normal manual setting untouched
    old=w.workflow.cursor.run_id;w.workflow.advance();p.tick()
    assert not p.active and p.phase=='completed' and p.completed==1
    assert p.display.again.isEnabled()
    for _ in range(5):p.tick()
    assert w.workflow.cursor.run_id==old
    with patch.object(Q.QMessageBox,'question',side_effect=AssertionError('Repeat is the explicit button')):
        p.repeat_single()
    assert p.active and w.workflow.cursor.run_id!=old and p.completed==0


def test_loop_interval_count_stale_duplicate_and_snapshot(window):
    w=window;p=w.automatic;p.cycles.setValue(2);p.interval.setValue(20);start(w,'loop')
    old=w.workflow.cursor.run_id;w.workflow.advance()
    assert p.phase=='waiting' and p.completed==1
    w.workflow.completed.emit(old);w.workflow.completed.emit('stale')
    assert p.completed==1
    p.tick();assert w.workflow.cursor.run_id==old
    p.library['自动递送']['steps'][0]['target']='不应改变'
    p.deadline=0;p.tick()
    assert w.workflow.cursor.step['target']=='盒子'
    assert w.workflow.cursor.run_id!=old
    w.workflow.advance();p.tick()
    assert not p.active and p.completed==2 and p.phase=='completed'
    assert not p.display.again.isVisible()


@pytest.mark.parametrize('at_interval',[False,True])
@pytest.mark.parametrize('method',['global','page','display_close','worker_exit'])
def test_stop_cancels_current_and_future_runs(window,at_interval,method):
    w=window;p=w.automatic;start(w,'loop')
    if at_interval:w.workflow.advance()
    old=w.workflow.cursor.run_id
    if method=='global':w.stop_motion()
    elif method=='page':p.stop_button.click()
    elif method=='display_close':p.display.close()
    else:w.finished()
    assert not p.active and not w.workflow.active and p.phase=='stopped'
    assert (w.run/'stop').exists()
    p.deadline=0;p.tick();w.workflow.completed.emit(old);p.tick()
    assert w.workflow.cursor.run_id==old and not p.active


def test_failure_no_repeat_or_success(window):
    w=window;p=w.automatic;start(w,'loop')
    w.workflow.fail('深度数据不足');p.deadline=0;p.tick()
    assert not p.active and p.phase=='stopped' and p.completed==0
    assert '深度数据不足' in p.state.text()
    assert not p.display.again.isEnabled()


def test_drain_previous_operations_bounded(window):
    w=window;p=w.automatic;start(w,'loop');w.workflow.advance();p.deadline=0
    w.busy=True;p.tick();assert p.active and p.phase=='waiting'
    p.settle_since=time.monotonic()-31;p.tick()
    assert not p.active and p.phase=='stopped'


def test_second_click_no_parallel_run_and_settings_saved(window):
    w=window;p=w.automatic;p.interval.setValue(37);p.cycles.setValue(9);start(w)
    old=w.workflow.cursor.run_id;p.start('loop',confirmed=True)
    assert w.workflow.cursor.run_id==old and p.mode=='single'
    settings=json.loads(p.path.read_text('utf-8'))
    assert settings==dict(name='自动递送',interval_s=37,cycles=9,display_direction='robot')
    with pytest.raises(ValueError,match='自动任务正在运行'):
        w.workflow.start(dict(steps=[dict(kind='grasp',target='other')]))


def test_no_worker_or_busy_cannot_start(window):
    w=window;p=w.automatic;w.ready=False;p.start('loop',confirmed=True)
    assert not p.active and not w.workflow.active
    assert '未启动' in p.state.text()


def test_cancel_consent_and_fullscreen_restore(window,qt):
    w=window;p=w.automatic
    with patch.object(Q.QMessageBox,'question',return_value=Q.QMessageBox.No):p.start('single')
    assert not p.active
    start(w);p.display.toggle_fullscreen();qt.processEvents()
    assert not p.display.isFullScreen() and p.active
    p.display.toggle_fullscreen();assert p.display.isFullScreen()
    event=G.QKeyEvent(C.QEvent.KeyPress,C.Qt.Key_Escape,C.Qt.NoModifier)
    p.display.keyPressEvent(event);assert not p.display.isFullScreen() and p.active


def test_synchronous_empty_base_completion(window):
    w=window;p=w.automatic
    p.library={'空步骤':dict(steps=[dict(kind='base',base=dict(enabled=False,moves=[]))])}
    p.names.clear();p.names.addItem('空步骤')
    w.workflow.next=lambda:w.workflow.advance()
    p.start('single',confirmed=True)
    assert p.phase=='completed' and not p.active and p.completed==1


def test_real_auto_grasp_gate_on_auto_page(window,qt):
    w=window;p=w.automatic;start(w)
    # Exercise existing scene -> plan -> auto execution code, without a process.
    class Process:
        def write(self,data):pass
    w.process=Process();w.send=Window.send.__get__(w);w.workflow.phase='scan';w.workflow.semantic_target=True
    w.receive_event(dict(event='scene',items=[dict(id=1,name='盒子',product_id='box',score=.99)],shape=[720,1280,3],elapsed_ms=1))
    (w.run/'prepare_result.json').write_text(json.dumps(dict(status='prepared',token='ok')))
    event=dict(event='planned',request_id=w.request,plan_id='a'*32,filename='plan_'+'a'*32+'.json',prepare_token='ok',
               target=dict(instance_id=1,product_id='box',name='盒子'),blockers=[],shape=[720,1280,3],pixels=[],seconds=1)
    moves=[];w.launch_motion=lambda name,**kw:moves.append(name)
    with patch.object(Q.QMessageBox,'question',side_effect=AssertionError('No per-grasp dialog')):
        w.receive_event(event);qt.processEvents()
    assert moves==[event['filename']] and w.workflow.phase=='grasp'
    w.process=None


def test_margin50_clamped_and_validated():
    assert checked_margin(50)==50
    for value in (50.01,True,float('nan')):
        with pytest.raises(ValueError):checked_margin(value)
    cmd=closing_command(.01,50,dict(gripper_max_width_m=.095,gripper_closed_deg=330,gripper_open_deg=10))
    assert cmd['requested_width_mm']==0 and cmd['target_deg']==330


def test_changed_map_stops_next_round(window):
    w=window;p=w.automatic
    p.library['自动递送']={'steps':[dict(kind='navigate',target='前台')]}
    w.navigation.data=dict(snapshot_time=time.time(),ready=True,pose=[0,0,0],active_map='map1',names=['前台'])
    start(w,'loop');w.workflow.advance()
    w.navigation.data['active_map']='map2';p.deadline=0;p.tick()
    assert not p.active and p.phase=='stopped' and '地图已变化' in p.state.text()


def test_closing_main_window_stops_interval(window):
    w=window;p=w.automatic;start(w,'loop');w.workflow.advance()
    w.close();assert not p.active and not p.timer.isActive()
