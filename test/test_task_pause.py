import json
import time
from types import SimpleNamespace
from copy import deepcopy
from unittest.mock import patch
import pytest
from PyQt5 import QtWidgets as Q
from autolife_shelf_grasp_306.task_pause import BridgePause,archive_held,defer_call
from autolife_shelf_grasp_306.base_sequence import Controller
from autolife_shelf_grasp_306.app import Window
from control_center_joint_control.joint_definitions import ALL_JOINTS


def test_hold_freezes_clocks_targets_frame_and_preserves_grip(tmp_path):
    now=[100.];q={n:0. for n in ALL_JOINTS};q['Joint_Left_Gripper']=180
    logs=[];locks=[]
    node=SimpleNamespace(_hardware_ready=True,_is_feedback_fresh=lambda:True,
        _communication_lost_joints=lambda _:[],_current=q,_whole_body_pub='body',_gripper_pub='grip',
        _publish_vendor=lambda pub,msg:logs.append((pub,msg)),_lock=locks.append,
        _active_goals={'arm':dict(commands=dict(q),targets=dict(q))},
        _action_state=dict(frame_started=99.,frame_deadline=110.,last_progress_at=99.,hold_started=99.,
                           commanded_values=dict(q),frame_index=3),progress_watch={'arm':(99.,dict(q))})
    gate=BridgePause(node,tmp_path,lambda:now[0]);target=deepcopy(node._active_goals)
    (tmp_path/'task_pause').touch();assert gate.tick();assert len(logs)==2
    assert logs[1][1]['left_gripper_target_joints_position']==[180]
    now[0]=300.;assert gate.tick() and gate.now()==100.
    assert node._active_goals==target and node._action_state['frame_index']==3
    (tmp_path/'task_pause').unlink();assert not gate.tick()
    assert gate.now()==100. and node._action_state['frame_started']==299.
    assert node._action_state['frame_deadline']==310. and gate.generation==1
    assert node.progress_watch['arm'][0]==300 and not locks


def test_hold_drift_or_feedback_still_stops(tmp_path):
    q={n:0. for n in ALL_JOINTS};locks=[];fresh=[True]
    n=SimpleNamespace(_hardware_ready=True,_is_feedback_fresh=lambda:fresh[0],_communication_lost_joints=lambda _:[],
        _current=q,_whole_body_pub=None,_gripper_pub=None,_publish_vendor=lambda *a:None,_lock=locks.append)
    g=BridgePause(n,tmp_path);(tmp_path/'task_pause').touch();assert g.tick()
    q[next(iter(q))]=4;assert g.tick() and '姿态' in locks[-1]
    fresh[0]=False;assert g.tick() and '反馈' in locks[-1]


def test_real_fixed_action_transition_and_dwell_resume_same_frame(tmp_path,monkeypatch):
    from control_center_joint_control.joint_bridge import JointBridge
    now=[100.];q={n:0. for n in ALL_JOINTS};joint='Joint_Left_Gripper'
    frame=dict(transition_sec=10.,hold_sec=3.,easing='smooth',joints_deg={joint:100.})
    state=dict(record=dict(groups=['grippers'],speed_percent=100,keyframes=[frame]),
        frame_index=0,loop_index=0,joint_names=[joint],stage='transition',frame_started=100.,
        frame_deadline=150.,last_progress_at=100.,best_error_deg=100.,
        start_values={joint:0.},commanded_values={joint:0.})
    finished=[]
    n=SimpleNamespace(_hardware_ready=True,_hardware_enabled=True,_leg_waist_enabled=True,
        _current=q,_action_state=state,_active_goals={},_is_feedback_fresh=lambda:True,
        _communication_lost_joints=lambda _:[],_whole_body_pub=None,_gripper_pub=None,
        _publish_vendor=lambda *a:None,_lock=lambda reason:pytest.fail(reason),
        _preview={},get_parameter=lambda name:SimpleNamespace(value=.2 if name=='action_progress_epsilon_deg' else 30.),
        _publish_action_targets=lambda s,d,f:q.update(d),
        _action_frame_max_error=lambda f:abs(q[joint]-100),
        _action_target_reached=lambda targets:abs(q[joint]-100)<.01,
        _advance_action_frame=lambda t:finished.append(t),
        _cancel_action=lambda reason:pytest.fail(reason))
    monkeypatch.setattr('control_center_joint_control.joint_bridge.time.monotonic',lambda:now[0])
    g=BridgePause(n,tmp_path,lambda:now[0]);now[0]=104.;JointBridge._advance_action(n)
    before=q[joint];(tmp_path/'task_pause').touch();assert g.tick();now[0]=204.;assert g.tick()
    (tmp_path/'task_pause').unlink();g.tick();JointBridge._advance_action(n)
    assert q[joint]==pytest.approx(before) and state['frame_index']==0
    now[0]=210.;JointBridge._advance_action(n);assert state['stage']=='hold'
    now[0]=211.;(tmp_path/'task_pause').touch();g.tick();now[0]=311.;g.tick()
    (tmp_path/'task_pause').unlink();g.tick();JointBridge._advance_action(n);assert not finished
    now[0]=313.;JointBridge._advance_action(n);assert finished==[313.]


def test_receipt_archived_and_new_one_not_touched(tmp_path):
    p=tmp_path/'held_object.json';p.write_text('original')
    archive_held(tmp_path);assert not p.exists()
    assert next((tmp_path/'held_history').iterdir()).read_text()=='original'
    p.write_text('current round');assert p.read_text()=='current round'


@pytest.fixture(scope='session')
def qt():return Q.QApplication.instance() or Q.QApplication([])


@pytest.fixture
def window(qt,tmp_path):
    (tmp_path/'data').mkdir();run=tmp_path/'run';run.mkdir()
    (tmp_path/'data/task_flows.json').write_text(json.dumps({'测试':{'steps':[dict(kind='grasp',target='盒子')]}}))
    w=Window(tmp_path,run,start_worker=False)
    for timer in (w.timer,w.automatic.timer,w.navigation.product_timer,w.studio.timer):timer.stop()
    w.busy=False;w.ready=True;w.module_tabs.setCurrentIndex(2)
    commands=[];w.send=lambda a,**k:commands.append((a,k));w.test_commands=commands
    w.workflow.next=lambda:None
    yield w
    w.automatic.abort('test');w.workflow.cursor=None;w.automatic.shutdown();w.hide();w.deleteLater();qt.processEvents()


def started(w):
    w.automatic.start('loop',confirmed=True)
    assert w.automatic.phase=='preparing'
    w.automatic.prepared(dict(status='prepared'))
    assert w.workflow.active


def test_new_round_checks_prepare_and_clears_only_boundary(window):
    w=window;(w.run/'held_object.json').write_text('old')
    w.automatic.start('loop',confirmed=True)
    assert w.test_commands==[('auto_prepare',{})] and not w.workflow.active
    assert not (w.run/'held_object.json').exists()
    w.automatic.prepared(dict(status='prepared'))
    (w.run/'held_object.json').write_text('new')
    w.automatic.tick();assert (w.run/'held_object.json').read_text()=='new'
    w.workflow.advance();w.automatic.deadline=0;w.automatic.tick()
    assert not (w.run/'held_object.json').exists() and len(w.test_commands)==2


def test_prepare_failure_stops_no_navigation(window):
    w=window;w.automatic.start('single',confirmed=True)
    w.automatic.prepared(dict(status='stopped',reason='测试失败'))
    assert not w.automatic.active and not w.workflow.active


def test_pause_countdown_frozen_and_second_click_resumes(window):
    w=window;started(w);w.workflow.advance();p=w.automatic
    clock=[100.];p.deadline=120.
    with patch('autolife_shelf_grasp_306.auto_task_ui.time.monotonic',side_effect=lambda:clock[0]):
        p.toggle_pause();clock[0]=200;p.tick()
        assert '20 秒' in p.state.text() and p.completed==1
        assert (w.run/'task_pause').exists()
        p.toggle_pause();assert p.deadline==220 and not (w.run/'task_pause').exists()
        p.tick();assert p.phase=='waiting'


def test_pause_does_not_replay_completed_step_and_stop_revokes(window):
    w=window;started(w);p=w.automatic;p.toggle_pause()
    w.workflow.advance();assert w.workflow.cursor.index==0
    assert len(p.deferred)==1
    w.workflow.advance();assert len(p.deferred)==1
    p.toggle_pause();assert w.workflow.cursor.index==1 and p.completed==1
    p.toggle_pause();p.stop();assert not p.paused and not p.active and not p.deferred
    assert not (w.run/'task_pause').exists() and (w.run/'stop').exists()


def test_paused_planning_replans_on_resume(window):
    w=window;started(w);p=w.automatic;p.toggle_pause()
    w.last_worker_request=('plan',{'id':7});w.busy=True
    assert p.defer_worker_event(dict(event='planned'))
    assert not w.busy
    p.toggle_pause();assert w.test_commands[-1]==('plan',{'id':7})


def test_paused_plan_uses_its_own_request_not_later_worker_command(window):
    w=window;started(w);p=w.automatic;p.toggle_pause()
    w.last_worker_request=('plan',{'id':7});p.defer_worker_event(dict(event='planned'))
    w.last_worker_request=('prepare',{})
    p.toggle_pause();assert w.test_commands[-1]==('plan',{'id':7})


def test_paused_prepare_receipt_does_not_reset_again(window):
    w=window;p=w.automatic;p.start('single',confirmed=True)
    w.last_worker_request=('auto_prepare',{});p.toggle_pause()
    p.defer_worker_event(dict(event='auto_prepared'))
    count=len(w.test_commands);p.toggle_pause()
    assert p.phase=='running' and len(w.test_commands)==count


def test_fixed_base_pause_excludes_time_without_restarting_origin():
    c=Controller(dict(enabled=True,moves=[dict(kind='forward',amount=50,speed=10)]))
    c.update([0,0,0],[0,0,0],0.);origin=c.origin.copy()
    for i in range(1,1500):assert c.update([0,0,0],[0,0,0],i*.02,paused=True)==[0.,0.,0.]
    assert (c.origin==origin).all() and c.index==0
    assert c.update([0,0,0],[0,0,0],30.)[0]>0


def test_portrait_window_rotation_and_return_desktop(window,qt):
    p=window.automatic;p.show_display();d=p.display
    d.resize(1280,720);d.relayout();qt.processEvents()
    assert d.proxy.rotation()==-90
    assert d.content.height()>d.content.width()
    p.display_direction.setCurrentIndex(1);d.relayout();assert d.proxy.rotation()==0
    d.toggle_fullscreen();assert not d.isFullScreen() and d.proxy.rotation()==0


@pytest.mark.parametrize('outcome',[None,('test',False,'暂停保持失败')])
def test_paused_action_fault_is_reported_immediately(window,outcome):
    from types import SimpleNamespace
    w=window;s=w.studio
    (w.run/'task_pause').touch()
    s.bridge=SimpleNamespace(outcome=outcome,_hardware_enabled=False,_hardware_ready=False,
        task_pause=SimpleNamespace(started=0.),_is_feedback_fresh=lambda:True)
    s.job=dict(token='test',phase='running',started=0.)
    results=[];s.completed.connect(lambda *args:results.append(args))
    s.stop=lambda:setattr(s,'job',None)
    with patch('rclpy.spin_once'):
        s.tick()
    assert s.job is None and len(results)==1 and results[0][1] is False
    s.bridge=None
