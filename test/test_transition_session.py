"""只运行替身：验证会话次数、生命周期和失效时不开始动作。"""
import json
from types import SimpleNamespace as NS
import pytest
from PyQt5 import QtWidgets as Q,QtCore as C
from autolife_shelf_grasp_306.pose_manual_session import ManualHoldSession
from control_center_joint_control.joint_definitions import ALL_JOINTS


def session_fixture(monkeypatch,loaded=False):
    clock=[100.];monkeypatch.setattr('time.monotonic',lambda:clock[0])
    q={n:0. for n in ALL_JOINTS};calls=[];sent=[];clients={}
    future=NS(done=lambda:True,result=lambda:NS(success=True,message='held'))
    class Client:
        def __init__(self,name):self.name=name
        def service_is_ready(self):return True
        def call_async(self,r):calls.append((self.name,r.data));return future
    def client(kind,name):clients[name]=Client(name);return clients[name]
    n=NS(_current=q,_details={n:dict(speed=0.) for n in q},_feedback_monotonic=100.,
        _is_feedback_fresh=lambda:True,_communication_lost_joints=lambda _:[],
        create_client=client,create_subscription=lambda *a:object(),destroy_subscription=lambda *a:None,
        create_publisher=lambda *a:NS(publish=sent.append))
    s=ManualHoldSession(n,preserve_gripper_targets=loaded)
    def spin():
        clock[0]+=.05;n._feedback_monotonic=clock[0]
        s.on_state(NS(data=json.dumps({'state':2 if calls else 0})))
        s.tick()
    # 任何重复主动读取均视为失败。
    monkeypatch.setattr('autolife_shelf_grasp_306.pose_feedback_read.PoseFeedbackRead',
        lambda *a:pytest.fail('duplicated hardware refresh'))
    return n,s,clock,calls,sent,spin


@pytest.mark.parametrize('loaded',[False,True])
def test_prepared_entry_has_one_vendor_call_no_repeated_unlock(monkeypatch,loaded):
    n,s,clock,calls,sent,spin=session_fixture(monkeypatch,loaded)
    s.requires_grasp_finish=not loaded
    s.enter_prepared(spin,dict(n._current))
    assert s.verified and s.active and n._hardware_ready
    assert len(calls)==1 and calls[0][1] is True
    assert ('loaded_session' if loaded else 'hold_session') in calls[0][0]
    assert sent and all(json.loads(m.data)['active'] for m in sent)
    assert len(s.node._current)==23 and all(v==0 for v in n._current.values())


@pytest.mark.parametrize('fault',['missing','nan_read','nan_live','changed','stale','foreign','reject','offline','stop'])
def test_prepared_entry_never_starts_on_invalid_state(monkeypatch,fault):
    n,s,clock,calls,sent,spin=session_fixture(monkeypatch)
    values=dict(n._current);key=next(iter(values))
    if fault=='missing':values.pop(key)
    if fault=='nan_read':values[key]=float('nan')
    if fault=='nan_live':n._current[key]=float('nan')
    if fault=='changed':n._current[key]+=3
    if fault=='stale':n._is_feedback_fresh=lambda:False
    if fault=='offline':n._communication_lost_joints=lambda _:[key]
    if fault=='reject':s.client.call_async=lambda r:NS(done=lambda:True,result=lambda:NS(success=False,message='busy'))
    def drive():
        spin()
        if fault=='foreign':s.on_state(NS(data='{"state":2}'))
        if fault=='stop':raise ValueError('用户停止')
    with pytest.raises(ValueError):s.enter_prepared(drive,values)
    assert not s.verified
    if fault in ('missing','nan_read','nan_live','changed','stale','foreign','stop'):assert not calls


def test_workflow_keeps_only_owned_session_between_actions_and_navigation(tmp_path,monkeypatch):
    from autolife_shelf_grasp_306.task_flow_ui import Workflow
    from autolife_shelf_grasp_306.task_flow import Cursor
    app=Q.QApplication.instance() or Q.QApplication([])
    calls=[];label=NS(setText=lambda *a:None)
    studio=NS(release=lambda:calls.append('release'))
    owner=NS(run=tmp_path,root=tmp_path,ready=True,studio=studio,flow_label=label,
        navigation=NS(proc=NS(state=lambda:C.QProcess.Running)),controls=lambda:None)
    monkeypatch.setattr('autolife_shelf_grasp_306.task_flow_ui.require_perception',lambda *a:None)
    steps=[dict(kind='navigate',target='A'),dict(kind='action',target='one'),dict(kind='grasp',target='bottle')]
    w=NS(owner=owner,active=True,cursor=Cursor({'steps':steps}),send_navigation=lambda:calls.append('nav'),
        adjust_posture=lambda fn:fn(),start_fixed_action=lambda:calls.append('action'),
        start_scan=lambda:calls.append('scan'),fail=lambda e:pytest.fail(str(e)))
    monkeypatch.setattr('autolife_shelf_grasp_306.grasp_start.requires_reset',lambda _:False)
    Workflow.next(w);assert calls==['nav']
    w.cursor.complete(w.cursor.token);Workflow.next(w);assert calls==['nav','action']
    w.cursor.complete(w.cursor.token);Workflow.next(w);assert calls==['nav','action','release','scan']


def test_workflow_completion_releases_session_before_success(tmp_path):
    from autolife_shelf_grasp_306.task_flow_ui import Workflow
    from autolife_shelf_grasp_306.task_flow import Cursor
    app=Q.QApplication.instance() or Q.QApplication([]);calls=[]
    owner=NS(studio=NS(release=lambda:calls.append('release')),flow_label=NS(setText=lambda *a:None),controls=lambda:None)
    class Flow:
        def __init__(self):
            self.owner=owner;self.cursor=Cursor({'steps':[dict(kind='action',target='one')]})
            self.completed=NS(emit=lambda _:calls.append('complete'))
        @property
        def active(self):return self.cursor.active
        def fail(self,e):pytest.fail(str(e))
    w=Flow();Workflow.advance(w);assert calls==['release','complete']


@pytest.mark.parametrize('kind,keep',[('navigate',True),('action',True),('grasp',False),('place',False),('base',False)])
def test_posture_reuse_never_leaks_into_separate_grasp_process(kind,keep):
    from autolife_shelf_grasp_306.task_flow_ui import Workflow
    calls=[]
    owner=NS(flow_label=NS(setText=lambda *a:None),controls=lambda:None,
        studio=NS(adjust_posture=lambda *a,**kw:calls.append(kw)))
    w=NS(owner=owner,cursor=NS(token='one',step=dict(kind=kind,posture=dict(head_pitch_deg=5))),phase=kind,
        adjust_posture=lambda *a:None)
    Workflow.adjust_posture(w,lambda:None)
    assert calls==[dict(keep_session=keep)]


@pytest.mark.parametrize('native,placing',[(True,False),(True,True),(False,False)])
def test_actual_executor_preparation_branch_one_enable(monkeypatch,native,placing):
    # 执行正式main里的准备块，替换外部服务；比较抓取、放置与原复位分支。
    from pathlib import Path
    import textwrap
    import autolife_shelf_grasp_306.execution as execution
    source=Path(execution.__file__).read_text()
    start=source.index("        mark('enable_requested')")
    stop=source.index('        if native_motion:\n            response=node.enable_planned_waist',start)
    calls=[]
    class Session:
        def __init__(self,node,**kw):self.kw=kw;self.requires_grasp_finish=False
        def enter_prepared(self,spin,refreshed):calls.append(('prepared',self.kw,self.requires_grasp_finish))
        def enter(self,spin):calls.append(('sync',self.kw))
    monkeypatch.setattr('autolife_shelf_grasp_306.pose_manual_session.ManualHoldSession',Session)
    def unlock():calls.append(('unlock',));return NS(success=True)
    env=dict(__package__='autolife_shelf_grasp_306',native_motion=native,placing=placing,observing=False,
        vendor_grasp=native and not placing,node=NS(unlock=unlock,_hardware_ready=True),
        PositionSyncSession=Session,mark=lambda n:calls.append(('mark',n)),spin=lambda:None,
        check_start_pose=lambda s:calls.append(('check',s)),refreshed={'validated':True},
        cfg=dict(use_position_sync_session=True),preparing=not native,neck_contact_segments=[],time=__import__('time'))
    exec(compile(textwrap.dedent(source[start:stop]),'<actual preparation>','exec'),env)
    if native:
        assert [c[0] for c in calls].count('prepared')==1 and not any(c[0]=='unlock' for c in calls)
        assert ('prepared',dict(preserve_gripper_targets=placing),not placing) in calls
    else:assert [c[0] for c in calls].count('unlock')==1 and [c[0] for c in calls].count('sync')==1
    assert calls[-1]==('check','SYNC切换期间')


def test_navigation_prepare_height_then_action_share_one_session(monkeypatch,tmp_path):
    from autolife_shelf_grasp_306.action_studio import ActionExecutor
    from autolife_shelf_grasp_306.teaching_motion import compile_action
    from test_teaching import pose,draft
    import rclpy
    monkeypatch.setattr(rclpy,'spin_once',lambda *a,**k:None)
    monkeypatch.setattr('autolife_shelf_grasp_306.pose_start_ready.ready',lambda *a:True)
    clock=[100.];monkeypatch.setattr('time.monotonic',lambda:clock[0])
    app=Q.QApplication.instance() or Q.QApplication([]);sessions=[];starts=[];lower=[];receipts=[]
    class Session:
        def __init__(self,b,**kw):self.active=True;self.verified=False;self.lower_enabled=set();sessions.append(self)
        def begin(self):pass
        def tick(self):pass
        def poll(self):self.verified=True;return True
        def close(self):return True
    monkeypatch.setattr('autolife_shelf_grasp_306.pose_manual_session.ManualHoldSession',Session)
    q=pose();b=NS(_current=q,_details={n:dict(speed=0.) for n in q},_is_feedback_fresh=lambda:True,
        _communication_lost_joints=lambda _:[],_action_state=None,_active_goals={},_hardware_prepare=None,
        _hardware_enabled=True,_hardware_ready=True,outcome=None,_last_event='',_instance_lock_handle=None,
        _event=lambda *a:None,destroy_node=lambda:None,_lock=lambda *a:None,
        _set_hardware_enabled_without_reset=lambda *a:pytest.fail('duplicate enable'),
        _set_leg_waist_enabled=lambda *a:NS(success=True),_joint_enable_pub='enable',
        _publish_hardware_hold_pose=lambda *a:None,_publish_vendor=lambda pub,payload:lower.append(payload))
    def apply(msg):
        data=json.loads(msg.data);b._active_goals[data['group']]=data
    b._on_apply=apply
    def command(msg):starts.append(msg.data);b._action_state={'name':'action'}
    b._on_action_command=command
    e=ActionExecutor(NS(root=tmp_path,run=tmp_path));e.timer.stop();e.bridge=b;e.ensure=lambda:b
    e.panel=NS(setEnabled=lambda *a:None,stop_action_preview=lambda **k:None,preview_timer=NS(stop=lambda:None),
        _pending_live_groups=set(),_pending_live_joint_names=set(),_pending_preview_groups=set(),
        _set_checked_silently=lambda *a:None,live_toggle=None)
    e.completed.connect(lambda *a:receipts.append(a))
    e.prepare_posture('nav');e.tick();e.tick()
    assert e.job is None and e.manual_session and len(sessions)==1 and receipts[-1][1]
    e.tick()  # 导航期间空闲保持只发心跳。
    e.adjust_posture('height',dict(body_mode='forward',lowering_cm=5),keep_session=True)
    e.tick();assert e.job['phase']=='posture' and len(lower)==1
    b._current.update(e.job['posture']);b._active_goals={}
    e.tick();clock[0]+=.51;e.tick()
    assert e.job is None and e.bridge is b and len(sessions)==1
    e.execute('action',compile_action(draft([dict(b._current)])),keep_session=True)
    e.tick();e.tick();assert len(sessions)==1 and len(starts)==1 and len(lower)==1
    e.job=None;b._action_state=None;e.release();assert e.bridge is None
    e.deleteLater()
