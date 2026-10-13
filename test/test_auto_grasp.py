import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['ROS_DOMAIN_ID']='217'
from PyQt5 import QtWidgets as Q
from autolife_shelf_grasp_306.auto_grasp import AutoExecutionGate,rank_candidates
from autolife_shelf_grasp_306.task_flow import select_candidates,Cursor
from autolife_shelf_grasp_306.app import Window,STYLE


def test_filter_and_rank():
    items=[dict(id=2,name='绿色瓶子',score=.9,box=[0,0,10,10]),
           dict(id=1,name='绿色瓶子',score=.9,box=[0,0,20,20]),dict(id=3,name='红色瓶子',score=1)]
    assert select_candidates(items,'绿色瓶子',[])==[1,2]
    assert select_candidates(items,'绿色瓶子',[],semantic=True)==[3,1,2]
    with pytest.raises(ValueError):select_candidates(items,'不存在',[])
    assert rank_candidates([items[0],items[0]])==[2]


def test_consent():
    g=AutoExecutionGate();g.arm(1,'step',False);assert g.intent is None
    g.arm(2,'step',True);old=g.intent
    assert not g.consume(old,3,'step',True)
    assert not g.consume(old,2,'other-step',True)
    assert g.consume(old,2,'step',True)
    assert not g.consume(old,2,'step',True)
    g.arm(3,None,True);old=g.intent;g.revoke();g.arm(3,None,True)
    assert not g.consume(old,3,None,True)


@pytest.fixture(scope='session')
def qt():
    app=Q.QApplication.instance() or Q.QApplication([])
    app.setStyleSheet(STYLE)
    return app


@pytest.fixture
def window(qt,tmp_path):
    (tmp_path/'data').mkdir();run=tmp_path/'run';run.mkdir()
    w=Window(tmp_path,run,start_worker=False)
    w.timer.stop();w.navigation.product_timer.stop();w.studio.timer.stop()
    w.ready=True;w.blockers=[]
    class Process:
        def __init__(self):self.commands=[]
        def write(self,data):self.commands.append(json.loads(data))
    w.process=Process()
    w.launch_motion=lambda filename,**kwargs:(_ for _ in ()).throw(AssertionError('Unexpected motion'))
    yield w
    w.process=None;w.workflow.cursor=None;w.hide();w.deleteLater()


def scene(w):
    w.receive_event(dict(event='scene',items=[dict(id=1,name='绿色瓶子',product_id='green',score=.95),
        dict(id=2,name='绿色瓶子',product_id='green',score=.9)],shape=[720,1280,3],elapsed_ms=1))


def plan(w):
    pid='a'*32
    (w.run/'prepare_result.json').write_text(json.dumps(dict(status='prepared',token='prepared')))
    return dict(event='planned',request_id=w.request,plan_id=pid,filename='plan_'+pid+'.json',
        prepare_token='prepared',target=dict(instance_id=w.selected_id(),product_id='green',name='绿色瓶子'),
        blockers=[],shape=[720,1280,3],pixels=[],seconds=1)


def flow(w):
    w.workflow.cursor=Cursor(dict(steps=[dict(kind='grasp',target='绿色瓶子')]))
    w.workflow.phase='scan';w.workflow.semantic_target=True


def test_candidates_retry_then_manual_confirm(window,qt):
    w=window;flow(w);scene(w)
    assert w.process.commands[-1]['id']==1
    w.receive_event(dict(event='error',request_id=w.request,text='路径碰撞',candidate_unavailable=True))
    assert w.process.commands[-1]['id']==2
    assert w.workflow.active
    w.receive_event(plan(w));qt.processEvents()
    assert w.workflow.phase=='confirm' and w.execute.isEnabled()
    assert not w.auto_grasp.isChecked()


def test_candidates_do_not_retry_global_error(window):
    w=window;flow(w);scene(w)
    w.stop_motion=lambda **kw:w.workflow.cursor.stop(kw['reason'])
    before=len(w.process.commands)
    w.receive_event(dict(event='error',request_id=w.request,text='反馈失效',candidate_unavailable=False))
    assert not w.workflow.active and len(w.process.commands)==before


def test_candidate_exhaustion_stops_current_node(window):
    w=window;flow(w);scene(w)
    w.stop_motion=lambda **kw:w.workflow.cursor.stop(kw['reason'])
    for _ in range(2):
        w.receive_event(dict(event='error',request_id=w.request,text='无解',candidate_unavailable=True))
    assert not w.workflow.active and w.workflow.cursor.index==0
    assert '同类候选均未通过' in w.workflow.cursor.reason


@pytest.mark.parametrize('workflow',[False,True])
def test_auto_execute_once(window,qt,workflow):
    w=window;w.auto_grasp.setChecked(True)
    if workflow:flow(w)
    scene(w)
    if not workflow:w.send('plan',id=1)
    moves=[];w.launch_motion=lambda filename,**kw:moves.append(filename)
    with patch.object(Q.QMessageBox,'question',side_effect=AssertionError('Auto must not ask')):
        event=plan(w);w.receive_event(event);w.receive_event(event);qt.processEvents()
    assert moves==[event['filename']]
    if workflow:assert w.workflow.phase=='grasp'


@pytest.mark.parametrize('revoke',['unchecked','selection','invalidate','new_request','navigation','wrong_step','blockers','late_enable'])
def test_auto_cancel_and_stale_guards(window,qt,revoke):
    w=window
    w.auto_grasp.setChecked(revoke!='late_enable');flow(w);scene(w)
    event=plan(w)
    if revoke=='blockers':event['blockers']=['硬件未就绪']
    w.stop_motion=lambda **kw:w.workflow.cursor.stop(kw['reason'])
    w.receive_event(event)
    if revoke=='unchecked':w.auto_grasp.setChecked(False);w.auto_grasp.setChecked(True)
    elif revoke=='selection':w.list.setCurrentRow(1)
    elif revoke=='invalidate':w.invalidate_plan()
    elif revoke=='new_request':w.request+=1
    elif revoke=='navigation':w.navigation.inflight=True
    elif revoke=='wrong_step':w.workflow.cursor.run_id='new'
    elif revoke=='late_enable':w.auto_grasp.setChecked(True)
    qt.processEvents()


def test_manual_confirmation_retained(window,qt):
    w=window;scene(w);w.send('plan',id=1);event=plan(w);w.receive_event(event)
    moves=[];w.launch_motion=lambda filename,**kw:moves.append(filename)
    with patch.object(Q.QMessageBox,'question',return_value=Q.QMessageBox.No) as ask:
        w.start_execution();assert ask.called
    assert not moves
    with patch.object(Q.QMessageBox,'question',return_value=Q.QMessageBox.Yes):w.start_execution()
    assert moves==[event['filename']]


def test_checkbox_visible_in_action_bar(window,qt):
    w=window;w.resize(1280,800);w.busy=False;w.controls();w.show();qt.processEvents()
    for button in (w.prepare,w.plan,w.auto_grasp,w.execute,w.place,w.stop):
        assert button.isVisible()
        p=button.mapTo(w,button.rect().bottomRight())
        assert p.x()<w.width() and p.y()<w.height()
    if os.environ.get('AUTO_PICK_SCREENSHOT'):w.grab().save(os.environ['AUTO_PICK_SCREENSHOT'])
