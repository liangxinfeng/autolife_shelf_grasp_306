"""Offscreen UI state transitions only. No worker or motion subprocess."""
import json,tempfile
from pathlib import Path
from PyQt5 import QtWidgets as Q
from autolife_shelf_grasp_306.app import Window

app=Q.QApplication([]);root=Path(tempfile.mkdtemp());(root/'data').mkdir();run=root/'run';run.mkdir()
w=Window(root,run,start_worker=False);w.timer.stop();w.ready=True;w.busy=False;w.preparing=False
assert not w.demo.isChecked()
calls=[];w.navigate_demo=lambda leg:calls.append(leg);w.send=lambda action,**kw:calls.append(action)
w.demo_context={'grasp_plan_id':'id','run_id':'run','map':'map','front':'前台'}
for phase,filename,result,expected in [
    ('prepare','execution_result.json',{'status':'prepared'},'to_shelf'),
    ('grasp','execution_result.json',{'status':'completed','plan_id':'id'},'to_front'),
    ('grasp','execution_result.json',{'status':'stopped','reason':'障碍'},None)]:
    calls.clear();w.demo_active=True;w.demo_phase=phase;(run/filename).write_text(json.dumps(result))
    w.motion_finished()
    assert calls==([expected] if expected else [])
    if expected is None:assert not w.demo_active
calls.clear();w.demo_active=True;w.demo_phase='to_shelf'
w.navigation_done(dict(run_id='old',leg='to_shelf',success=True,map='map'))
assert not calls
w.navigation_done(dict(run_id='run',leg='to_shelf',success=True,map='map',name='货架'))
assert calls==['scan']
calls.clear();w.demo_phase='to_front'
w.navigation_done(dict(run_id='run',leg='to_front',success=True,map='map',name='前台'))
assert calls==['demo_place']
calls.clear();w.demo_active=True;w.demo_phase='observe';w.demo_context['observation_plan_id']='view'
(run/'execution_result.json').write_text(json.dumps({'status':'observed','plan_id':'view'}))
w.motion_finished();assert calls==['demo_place'] and w.demo_phase=='planning'
calls.clear();w.demo_active=True;w.demo_phase='observe'
(run/'execution_result.json').write_text(json.dumps({'status':'observed','plan_id':'stale'}))
w.motion_finished();assert not calls and not w.demo_active
w.demo_active=True;w.demo_phase='planning';w.busy=True;w.request=5
w.stop_motion();w.receive_event({'event':'place_planned','request_id':5})
assert not w.demo_active and w.motion is None
assert w.demo_phase is None
w.demo_active=True;w.demo_phase='planning';w.busy=True;w.close()
assert not w.demo_active and (run/'stop').exists()
print('DEMO_UI_PASS: phase transitions, failure stop, cancel and close; no subprocess launched')
