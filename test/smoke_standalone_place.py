"""Offscreen UI: no ROS worker, motion process or hardware commands."""
import tempfile,sys
from pathlib import Path
from PyQt5 import QtWidgets as Q
from autolife_shelf_grasp_306.app import Window
from autolife_shelf_grasp_306.core import atomic_json
app=Q.QApplication([]);errors=[];sys.excepthook=lambda *a:errors.append(str(a[1]))
run=Path(tempfile.mkdtemp(prefix='shelf_manual_ui_'))
w=Window('/home/ubuntu/ros2_ws/src/autolife_shelf_grasp_306',run,start_worker=False)
w.receive_event(dict(event='ready',blockers=[]));w.show();app.processEvents()
assert w.place.isVisible() and w.place.isEnabled()
w.demo.setChecked(True);assert not w.place.isVisible()
w.demo.setChecked(False)
w.start_placement();assert not w.placement_active and '没有有效持物记录' in w.status.text()
pid='a'*32
atomic_json(run/('plan_'+pid+'.json'),dict(plan_id=pid,run_id=run.name,product=dict(id='water',name='水',height_m=.2,diameter_m=.07)))
atomic_json(run/'held_object.json',dict(status='completed',plan_id=pid,q={'Joint_Left_Gripper':110}))
sent=[];launches=[]
w.send=lambda action,**kw:sent.append(action)
w.launch_motion=lambda filename,**kw:launches.append((filename,kw))
Q.QMessageBox.question=lambda *a,**kw:Q.QMessageBox.No
w.start_placement();assert not sent
Q.QMessageBox.question=lambda *a,**kw:Q.QMessageBox.Yes
w.start_placement();assert sent==['standalone_place'] and w.placement_active and not w.demo_active
w.controls();assert not w.place.isEnabled() and not w.prepare.isEnabled()
w.receive_event(dict(event='observe_planned',filename='observe.json',plan_id='b'*32,product_id='water'))
assert w.demo_phase=='observe' and launches[-1][0]=='observe.json'
atomic_json(run/'execution_result.json',dict(status='observed',plan_id='b'*32))
w.motion_finished();assert sent==['standalone_place','standalone_place'] and w.demo_phase=='planning'
w.receive_event(dict(event='place_planned',filename='place.json',plan_id='c'*32,product_id='water',pixels=[],shape=[720,1280,3]))
assert w.demo_phase=='place' and launches[-1][0]=='place.json'
atomic_json(run/'execution_result.json',dict(status='placed',plan_id='c'*32));w.motion_finished()
assert not w.placement_active and '放置完成' in w.status.text()
assert sent[-1]=='prepare'
before=len(sent)
w.finish_placement(dict(status='placed',prepared_after_place=True,auto_prepare_allowed=True))
assert len(sent)==before+1 and sent[-1]=='prepared'
w.finish_placement(dict(status='placed',auto_prepare_allowed=False,preparation_note='hold failed'))
assert len(sent)==before+1
w.stop_motion()
w.finish_placement(dict(status='placed',prepared_after_place=True))
assert len(sent)==before+1
(run/'stop').unlink()
w.start_placement();w.stop_motion();assert not w.placement_active and (run/'stop').exists()
count=len(launches)
w.receive_event(dict(event='place_planned',filename='stale.json',plan_id='d'*32,product_id='water',pixels=[],shape=[720,1280,3]))
assert len(launches)==count
w.close();assert not errors,errors
print('STANDALONE_PLACE_UI_OK')
