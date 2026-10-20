"""Isolated DDS domain, mocked motion: no live robot commands."""
import os
os.environ['ROS_DOMAIN_ID']='217'
os.environ['QT_QPA_PLATFORM']='offscreen'
import json
import tempfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unittest.mock import patch
from PyQt5 import QtWidgets as Q
from autolife_shelf_grasp_306.app import Window
from autolife_shelf_grasp_306.task_flow import Cursor
from autolife_shelf_grasp_306.task_flow_ui import FlowDialog
from control_center_joint_control.joint_definitions import ALL_JOINTS, JOINT_LIMITS

app=Q.QApplication([]);root=Path(tempfile.mkdtemp(prefix='flow_ui_'));(root/'data').mkdir();run=root/'run';run.mkdir()
w=Window(root,run,start_worker=False);w.timer.stop();w.ready=True;w.busy=False;w.preparing=False
assert w.module_tabs.count()==2
d=FlowDialog(w);d.add('navigate','前台');d.add('grasp','怡宝');d.add('action','丢垃圾')
d.table.selectRow(2);d.move(-1);assert d.record()['steps'][1]['target']=='丢垃圾'
d.remove();assert len(d.record()['steps'])==2
d.name.setText('测试流程');d.save();assert (root/'data/task_flows.json').exists()

# Full control-center UI under a robot-disconnected DDS domain. Reading,
# editing, recording and previewing may publish virtual preview only.
b=w.studio.ensure()
def forbidden(*a,**k):raise AssertionError('Unexpected physical target publication')
b._publish_vendor=forbidden
p=w.studio.panel;p.spin_timer.stop()
values={n: (JOINT_LIMITS[n][0]+JOINT_LIMITS[n][1])/2 for n in ALL_JOINTS}
status=dict(connected=True,hardware_enabled=False,hardware_ready=False,leg_waist_enabled=False,
    joints={n:dict(current_deg=v,preview_deg=v,communication_lost=False,speed_deg_s=0.) for n,v in values.items()})
p.on_status(status);p.capture_action_frame(False)
record=p.action_page.build_record();assert len(record['keyframes'][0]['joints_deg'])==23
p.action_page.new_action();assert len(p.action_page.selected_groups())==5
p.capture_action_frame(False);p.action_page.copy_frame();assert len(p.action_page.build_record()['keyframes'])==2
p.action_page.transition_box.setValue(4.5);assert p.action_page.build_record()['keyframes'][-1]['transition_sec']==4.5
p.preview_action_frame(values);assert not b._hardware_enabled
w.studio.release();assert w.studio.bridge is None

calls=[];w.send=lambda action,**kw:calls.append((action,kw));w.navigation.command=lambda *a,**k:calls.append((a,k))
w.workflow.cursor=Cursor({'steps':[dict(kind='grasp',target='怡宝'),dict(kind='navigate',target='前台')]})
w.workflow.phase='prepare'
(root/'data/products.json').write_text(json.dumps({'products':[dict(id='p',name='怡宝')]}))
assert w.workflow.motion_done({'status':'prepared'});assert calls[-1][0]=='scan_task'
w.items=[dict(id=1,name='怡宝',product_id='p')];item=Q.QListWidgetItem('怡宝');item.setData(256,1);w.list.addItem(item)
w.workflow.worker_event(dict(event='scene'));assert calls[-1][0]=='plan'
w.pending_plan={'plan_id':'p'};w.workflow.worker_event(dict(event='planned',plan_id='p',blockers=[]))
assert w.workflow.phase=='confirm' and w.motion is None
w.workflow.phase='grasp';w.workflow.plan_id='p'
assert w.workflow.motion_done(dict(status='stopped',reason='测试失败'))
assert not w.workflow.active and w.workflow.cursor.index==0
w.studio.shutdown();w.close()
print('WORKFLOW_UI_PASS: editor, all 23 joints, keyframe timing, preview no motion, confirm gate, fail-stop')
