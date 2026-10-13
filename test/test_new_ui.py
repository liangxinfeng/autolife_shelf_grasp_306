import os,json
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['ROS_DOMAIN_ID']='217'
from PyQt5 import QtWidgets as Q
from autolife_shelf_grasp_306.app import Window
from autolife_shelf_grasp_306.task_flow_ui import FlowDialog
from autolife_shelf_grasp_306.base_sequence_ui import BaseEditor

def test_ui_roundtrip_and_disabled_defaults(tmp_path):
    app=Q.QApplication.instance() or Q.QApplication([])
    root=tmp_path/'root';root.mkdir();(root/'data').mkdir();run=tmp_path/'run';run.mkdir()
    w=Window(root,run,start_worker=False);w.timer.stop()
    try:
        assert w.grip_margin.value()==5
        w.grip_margin.setValue(7.5)
        assert json.loads((root/'data/grip_settings.json').read_text())['tightening_mm']==7.5
        w.pending_plan={'plan_id':'stale'};w.grip_margin.setValue(6);assert w.pending_plan is None
        d=FlowDialog(w);d.add('base','');assert not d.record()['steps'][0]['base']['enabled']
        value=dict(enabled=True,moves=[dict(kind='turn_right',amount=90.,speed=20.),dict(kind='forward',amount=50.,speed=10.)])
        d.table.cellWidget(0,1).value=value;d.add('place','当前桌面');d.table.selectRow(0);d.move(1)
        assert d.record()['steps'][1]['base']==value
        edit=BaseEditor(d,value);edit.table.selectRow(0);edit.move(1)
        assert edit.record()['moves'][0]['kind']=='forward'
        edit.enabled.setChecked(False);assert not edit.record()['enabled'] and len(edit.record()['moves'])==2
        edit.close();d.close()
        from autolife_shelf_grasp_306.task_flow import Cursor
        calls=[];w.workflow.base.start=lambda token:calls.append(token)
        w.workflow.cursor=Cursor(dict(steps=[dict(kind='base',base=value)]))
        w.workflow.next();assert w.workflow.phase=='base' and calls==[w.workflow.cursor.token]
        w.workflow.base_done('wrong-token',True,'');assert w.workflow.active
        w.workflow.base_done(w.workflow.cursor.token,True,'');assert not w.workflow.active
        calls.clear();w.workflow.cursor=Cursor(dict(steps=[dict(kind='base',base={'enabled':False})]))
        w.workflow.next();assert not w.workflow.active and not calls
    finally:w.close();app.processEvents()
