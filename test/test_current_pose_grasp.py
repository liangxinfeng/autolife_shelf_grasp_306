import json
from unittest.mock import patch
import pytest
from PyQt5 import QtWidgets as Q
from test_auto_grasp import window,qt,scene,plan
from autolife_shelf_grasp_306.task_flow import Cursor
from autolife_shelf_grasp_306.grasp_start import requires_reset


def test_current_pose_manual_execute_no_receipt(window):
    w=window;scene(w);w.send('plan',id=1);e=plan(w)
    (w.run/'prepare_result.json').unlink();e.update(requires_reset=False,prepare_token=None,start_mode='current_pose')
    w.receive_event(e);assert w.execute.isEnabled()
    moves=[];w.launch_motion=lambda filename,**kw:moves.append(filename)
    with patch.object(Q.QMessageBox,'question',return_value=Q.QMessageBox.Yes):w.start_execution()
    assert moves==[e['filename']]


def test_task_grasp_scans_from_current_pose_no_prepare(window):
    w=window;(w.root/'config').mkdir();(w.root/'config/grasp.yaml').write_text('require_reset_before_grasp: false\n')
    assert not requires_reset(w.root)
    w.workflow.cursor=Cursor(dict(steps=[dict(kind='grasp',target='水瓶')]))
    w.workflow.next()
    assert w.workflow.phase=='scan'
    assert w.process.commands[-1]['action']=='scan_task'
    assert all(c['action'] not in ('prepare','auto_prepare') for c in w.process.commands)


def test_old_receipt_mode_not_silently_accepted(window):
    w=window;scene(w);w.send('plan',id=1);e=plan(w)
    (w.run/'prepare_result.json').unlink();w.receive_event(e)
    w.start_execution();assert w.pending_plan is None
