import json
from unittest.mock import patch
import pytest
from PyQt5 import QtWidgets as Q
from autolife_shelf_grasp_306.task_capabilities import starts_with_grasp
from test_task_pause import qt,window

@pytest.mark.parametrize('kinds,expected', [(['navigate','grasp','action'],True),(['place','grasp'],False),(['action','grasp'],False),(['navigate'],False),([],False)])
def test_boundary_depends_on_first_manipulation(kinds,expected):
 assert starts_with_grasp({'steps':[{'kind':k} for k in kinds]}) is expected

def test_new_grasp_task_does_not_inherit_independent_grasp(window):
 w=window;p=w.run/'held_object.json';p.write_text('old manual grasp')
 with patch.object(Q.QMessageBox,'question',return_value=Q.QMessageBox.Yes):
  w.workflow.start({'steps':[dict(kind='grasp',target='纸巾')]})
 assert w.workflow.active and not p.exists()
 assert next((w.run/'held_history').iterdir()).read_text()=='old manual grasp'
 assert w.test_commands==[] # 状态清理本身不会发布张爪/复位指令。

def test_cancel_task_keeps_record(window):
 w=window;p=w.run/'held_object.json';p.write_text('keep')
 with patch.object(Q.QMessageBox,'question',return_value=Q.QMessageBox.No):
  w.workflow.start({'steps':[dict(kind='grasp',target='纸巾')]})
 assert p.read_text()=='keep' and not w.workflow.active

def test_place_first_keeps_grasp_provenance(window):
 w=window;p=w.run/'held_object.json';p.write_text('keep for placement')
 w.workflow.start({'steps':[dict(kind='place'),dict(kind='grasp',target='纸巾')]},confirmed=True)
 assert p.read_text()=='keep for placement'

def test_failed_grasp_retry_can_clear_record_after_manual_removal(window):
 w=window;w.workflow.start({'steps':[dict(kind='grasp',target='纸巾')]},confirmed=True)
 w.workflow.phase='grasp';w.workflow.stopped('previous stop')
 p=w.run/'held_object.json';p.write_text('manually removed')
 with patch.object(Q.QMessageBox,'question',return_value=Q.QMessageBox.No):w.workflow.retry_current()
 assert p.exists() and not w.workflow.active
 with patch.object(Q.QMessageBox,'question',return_value=Q.QMessageBox.Yes):w.workflow.retry_current()
 assert w.workflow.active and not p.exists()
