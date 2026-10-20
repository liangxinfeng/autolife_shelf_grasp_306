import json
from PyQt5 import QtWidgets as Q
from autolife_shelf_grasp_306.app import Window

def test_margin30_persist_and_reopen(tmp_path):
    app=Q.QApplication.instance() or Q.QApplication([])
    root=tmp_path/'root';(root/'data').mkdir(parents=True);run=tmp_path/'run';run.mkdir()
    w=Window(root,run,start_worker=False);w.timer.stop()
    try:
        assert w.grip_margin.maximum()==50
        assert w.grip_margin.value()==5
        w.grip_margin.setValue(50)
        assert json.loads((root/'data/grip_settings.json').read_text())['tightening_mm']==50
    finally:w.close();app.processEvents()
    w=Window(root,run,start_worker=False);w.timer.stop()
    try:assert w.grip_margin.value()==50
    finally:w.close();app.processEvents()
