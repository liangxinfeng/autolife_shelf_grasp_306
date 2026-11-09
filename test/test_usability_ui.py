from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from PyQt5 import QtWidgets as Q,QtGui as G
from autolife_shelf_grasp_306.app import Window
from autolife_shelf_grasp_306.task_flow_ui import FlowDialog

@pytest.fixture(scope='module')
def qt():return Q.QApplication.instance() or Q.QApplication([])

@pytest.fixture
def window(qt,tmp_path):
    (tmp_path/'data').mkdir();run=tmp_path/'run';run.mkdir()
    w=Window(tmp_path,run,start_worker=False)
    for timer in (w.timer,w.automatic.timer,w.navigation.product_timer,w.studio.timer):timer.stop()
    w.launch_motion=Mock(side_effect=AssertionError('No motion permitted'))
    yield w
    w.hide();w.deleteLater();qt.processEvents()

def test_recorded_default_ui_preserves_raw_action(window):
    from copy import deepcopy
    from control_center_joint_control.body_height import body_height_targets
    record={'keyframes':[{'joints_deg':body_height_targets(.2)},{'joints_deg':body_height_targets(.3)}]}
    original=deepcopy(record);window.studio.records=lambda:{'丢垃圾':record}
    d=FlowDialog(window);d.add('action','丢垃圾')
    assert d.table.cellWidget(0,2).currentData()=='recorded'
    assert not d.table.cellWidget(0,3).isEnabled()
    assert '全过程' in d.table.cellWidget(0,2).toolTip()
    assert d.record()['steps'][0]['posture']['body_mode']=='recorded'
    assert record==original
    d.add('action','丢垃圾',dict(body_mode='forward',lowering_cm=10))
    assert d.table.cellWidget(1,2).currentData()=='forward'
    assert d.table.cellWidget(1,3).value()==10
    d.done(0)

def test_point_snapshot_click_error_and_cancel(window):
    w=window;w.busy=False;w.ready=True
    image=G.QPixmap(100,80);image.fill();image.save(str(w.run/'point_scene.png'))
    w.receive_event(dict(event='point_ready',request_id=w.request,token='abc',shape=[80,100,3]))
    assert w._point_pick and not w.video.pix.isNull()
    w.send=Mock();w.image_clicked(.4,.5)
    w.send.assert_called_once_with('segment_point',token='abc',x=.4,y=.5)
    w.tick();assert (w.run/'heartbeat').exists()
    w.live();assert not w._point_pick
    w._point_pick=True
    w.receive_event(dict(event='error',request_id=w.request,text='failed'))
    assert not w._point_pick

def test_point_mode_unavailable_without_vision(window):
    window.receive_event(dict(event='vision_unavailable',text='CUDA不可用'))
    assert not window.point_select.isEnabled()
