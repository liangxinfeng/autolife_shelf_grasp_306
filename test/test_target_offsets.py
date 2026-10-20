import json,copy
from pathlib import Path
import numpy as np
import pytest
from types import SimpleNamespace as NS
from autolife_shelf_grasp_306.target_offsets import defaults,checked_profiles,from_display,to_display,description
from autolife_shelf_grasp_306 import arm_selection as arms

def test_direction_units_and_legacy_not_added_twice():
    p=defaults({'arm':'left','target_offset_torso_mm':[0,15,0]})
    assert p=={'left':[0,15,0],'right':[0,0,0]}
    assert description(p['left'])=='向右15mm'
    assert from_display(10,-15,8)==[10,-15,-8]
    assert to_display(from_display(10,-15,8))==[10,-15,8]
    for value in ({'left':[float('nan'),0,0],'right':[0,0,0]}, {'left':[40,40,0],'right':[0,0,0]}):
        with pytest.raises(ValueError):checked_profiles(value)

@pytest.mark.parametrize('side',['left','right'])
def test_arm_plan_applies_selected_profile_once(monkeypatch,side):
    cfg=dict(arm='left',target_offset_torso_mm=[0,15,0],tcp_offset_mm=15)
    profiles={'left':[10,15,-8],'right':[-10,-5,-3]}
    monkeypatch.setattr(arms,'rank_arms',lambda *a:[side])
    monkeypatch.setattr(arms,'tool_geometry',lambda cfg,cal,fk,s:({**cfg,'arm':s,'tool_geometry':{}},np.array([.2,0,0])))
    class Planner:
        def __init__(self,cfg):self.cfg=cfg
        def plan(self,q,tcp,point,product):return {'planned_center':point.tolist()}
    pool=arms.ArmPlannerPool(cfg,{},None,Planner)
    current=dict(up_torso=[0,0,-1],target_offsets_by_arm_torso_mm=profiles)
    result=pool.plan({},[.4,.1,-.2],{},current)
    plan,planner,selected,tcp,center,correction=result
    assert np.allclose(center,np.array([.4,.1,-.2])+np.array(profiles[side])/1000)
    assert selected['target_offset_torso_mm']==profiles[side]
    assert selected['tcp_offset_mm']==15 and pool.base==cfg
    profiles[side][0]=49
    assert selected['target_offset_torso_mm'][0]!=49

def test_editor_persistence_invalidation_and_command_snapshot(tmp_path):
    from PyQt5 import QtCore as C,QtWidgets as Q
    from PyQt5.QtTest import QTest
    from autolife_shelf_grasp_306.app import Window
    app=Q.QApplication.instance() or Q.QApplication([])
    root=tmp_path/'root';(root/'data').mkdir(parents=True);run=tmp_path/'run';run.mkdir()
    w=Window(root,run,start_worker=False);w.timer.stop()
    try:
        w.ready=True;w.busy=False;w.show();app.processEvents();w.controls();w.pending_plan={'plan_id':'old'}
        widget=w.target_offsets
        for spin,text in zip(widget.spins,['10','-15','8']):
            assert spin.isEnabled()
            spin.lineEdit().setFocus();spin.lineEdit().selectAll();QTest.keyClicks(spin.lineEdit(),text);QTest.keyClick(spin.lineEdit(),C.Qt.Key_Tab)
        assert widget.profiles()['left']==[10,-15,-8] and w.pending_plan is None
        widget.arm.setCurrentIndex(1);assert [s.value() for s in widget.spins]==[0,0,0]
        widget.spins[0].setValue(5)
        assert widget.profiles()['left']==[10,-15,-8] and widget.profiles()['right']==[5,0,0]
        packets=[]
        w.process=NS(write=lambda b:packets.append(json.loads(b)))
        w.send('plan',id=123)
        assert packets[0]['target_offsets_by_arm_torso_mm']==widget.profiles()
        assert not widget.isEnabled()
        w.process=None
    finally:w.process=None;w.close();app.processEvents()
    w=Window(root,run,start_worker=False);w.timer.stop()
    try:assert w.target_offsets.profiles()=={'left':[10,-15,-8],'right':[5,0,0]}
    finally:w.close();app.processEvents()

def test_real_robot_torso_axes():
    import yaml
    from autolife_handeye_calibration_306.core import Kinematics,TORSO,MOUNT
    path=Path('/home/ubuntu/ros2_ws/src/autolife_shelf_grasp_306/config/grasp.yaml')
    if not path.exists():pytest.skip('robot model unavailable')
    cfg=yaml.safe_load(path.read_text());fk=Kinematics(cfg['urdf_path']);q={j[0]:0. for j in fk.joints.values()}
    r=fk.fk('Link_Zero_Point',TORSO,q)[:3,:3]
    print('Torso rotation at zero joints:',r.tolist())
    print('Camera mount in torso:',fk.fk(TORSO,MOUNT,q)[:3,3].tolist())
    assert r[2,2]<-.99
    assert fk.fk(TORSO,MOUNT,q)[0,3]>0
