import math
import numpy as np
import pytest
from autolife_shelf_grasp_306.base_sequence import Controller, normalize, clearance, ObstacleHold, scan_points
from autolife_shelf_grasp_306.chassis import InputWatchdog


def test_braking_sweep_not_constant_speed_extrapolation():
    # 0.15m/s: 9cm reaction budget + 4.5cm braking, not an entire constant-speed horizon.
    assert clearance([[.44,0]],[.15,0,0])
    assert not clearance([[.36,0]],[.15,0,0])
    assert clearance([[0,.6]],[.30,0,0])
    assert not clearance([[.5,0]],[.30,0,0])
    assert not clearance([[.3,-.23]],[0,0,math.radians(60)])


def test_hold_immediate_stop_clear_debounce_and_timeout():
    h=ObstacleHold();command=[.15,0,0]
    assert h.update([[.31,0]],command,command,1)==[0,0,0] and h.active
    assert h.update([],command,[0,0,0],1.1)==[0,0,0] and h.active
    h.update([[.31,0]],command,[0,0,0],1.3)
    h.update([],command,[0,0,0],1.4)
    assert h.update([],command,[0,0,0],2.1)==[0,0,0] and not h.active
    assert h.update([],command,[0,0,0],2.2)==command
    h.update([[.31,0]],command,[0,0,0],3.)
    with pytest.raises(ValueError,match='30秒'):h.update([[.31,0]],command,[0,0,0],33.1)


def test_measured_speed_checks_braking_when_command_zero():
    h=ObstacleHold()
    assert h.update([[.4,0]],[0,0,0],[.3,0,0],0)==[0,0,0]
    assert h.active
    h.update([[.4,0]],[0,0,0],[0,0,0],.1)
    h.update([[.4,0]],[0,0,0],[0,0,0],1.)
    assert h.active


@pytest.mark.parametrize('front',[True,False])
@pytest.mark.parametrize('point,removed',[([.128,-.115],True),([.19,-.115],False),([.128,-.17],False),([.29,0],False),([0,.29],False)])
def test_verified_chassis_interior_only(point,removed,front):
    x,y=point
    lx,ly=(.215-x,y) if front else (x+.215,-y)
    ranges=np.full(360,np.inf);ranges[0]=math.hypot(lx,ly)
    out=scan_points(ranges,math.atan2(ly,lx),math.pi/180,.01,25,front)
    assert len(out)==(0 if removed else 1)
    if not removed:assert np.allclose(out[0],point)


@pytest.mark.parametrize('kind',['forward','backward','left','right','turn_left','turn_right'])
def test_max_speed_with_pause_keeps_distance_and_finishes(kind):
    turn=kind.startswith('turn_');speed=60 if turn else 30
    c=Controller(dict(enabled=True,moves=[dict(kind=kind,amount=180 if turn else 100,speed=speed)]))
    pose=np.zeros(3);v=np.zeros(3);peak=0.
    for i in range(4000):
        paused=100<=i<700
        v=np.array(c.update(pose,v,i*.02,paused=paused))
        if paused:assert not v.any() and not c.done
        peak=max(peak,abs(v[2]) if turn else np.linalg.norm(v[:2]))
        if c.done:break
        a=pose[2];pose[:2]+=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])@v[:2]*.02
        pose[2]+=v[2]*.02
    assert c.done
    assert abs(abs(pose[2])-math.pi)<.02 if turn else abs(np.linalg.norm(pose[:2])-1)<.01
    assert peak>.9*(math.radians(60) if turn else .3)


def test_manual_speed_unchanged_fixed_profile_bounded_and_watchdog_live():
    packet=dict(velocity=[.3,0,math.radians(60)],stamp=1.,sequence=1)
    with pytest.raises(ValueError):InputWatchdog().update(packet,1.01)
    w=InputWatchdog(fixed_sequence=True);w.update(packet,1.01)
    assert w.current(1.1)==tuple(packet['velocity'])
    assert w.current(1.3)==(0,0,0)
    for velocity in ([.4,0,0],[0,0,1.2],[.3,.3,0]):
        with pytest.raises(ValueError):InputWatchdog(fixed_sequence=True).update(dict(packet,velocity=velocity),1.01)
    with pytest.raises(ValueError):normalize(dict(enabled=True,moves=[dict(kind='forward',amount=100,speed=31)]))


def test_high_speed_editor_roundtrip():
    import os
    os.environ['QT_QPA_PLATFORM']='offscreen'
    from PyQt5 import QtWidgets as Q
    from autolife_shelf_grasp_306.base_sequence_ui import BaseEditor
    app=Q.QApplication.instance() or Q.QApplication([])
    data=dict(enabled=True,moves=[dict(kind='forward',amount=100.,speed=30.),dict(kind='turn_right',amount=180.,speed=60.)])
    d=BaseEditor(None,data)
    try:assert d.record()==data
    finally:d.close();app.processEvents()
