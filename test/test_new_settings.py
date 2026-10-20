import math,json
import numpy as np
import pytest
from autolife_shelf_grasp_306.grip_width import closing_command,checked_margin,command_from_plan
from autolife_shelf_grasp_306.base_sequence import normalize,Controller,clearance,scan_points
from autolife_shelf_grasp_306.task_flow import validate,Cursor

@pytest.mark.parametrize('width,margin',[(.07,5),(.05,0),(.085,10),(.008,20),(.07,30),(.008,30),(.07,50),(.008,50)])
def test_closure_mapping_and_plan_binding(width,margin):
    cfg=dict(gripper_max_width_m=.095,gripper_open_deg=10.,gripper_closed_deg=330.)
    command=closing_command(width,margin,cfg)
    assert 10<=command['target_deg']<=330
    assert command['requested_width_mm']==max(0,width*1000-margin)
    p=dict(grasp_width_m=width,grip_command=command,config=cfg)
    assert command_from_plan(p)==command['target_deg']
    p['grip_command']['target_deg']+=1
    with pytest.raises(ValueError):command_from_plan(p)

@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1,50.1,True])
def test_margin_bounds(value):
    with pytest.raises(ValueError):checked_margin(value)

def test_sequence_disabled_preserved_and_no_motion():
    data=dict(enabled=False,moves=[dict(kind='forward',amount=50,speed=10)])
    v=validate(dict(steps=[dict(kind='base',base=data)]))
    assert v['steps'][0]['base']==data
    assert Controller(data).update([0,0,0],[0,0,0],0)==[0,0,0]
    c=Cursor(v);assert c.complete(c.token) and not c.active

@pytest.mark.parametrize('kind',['forward','backward','left','right','turn_left','turn_right'])
@pytest.mark.parametrize('amount',[1.,50.,180.])
def test_simulated_relative_movement_converges(kind,amount):
    turn=kind.startswith('turn_');data=dict(enabled=True,moves=[dict(kind=kind,amount=amount,speed=20 if turn else 10)])
    ctrl=Controller(data);pose=np.array([1.,2.,math.radians(173)]);initial=pose.copy();velocity=np.zeros(3);dt=.02
    for i in range(10000):
        velocity=np.array(ctrl.update(pose,velocity,i*dt))
        if ctrl.done:break
        c,s=math.cos(pose[2]),math.sin(pose[2]);pose[:2]+=np.array([[c,-s],[s,c]])@velocity[:2]*dt
        pose[2]=math.atan2(math.sin(pose[2]+velocity[2]*dt),math.cos(pose[2]+velocity[2]*dt))
    assert ctrl.done
    if not turn:assert abs(np.linalg.norm(pose[:2]-initial[:2])-amount/100)<.009

def test_sequence_order_heading_and_rotation_wrap():
    moves=[dict(kind='turn_left',amount=90,speed=20),dict(kind='forward',amount=50,speed=10),dict(kind='right',amount=20,speed=10)]
    ctrl=Controller(dict(enabled=True,moves=moves));pose=np.zeros(3);v=np.zeros(3)
    for i in range(5000):
        v=np.array(ctrl.update(pose,v,i*.02))
        if ctrl.done:break
        c,s=math.cos(pose[2]),math.sin(pose[2]);pose[:2]+=np.array([[c,-s],[s,c]])@v[:2]*.02;pose[2]+=v[2]*.02
    assert ctrl.done and np.allclose(pose[:2],[.2,.5],atol=.025)

def test_feedback_dropout_and_teleport_stop():
    c=Controller(dict(enabled=True,moves=[dict(kind='forward',amount=50,speed=10)]));c.update([0,0,0],[0,0,0],1.)
    with pytest.raises(ValueError,match='中断'):c.update([0,0,0],[0,0,0],2.)
    c=Controller(c.config);c.update([0,0,0],[0,0,0],1.)
    with pytest.raises(ValueError,match='突变'):c.update([1,0,0],[0,0,0],1.02)

def test_obstacle_guard_direction_and_turn_sweep():
    assert not clearance([[.31,0]],[.15,0,0])
    assert clearance([[.8,0]],[.15,0,0])
    assert not clearance([[-.31,0]],[-.15,0,0])
    assert not clearance([[0,.31]],[0,.15,0])
    assert not clearance([[.3,-.23]],[0,0,.38])
    assert clearance([[.8,0]],[0,0,.38])

def test_lidar_known_mounting():
    r=np.full(361,np.inf);r[180]=1
    assert np.allclose(scan_points(r,0,math.pi/180,.01,20,True),[[1.215,0]])
    assert np.allclose(scan_points(r,0,math.pi/180,.01,20,False),[[-1.215,0]])
    with pytest.raises(ValueError):scan_points(np.zeros(360),0,.01,.01,20,True)
