"""Real executor callbacks + full place state machine; no hardware imports/init."""
import ast,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from test_vendor_place import fixture
from autolife_shelf_grasp_306 import execution as executor
from autolife_shelf_grasp_306.arm_selection import gripper_targets,gripper_name
from autolife_shelf_grasp_306.core import open_tolerance
from autolife_shelf_grasp_306.motion import Arrival
from autolife_shelf_grasp_306.vendor_place_trajectory import execute_blocks
from autolife_handeye_calibration_306.core import BODY,ARMS
from control_center_joint_control.joint_definitions import GROUPS


def scenario(tmp_path,side='right',floor=16.94130039215088,fault=None,plan_override=None):
    plan=fixture(side) if plan_override is None else plan_override
    jaw=gripper_name(side);other=gripper_name('left' if side=='right' else 'right')
    q=dict(plan['source_q']);q[jaw]=231.827774;q[other]=10.43858
    path=tmp_path/'width.json'
    path.write_text(json.dumps(dict(robot_id=345,hardware='dm_type1',measured=True,
        arms={a:[[5,330],[45,180],[90,10]] for a in ('left','right')})))
    cfg={**plan['config'],'gripper_width_calibration':str(path) if plan_override is None else plan['config']['gripper_width_calibration'],
         'gripper_open_deg':10.,'gripper_speed_scale':.5}
    now=[0.];events=[];writes=[];last=[None];release_start=[None];reports=[]
    node=SimpleNamespace(_hardware_ready=True,_active_goals={},progress_watch={},_goal_sent_groups=set(),
        _gripper_pub=object(),_whole_body_pub=object(),preserve_loaded_grippers=False,snapshot=lambda:dict(q))
    def apply(*a,**k):raise AssertionError('Legacy progressive gripper/watchdog must not run')
    node.apply=apply
    def publish(pub,data):
        if pub is node._gripper_pub:
            target=data[side+'_gripper_target_joints_position'][0]
            assert data[('left' if side=='right' else 'right')+'_gripper_target_joints_position']==[q[other]]
            last[0]=target;writes.append((now[0],target))
        else:
            for group in ('leg_waist',side+'_arm'):
                if GROUPS[group]['command_key'] in data:
                    for (name,*_),value in zip(GROUPS[group]['joints'],data[GROUPS[group]['command_key']]):q[name]=value
    node._publish_vendor=publish
    def checked():
        now[0]+=.02
        if release_start[0] is not None:
            if fault and now[0]-release_start[0]>.4:raise ValueError(fault)
            if last[0] is not None:q[jaw]+=float(np.clip(max(floor,last[0])-q[jaw],-2.4,2.4))
        return dict(q)
    env=dict(__name__=executor.__name__,__package__=executor.__package__,observing=False,checked=checked,motion_now=lambda:now[0],arm=side,cfg=cfg,
        gripper_targets=gripper_targets,node=node,native_motion=True,vendor_place=True,vendor_grasp=False,
        open_tolerance=open_tolerance,Arrival=Arrival,hold_groups=executor.hold_groups,
        atomic_json=lambda p,v:reports.append(v),run=tmp_path,plan=plan)
    # Use the actual nested callback code, so routing regressions are exercised.
    tree=ast.parse(Path(executor.__file__).read_text(encoding='utf-8'))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    funcs=[n for n in ast.walk(main) if isinstance(n,ast.FunctionDef) and n.name in ('start_gripper','finish_gripper')]
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'<actual executor callbacks>','exec'),env)
    def release():
        events.append('release');release_start[0]=now[0]
        state=env['start_gripper'](10.)
        result=env['finish_gripper'](state)
        assert result=='released';events.append('released')
    def trajectory(adapter,rows,**kw):
        events.append('trajectory')
        for row in rows:adapter._set_target_body_joints_deg(row[:11],row[:4]+row[11:])
    try:
        execute_blocks(node,plan,checked,lambda:now[0],release,lambda *a:events.append('hold'),
            lambda v:None,lambda *a:True,trajectory)
    except ValueError:
        assert events==['trajectory','trajectory','release']
        raise
    assert events==['trajectory','trajectory','release','released','trajectory','hold']
    assert all(abs(q[n]-plan['vendor_blocks'][-1]['waypoints'][-1][n])<1e-8 for n in plan['controlled_joints'])
    assert q[other]==10.43858
    assert max(abs((b[1]-a[1])/(b[0]-a[0])) for a,b in zip(writes,writes[1:]) if b[0]>a[0])<=120.01
    assert any(v.get('release_width_mm')==plan['release_opening_m']*1000 for v in reports)
    return now[0]-release_start[0],q[jaw]


@pytest.mark.parametrize('side',['left','right'])
@pytest.mark.parametrize('floor',[10.,12.5,16.94130039215088,40.])
def test_release_then_return_with_measured_open_endpoint_residual(tmp_path,side,floor):
    elapsed,actual=scenario(tmp_path,side,floor)
    assert elapsed<5. and actual>=floor


@pytest.mark.parametrize('floor',[100.,231.827774])
def test_real_incomplete_opening_never_starts_return(tmp_path,floor):
    with pytest.raises(ValueError,match='张开未到位'):scenario(tmp_path,floor=floor)


@pytest.mark.parametrize('fault',['用户停止','关节反馈过期','通信离线'])
def test_stop_or_feedback_failure_never_starts_return(tmp_path,fault):
    with pytest.raises(ValueError,match=fault):scenario(tmp_path,fault=fault)
