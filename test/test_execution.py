"""Full reset state machine with fake time/ROS/motors; NEVER hardware."""
import json
import sys
import pytest
import numpy as np
from types import SimpleNamespace
from autolife_shelf_grasp_306 import execution as e
from autolife_handeye_calibration_307.core import ARMS,BODY,NECK


@pytest.mark.parametrize('mode',['hold_reset','hold_reset_failure','hold_reset_drift','hold_reset_stop','old_vendor_reset','unchecked_reset','reset','reset_recovery','reset_recovery_blocked','reset_ready','reset_parallel','reset_open_offset','reset_open_blocked','grasp','grasp_contact','grasp_blocked','grasp_sync_refused','grasp_fixed','grasp_place','grasp_observe','grasp_observe_slip'])
def test_execution_finishes_without_stop(tmp_path,monkeypatch,mode,handoff='unchanged'):
    names=BODY+NECK+ARMS['left']+ARMS['right']
    source={n:0. for n in names};source.update(Joint_Left_Gripper=180.,Joint_Right_Gripper=180.)
    goal={**source,NECK[1]:-20.,ARMS['left'][3]:20.,ARMS['right'][3]:-20.}
    if mode.startswith('reset_recovery'):source[NECK[1]]=-46.13
    if mode=='reset_ready':source.update(goal);source.update(Joint_Left_Gripper=10.,Joint_Right_Gripper=10.)
    now=[100.];instances=[];grip_prompt=[False];arm_started=[];unlock_calls=[];jaw_events=[]
    class FakeBridge:
        tolerance=1.
        max_step=5.
        def get_parameter(self,n):return SimpleNamespace(value=self.tolerance if n=='goal_tolerance_deg' else self.max_step)
        def set_parameters(self,values):
            for v in values:
                if v.name=='goal_tolerance_deg':self.tolerance=v.value
                else:self.max_step=v.value
            return [SimpleNamespace(successful=True) for _ in values]
        def __init__(self):
            self.q=dict(source);self._hardware_ready=True;self._active_goals={};self._goal_sent_groups=set()
            self.progress_watch={};self._whole_body_pub='body';self._gripper_pub='jaw';instances.append(self)
        def _is_feedback_fresh(self):return True
        def snapshot(self):return dict(self.q)
        def unlock(self):
            unlock_calls.append(True)
            if mode=='hold_reset_failure':return SimpleNamespace(success=False,message='使能拒绝')
            if mode=='hold_reset_drift':self.q[BODY[0]]=5.
            if mode=='hold_reset_stop':
                self._hardware_ready=False;self._hardware_enabled=False;self.last_lock_reason='硬件停止'
            return SimpleNamespace(success=True)
        def unlock_with_reset(self):raise AssertionError('禁止调用厂家系统复位')
        def apply(self,group,targets,speed_scale=None):
            assert .1<=speed_scale<=2.
            self._active_goals[group]={'targets':dict(targets),'speed_scale':speed_scale};self._goal_sent_groups.add(group)
            if group=='left_arm':arm_started.append(now[0])
            if group=='grippers':jaw_events.append((self.q[ARMS['left'][3]],targets['Joint_Left_Gripper']))
        def tick(self):
            for entry in self._active_goals.values():self.q.update(entry['targets'])
            if mode=='grasp_observe_slip' and arm_started:self.q['Joint_Left_Gripper']=200.
            if mode=='reset_recovery_blocked':self.q[NECK[1]]=-46.13
            if mode=='reset_parallel' and now[0]<102 and 'grippers' in self._active_goals:
                self.q['Joint_Left_Gripper']=180.;self.q['Joint_Right_Gripper']=180.
            if mode.startswith('reset_open_') and 'grippers' in self._active_goals:
                self.q['Joint_Left_Gripper']=11.627891
                self.q['Joint_Right_Gripper']=12.851880 if mode=='reset_open_offset' else 17.
            if mode in ('grasp_contact','grasp_blocked') and 'grippers' in self._active_goals:
                if self._active_goals['grippers']['targets']['Joint_Left_Gripper']==330.:
                    self.q['Joint_Left_Gripper']=150. if mode=='grasp_contact' else 10.
            # Deliberately keep stale bookkeeping to reproduce old completion
            # dependency: measured stable arrival must still finish autonomously.
        def _publish_vendor(self,*_):pass
        def _lock(self,*_):self._hardware_ready=False
        def destroy_node(self):pass
    monkeypatch.setattr(e,'CalibrationBridge',FakeBridge)
    sessions=[]
    class FakeSession:
        def __init__(self,node):self.node=node;self.active=False;sessions.append(self);self.closed=False
        def enter(self,spin):
            if mode=='grasp_sync_refused':raise ValueError('SYNC rejected')
            self.active=True
            if handoff=='jaw_settle':self.node.q['Joint_Left_Gripper']-=.8524
            if handoff=='jaw_large_drift':self.node.q['Joint_Left_Gripper']-=2.1
            if handoff=='arm_drift':self.node.q[ARMS['left'][3]]+=.6
        def tick(self):pass
        def close(self):self.active=False;self.closed=True;return True
    monkeypatch.setattr(e,'PositionSyncSession',FakeSession)
    monkeypatch.setattr(e,'competing_controller',lambda:'')
    monkeypatch.setattr(e,'validate_grasp_plan',lambda *args:{'bounds_deg':{n:[-180,180] for n in ARMS['left']}})
    monkeypatch.setattr(e.rclpy,'init',lambda **_:None)
    monkeypatch.setattr(e.rclpy,'spin_once',lambda n,**_:n.tick())
    monkeypatch.setattr(e.rclpy,'ok',lambda:False)
    monkeypatch.setattr(e.time,'monotonic',lambda:now[0])
    def sleep(dt):
        now[0]+=dt
    monkeypatch.setattr(e.time,'sleep',sleep)
    def printed(*args,**kwargs):
        if any('GRIP_CHECK' in str(a) for a in args):grip_prompt[0]=True
    monkeypatch.setattr('builtins.print',printed)
    import autolife_handeye_calibration_307.core as core
    monkeypatch.setattr(core,'digest',lambda _:'hash')
    monkeypatch.setattr(core,'Kinematics',lambda _:SimpleNamespace(fk=lambda *args:np.eye(4)))
    from autolife_shelf_grasp_306 import prepare
    monkeypatch.setattr(prepare,'digest',lambda _:'hash')
    if mode=='grasp_place' or mode.startswith('grasp_observe'):
        from autolife_shelf_grasp_306 import demo_protocol
        monkeypatch.setattr(demo_protocol,'validate_place',lambda *args:None)
        monkeypatch.setattr(demo_protocol,'StationaryBaseGuard',lambda node:SimpleNamespace(last=True,check=lambda:None,close=lambda:None))
    cfg={'arm':'left','head_offsets_deg':[0.,0.,0.],'reset_hardware_enabled':True,'plan_ttl_sec':20,'urdf_path':'fake','reset_tolerance_deg':1.,
         'reset_joint_speed_deg_s':70.2,'reset_feedback_lead_deg':5.,'gripper_speed_scale':.5,
         'gripper_open_deg':10.,'gripper_open_tolerance_deg':4.,'gripper_closed_deg':330.,'max_joint_speed_deg_s':120.}
    plan={'kind':'prepare','config':cfg,'created_monotonic':100.,'source_q':source,'urdf_sha256':'hash',
          'self_collision_checked':True,'segments':[{'name':'reset','waypoints':[source,goal]}]}
    if mode.startswith('hold_reset') or mode=='unchecked_reset':
        plan.update(prepare_mode='hold_then_vr',reset_protocol_version=2)
    if mode=='unchecked_reset':plan['self_collision_checked']=False
    if mode=='old_vendor_reset':
        plan.update(prepare_mode='vendor_then_vr',reset_protocol_version=1,self_collision_checked=False,segments=[])
    if mode.startswith('reset_recovery'):
        inside={**source,NECK[1]:-43.}
        plan['segments']=[{'name':'inward','waypoints':[source,inside],
                          'neck_recovery':{NECK[1]:dict(source_deg=-46.13,target_deg=-43.,lower_deg=-45.,upper_deg=30.)}},
                          {'name':'reset','waypoints':[inside,goal]}]
    if mode.startswith('grasp'):
        cfg.update(hardware_execution=True,supervised_experiment=True,calibration='fake',tcp_offset_mm=15.)
        plan.update(kind='grasp',geometry_version=2,tcp=[0,0,0],up_torso=[0,0,-1],calibration_sha256='hash',tcp_offset_mm=15.,
                    plan_id='plan-token',run_id=tmp_path.name,prepare_token='reset-token',
                    product={'id':'ice-tea','name':'冰红茶','diameter_m':.07},segments=[])
        previous=dict(source)
        for i,phase in enumerate(['pregrasp','grasp','lift','withdraw']+(['carry'] if mode=='grasp_fixed' else [])):
            end={**previous,ARMS['left'][3]:float((i+1)*5)}
            plan['segments'].append({'name':phase,'phase':phase,'waypoints':[previous,end]});previous=end
        if mode=='grasp_fixed':
            cfg.update(fixed_carry_enabled=True,carry_pose_deg=[previous[n] for n in ARMS['left']])
            plan['carry_protocol_version']=1
        if mode=='grasp_place':
            plan['kind']='place'
            for segment,phase in zip(plan['segments'],['place_approach','place_lower','place_release_retreat','place_return']):segment['phase']=phase
        if mode.startswith('grasp_observe'):
            plan.update(kind='observe',held_gripper_deg=180.)
            plan['segments']=plan['segments'][:1];plan['segments'][0]['phase']='observe_lift'
        (tmp_path/'prepare_result.json').write_text(json.dumps({'status':'prepared','token':'reset-token'}))
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));(tmp_path/'heartbeat').touch()
    monkeypatch.setattr(sys,'argv',['executor','--plan',str(path),'--run',str(tmp_path),
                                  '--expected-plan-id','plan-token','--expected-product-id','ice-tea'])
    e.main()
    result=json.loads((tmp_path/'execution_result.json').read_text())
    if mode in ('old_vendor_reset','unchecked_reset'):
        assert result['status']=='stopped' and not instances and not unlock_calls
        return
    if mode in ('hold_reset_failure','hold_reset_drift','hold_reset_stop'):
        assert result['status']=='stopped' and not arm_started and unlock_calls==[True]
        assert instances[0].q['Joint_Left_Gripper']==180 and instances[0].q['Joint_Right_Gripper']==180
        assert result.get('token') is None
        return
    if mode=='hold_reset':assert unlock_calls==[True] and all(result['q'][n]==source[n] for n in BODY)
    if mode=='reset_recovery_blocked':
        assert result['status']=='stopped' and '颈部回退未到位' in result['reason']
        assert not arm_started and result.get('token') is None
        assert instances[0].q['Joint_Left_Gripper']==180 and instances[0].q['Joint_Right_Gripper']==180
        assert instances[0].tolerance==1.
        return
    if mode.startswith('grasp'):assert len(sessions)==1 and sessions[0].closed
    else:assert not sessions
    if handoff in ('jaw_large_drift','arm_drift'):
        assert result['status']=='stopped' and 'SYNC切换期间' in result['reason']
        assert not arm_started and not jaw_events
        diagnostic=json.loads((tmp_path/'pose_handoff.json').read_text())
        assert not diagnostic['phases']['SYNC切换期间']['ok']
        return
    if mode.startswith('grasp_observe'):
        assert not jaw_events
        assert all(result.get('q',instances[0].q)[n]==source[n] for n in BODY+NECK+ARMS['right'])
        if mode=='grasp_observe':
            assert result['status']=='observed',result
            assert result['q']['Joint_Left_Gripper']==(180-.8524 if handoff=='jaw_settle' else 180)
        else:assert result['status']=='stopped' and '夹爪状态改变' in result['reason']
        assert json.loads((tmp_path/'prepare_result.json').read_text())['status']=='prepared'
        return
    if mode=='grasp_place':
        assert result['status']=='placed',result
        assert len(jaw_events)==1 and abs(jaw_events[0][0]-10.)<=.35 and jaw_events[0][1]==10.
        assert result['q']['Joint_Left_Gripper']==10 and result['q']['Joint_Right_Gripper']==180
        assert all(result['q'][n]==source[n] for n in BODY+NECK+ARMS['right'])
        return
    if mode=='grasp_sync_refused':
        assert result['status']=='stopped' and 'SYNC rejected' in result['reason']
        assert not arm_started and instances[0].q==source
        assert json.loads((tmp_path/'prepare_result.json').read_text())['status']=='prepared'
        return
    if mode=='reset_open_blocked':
        assert result['status']=='stopped' and '未稳定到位' in result['reason'],result
        assert json.loads((tmp_path/'prepare_result.json').read_text())['status']!='prepared'
        # Reset and opening run concurrently on a prechecked full-jaw-envelope
        # path. Never issue a prepared receipt if jaw opening ultimately fails.
        return
    if mode=='grasp_blocked':
        assert result['status']=='stopped' and '未稳定到位' in result['reason']
        assert abs(instances[0].q[ARMS['left'][3]]-10.)<=.35 # no lift/withdraw
        return
    if mode.startswith('reset') or mode=='hold_reset':
        assert result['status']=='prepared',result
        assert abs(result['q']['Joint_Left_Gripper']-10)<=4 and abs(result['q']['Joint_Right_Gripper']-10)<=4
        assert abs(result['q'][NECK[1]]+20)<=1.
        assert result['token']
        if mode=='reset_parallel':assert min(arm_started)<102 and now[0]>=102.35
    else:
        assert result['status']=='completed',result
        assert result['q']['Joint_Left_Gripper']==(150 if mode=='grasp_contact' else 330)
        assert result['q']['Joint_Right_Gripper']==180
        assert abs(result['q'][ARMS['left'][3]]-(25. if mode=='grasp_fixed' else 20.))<=.35
        if mode=='grasp_fixed':assert all(result['q'][n]==source[n] for n in BODY+NECK+ARMS['right'])
        assert not grip_prompt[0] and not (tmp_path/'grip_confirmed').exists()
        assert result['grip_success_verified'] is False
        assert json.loads((tmp_path/'prepare_result.json').read_text())['status']=='consumed'
    assert not (tmp_path/'stop').exists() and now[0]-100<12.


@pytest.mark.parametrize('mode',['grasp_place','grasp_observe'])
@pytest.mark.parametrize('handoff',['jaw_settle','jaw_large_drift','arm_drift'])
def test_loaded_sync_handoff_full_executor(tmp_path,monkeypatch,mode,handoff):
    test_execution_finishes_without_stop(tmp_path,monkeypatch,mode,handoff)
