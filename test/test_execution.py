"""Full reset state machine with fake time/ROS/motors; NEVER hardware."""
import json
import sys
import pytest
import numpy as np
from types import SimpleNamespace
from autolife_shelf_grasp_306 import execution as e
from autolife_handeye_calibration_306.core import ARMS,BODY,NECK


@pytest.mark.parametrize('mode',['reset','reset_ready','reset_parallel','reset_open_offset','reset_open_blocked','grasp','grasp_contact','grasp_blocked','grasp_sync_refused'])
def test_execution_finishes_without_stop(tmp_path,monkeypatch,mode):
    names=BODY+NECK+ARMS['left']+ARMS['right']
    source={n:0. for n in names};source.update(Joint_Left_Gripper=180.,Joint_Right_Gripper=180.)
    goal={**source,NECK[1]:-20.,ARMS['left'][3]:20.,ARMS['right'][3]:-20.}
    if mode=='reset_ready':source.update(goal);source.update(Joint_Left_Gripper=10.,Joint_Right_Gripper=10.)
    now=[100.];instances=[];grip_prompt=[False];arm_started=[]
    class FakeBridge:
        def __init__(self):
            self.q=dict(source);self._hardware_ready=True;self._active_goals={};self._goal_sent_groups=set()
            self.progress_watch={};self._whole_body_pub='body';self._gripper_pub='jaw';instances.append(self)
        def _is_feedback_fresh(self):return True
        def snapshot(self):return dict(self.q)
        def unlock(self):return SimpleNamespace(success=True)
        def apply(self,group,targets,speed_scale=None):
            self._active_goals[group]={'targets':dict(targets)};self._goal_sent_groups.add(group)
            if group=='left_arm':arm_started.append(now[0])
        def tick(self):
            for entry in self._active_goals.values():self.q.update(entry['targets'])
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
        def __init__(self,node):self.active=False;sessions.append(self);self.closed=False
        def enter(self,spin):
            if mode=='grasp_sync_refused':raise ValueError('SYNC rejected')
            self.active=True
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
    import autolife_handeye_calibration_306.core as core
    monkeypatch.setattr(core,'digest',lambda _:'hash')
    monkeypatch.setattr(core,'Kinematics',lambda _:SimpleNamespace(fk=lambda *args:np.eye(4)))
    cfg={'arm':'left','head_offsets_deg':[0.,0.,0.],'reset_hardware_enabled':True,'plan_ttl_sec':20,'urdf_path':'fake','reset_tolerance_deg':1.,
         'reset_joint_speed_deg_s':70.2,'reset_feedback_lead_deg':5.,'gripper_speed_scale':.5,
         'gripper_open_deg':10.,'gripper_open_tolerance_deg':4.,'gripper_closed_deg':330.,'max_joint_speed_deg_s':120.}
    plan={'kind':'prepare','config':cfg,'created_monotonic':100.,'source_q':source,'urdf_sha256':'hash',
          'self_collision_checked':True,'segments':[{'name':'reset','waypoints':[source,goal]}]}
    if mode.startswith('grasp'):
        cfg.update(hardware_execution=True,supervised_experiment=True,calibration='fake',tcp_offset_mm=15.)
        plan.update(kind='grasp',geometry_version=2,tcp=[0,0,0],up_torso=[0,0,-1],calibration_sha256='hash',tcp_offset_mm=15.,
                    plan_id='plan-token',run_id=tmp_path.name,prepare_token='reset-token',
                    product={'id':'ice-tea','name':'冰红茶','diameter_m':.07},segments=[])
        previous=dict(source)
        for i,phase in enumerate(['pregrasp','grasp','lift','withdraw']):
            end={**previous,ARMS['left'][3]:float((i+1)*5)}
            plan['segments'].append({'name':phase,'phase':phase,'waypoints':[previous,end]});previous=end
        (tmp_path/'prepare_result.json').write_text(json.dumps({'status':'prepared','token':'reset-token'}))
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));(tmp_path/'heartbeat').touch()
    monkeypatch.setattr(sys,'argv',['executor','--plan',str(path),'--run',str(tmp_path),
                                  '--expected-plan-id','plan-token','--expected-product-id','ice-tea'])
    e.main()
    result=json.loads((tmp_path/'execution_result.json').read_text())
    if mode.startswith('grasp'):assert len(sessions)==1 and sessions[0].closed
    else:assert not sessions
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
    if mode.startswith('reset'):
        assert result['status']=='prepared',result
        assert abs(result['q']['Joint_Left_Gripper']-10)<=4 and abs(result['q']['Joint_Right_Gripper']-10)<=4
        assert abs(result['q'][NECK[1]]+20)<=1.
        assert result['token']
        if mode=='reset_parallel':assert min(arm_started)<102 and now[0]>=102.35
    else:
        assert result['status']=='completed',result
        assert result['q']['Joint_Left_Gripper']==(150 if mode=='grasp_contact' else 330)
        assert result['q']['Joint_Right_Gripper']==180
        assert abs(result['q'][ARMS['left'][3]]-20.)<=.35 # withdrew within arrival tolerance, no UI ack
        assert not grip_prompt[0] and not (tmp_path/'grip_confirmed').exists()
        assert result['grip_success_verified'] is False
        assert json.loads((tmp_path/'prepare_result.json').read_text())['status']=='consumed'
    assert not (tmp_path/'stop').exists() and now[0]-100<12.
