from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from autolife_shelf_grasp_306.teaching_track import make_draft,NAMES
from autolife_shelf_grasp_306.teaching_motion import compile_action,total_time
from autolife_shelf_grasp_306.teaching_trajectory import plan,sample
from autolife_shelf_grasp_306.teaching_replay import advance,edited_samples
from autolife_shelf_grasp_306.teaching_library import TeachingLibrary,digest
from control_center_joint_control.joint_definitions import GRIPPER_JOINTS
from test_teaching import pose

def fixture(tmp_path,seconds=19.387):
    t=np.r_[np.arange(0.,seconds,.04),seconds]
    neutral=pose(Joint_Right_Elbow=-70,Joint_Left_Elbow=-70,Joint_Right_Gripper=10)
    q=np.tile([neutral[n] for n in NAMES],(len(t),1))
    q[:,NAMES.index('Joint_Right_Elbow')]+=12*np.sin(t/seconds*2*np.pi)
    q[:,NAMES.index('Joint_Right_Gripper')]+=320*np.clip((t-4.)/.5,0,1)
    d=make_draft(tmp_path,'原录制',t,q,[100])
    return d,t,q,neutral

def test_exact_duration_all_measured_samples_and_simultaneous_jaws(tmp_path):
    d,t,q,entry=fixture(tmp_path);record,segments,_=plan(compile_action(d),entry,tmp_path)
    assert total_time(record)==pytest.approx(t[-1],abs=1e-10)
    assert all(s['scale']==1 for s in segments)
    for stamp,row in zip(t,q):
        actual,_,_=sample(record,segments,stamp)
        assert np.max(np.abs(np.array([actual[n] for n in NAMES])-row))<1e-7
    before,_,_=sample(record,segments,4.1);after,_,_=sample(record,segments,4.2)
    assert abs(before['Joint_Right_Elbow']-after['Joint_Right_Elbow'])>.01
    assert after['Joint_Right_Gripper']-before['Joint_Right_Gripper']>50

def test_preparation_is_separate_from_exact_recorded_clock(tmp_path):
    d,t,q,entry=fixture(tmp_path);entry['Joint_Right_Elbow']-=8
    record,segments,phases=plan(compile_action(d),entry,tmp_path)
    assert any('准备' in p[1] for p in phases)
    assert sum(s['seconds'] for s in segments if s.get('recorded_timing'))==pytest.approx(t[-1])
    assert total_time(record)>t[-1]

def test_only_explicit_speed_and_dwell_change_time(tmp_path):
    d,t,q,entry=fixture(tmp_path)
    for f in d['frames']:f['speed_percent']=50.;f['minimum_sec']=30.
    d['frames'][1]['hold_sec']=.6
    times,values,_=edited_samples(compile_action(d),tmp_path)
    assert times[-1]==pytest.approx(t[-1]*2+.6)
    assert np.max(np.abs(values[-1]-q[-1]))<1e-10

def test_reset_recording_is_not_time_compressed(tmp_path):
    d,t,q,entry=fixture(tmp_path);q[80:130]=q[80]
    d=make_draft(tmp_path,'含复位',t,q,reset_events=[dict(kind='reset',pressed=80,start=129,end=180)])
    record,segments,_=plan(compile_action(d),entry,tmp_path)
    assert total_time(record)==pytest.approx(t[-1])

def test_existing_library_digest_still_loads_without_rewriting(tmp_path):
    import yaml
    d,_,_,_=fixture(tmp_path);r=compile_action(d)
    r['trajectory'].pop('timing');d.pop('timing')
    path=tmp_path/'saved_joint_actions.yaml'
    path.write_text(yaml.safe_dump(dict(actions={'旧录制':r},teaching_editor={'旧录制':dict(record_digest=digest(r),draft=d)})))
    before=path.read_bytes();loaded=TeachingLibrary(tmp_path).load('旧录制')
    assert loaded['timing']=='recorded' and path.read_bytes()==before

class Controller:
    def __init__(self,record,segments,entry,clock):
        self.teaching_sync_verified=True
        self.teaching_segments=segments;self._current=dict(entry);self._preview={};self._hardware_enabled=self._hardware_ready=self._leg_waist_enabled=True
        self._action_state=dict(record=record,frame_index=0,stage='transition',name='test',joint_names=NAMES,commanded_values=dict(entry))
        self.task_pause=SimpleNamespace(now=lambda:clock[0]);self.reason='';self.published=0;self.outcome=None;self.fresh=True
    def _is_feedback_fresh(self):return self.fresh
    def _communication_lost_joints(self,names):return []
    def get_parameter(self,n):return SimpleNamespace(value=dict(max_single_step_deg=5.,max_gripper_command_lead_deg=30.,goal_tolerance_deg=1.,gripper_goal_tolerance_deg=2.)[n])
    def _cancel_action(self,text):self.reason=text;self._action_state=None
    _lock=_cancel_action
    def _record_action_event(self,*args):pass
    def _event(self,*args):pass
    def _publish_action_targets(self,state,desired,frame):self.published+=1;self._current=dict(desired)

def test_absolute_clock_and_guards(tmp_path,monkeypatch):
    d,t,q,entry=fixture(tmp_path);r,segs,_=plan(compile_action(d),entry,tmp_path)
    clock=[0.];monkeypatch.setattr('time.monotonic',lambda:clock[0]);n=Controller(r,segs,entry,clock)
    assert advance(n)
    while n._action_state and clock[0]<21:
        clock[0]+=.02;advance(n)
    assert not n.reason and n.outcome[1]
    assert t[-1]<=clock[0]<=t[-1]+.021
    for failure in ('lag','stale','stall'):
        clock[0]=0.;n=Controller(r,segs,entry,clock);advance(n);count=n.published
        if failure=='lag':n._current['Joint_Right_Elbow']-=25
        if failure=='stale':n.fresh=False
        clock[0]=.2 if failure=='stall' else .02
        advance(n);assert n.reason and n.published==count and n._action_state is None

def test_brief_native_clipping_allowed_but_persistent_lag_stops(tmp_path,monkeypatch):
    d,t,q,entry=fixture(tmp_path);r,segs,_=plan(compile_action(d),entry,tmp_path)
    clock=[0.];monkeypatch.setattr('time.monotonic',lambda:clock[0]);n=Controller(r,segs,entry,clock)
    n._current['Joint_Right_Elbow']-=13
    def frozen(*args):n.published+=1
    n._publish_action_targets=frozen
    assert advance(n) and n._action_state and n.published==1
    while n._action_state and clock[0]<.3:
        clock[0]+=.02;advance(n)
    assert .15<=clock[0]<=.18 and '限幅' in n.reason


def test_v5_hard_tracking_limit_and_session_gate(tmp_path,monkeypatch):
    d,t,q,entry=fixture(tmp_path);r,segs,_=plan(compile_action(d),entry,tmp_path)
    clock=[0.];monkeypatch.setattr('time.monotonic',lambda:clock[0])
    n=Controller(r,segs,entry,clock);n._current['Joint_Right_Elbow']-=22.01
    advance(n);assert n._action_state is None and n.published==0
    n=Controller(r,segs,entry,clock);n.teaching_sync_verified=False
    advance(n);assert n._action_state is None and n.published==0 and 'SYNC' in n.reason
