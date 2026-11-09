from copy import deepcopy
import json
import time
import numpy as np
import pytest
from control_center_joint_control.joint_definitions import ALL_JOINTS,JOINT_LIMITS,BODY_JOINTS,GRIPPER_JOINTS
from autolife_shelf_grasp_306.teaching_library import TeachingLibrary
from autolife_shelf_grasp_306.teaching_motion import compile_action,limits
from autolife_shelf_grasp_306.teaching_trajectory import plan,evaluate,sample
from autolife_shelf_grasp_306.teaching_track import NAMES,make_draft,load_raw,validate_arrays,jaw_events
from autolife_shelf_grasp_306.teaching_recorder import Recorder
from autolife_shelf_grasp_306.task_flow import full_record

def neutral():return {n:(JOINT_LIMITS[n][0]+JOINT_LIMITS[n][1])/2 for n in ALL_JOINTS}

def track(jaws=False):
    t=np.arange(0.,8.001,.02);q=np.tile(list(neutral().values()),(len(t),1))
    q[:,NAMES.index('Joint_Right_Elbow')]+=12*np.sin(2*np.pi*t/8.)
    if jaws:q[:,NAMES.index('Joint_Right_Gripper')]+=60*np.clip((t-3.)/.8,0,1)
    return t,q

def test_immutable_track_and_task_metadata(tmp_path):
    t,q=track();draft=make_draft(tmp_path,'示教轨迹',t,q,[100,250])
    library=TeachingLibrary(tmp_path);record=library.save(draft)
    assert library.load('示教轨迹')['trajectory_ref']==draft['trajectory_ref']
    assert full_record(library.records()['示教轨迹'])['trajectory']==record['trajectory']
    tr,qr=load_raw(tmp_path,draft['trajectory_ref']);assert np.array_equal(t,tr);assert np.array_equal(q,qr)
    path=tmp_path/'trajectories'/(draft['trajectory_ref']+'.npz');path.write_bytes(path.read_bytes()+b'broken')
    with pytest.raises(ValueError,match='校验'):load_raw(tmp_path,draft['trajectory_ref'])

def test_dense_reversal_preserved_c2_and_bounded(tmp_path):
    t,q=track();draft=make_draft(tmp_path,'往返',t,q,[200]);draft['timing']='interaction';record=compile_action(draft)
    prepared,segs,phases=plan(record,neutral(),tmp_path)
    assert any('curve' in s for s in segs)
    curve=[s for s in segs if 'curve' in s]
    j='Joint_Right_Elbow';samples=[evaluate(s,u)[j] for s in curve for u in np.linspace(0,1,80)]
    assert max(samples)-min(samples)>22 # equal endpoints must not flatten the movement
    for s in segs:
        for order in (1,2,3):
            for u in np.linspace(0,1,99):
                for n,v in evaluate(s,u,order).items():assert abs(v)<=limits(n)[order-1]*1.002
    for a,b in zip(segs,segs[1:]):
        if a.get('curve') is b.get('curve') and 'curve' in a:
            for order in (0,1,2):
                assert max(abs(evaluate(a,1,order)[n]-evaluate(b,0,order)[n]) for n in ALL_JOINTS)<1e-7
    for order in (1,2):
        assert max(abs(v) for v in evaluate(curve[0],0,order).values())<1e-7
        assert max(abs(v) for v in evaluate(curve[-1],1,order).values())<1e-7

def test_jaws_move_only_with_body_stationary(tmp_path):
    t,q=track(True);draft=make_draft(tmp_path,'夹取',t,q);draft['timing']='interaction';prepared,segs,phases=plan(compile_action(draft),neutral(),tmp_path)
    jaw_segments=0
    for f,s in zip(prepared['keyframes'],segs):
        jaw_move=any(abs(s['target'][n]-s['start'][n])>1e-5 for n in GRIPPER_JOINTS)
        if jaw_move:
            jaw_segments+=1
            for u in np.linspace(0,1,20):
                assert all(abs(evaluate(s,u)[n]-s['start'][n])<1e-6 for n in BODY_JOINTS)
    assert jaw_segments>=1
    assert abs(segs[-1]['target']['Joint_Right_Gripper']-q[-1,NAMES.index('Joint_Right_Gripper')])<1e-6

def test_edit_anchor_and_hold(tmp_path):
    t,q=track();draft=make_draft(tmp_path,'编辑',t,q,[200]);draft['frames'][1]['joints_deg']['Joint_Right_Elbow']+=2
    draft['timing']='interaction'
    draft['frames'][1]['hold_sec']=.5;draft['frames'][1]['speed_percent']=60
    prepared,segs,_=plan(compile_action(draft),neutral(),tmp_path)
    target=draft['frames'][1]['joints_deg']['Joint_Right_Elbow']
    assert any(f['hold_sec']==.5 and abs(s['target']['Joint_Right_Elbow']-target)<1e-6 for f,s in zip(prepared['keyframes'],segs))

def test_reject_bad_time_and_order(tmp_path):
    t,q=track();bad=t.copy();bad[2]=bad[1]
    with pytest.raises(ValueError):validate_arrays(bad,q)
    draft=make_draft(tmp_path,'节点',t,q);draft['frames'].reverse()
    with pytest.raises(ValueError):compile_action(draft)

class Feedback:
    def __init__(self):self.values=neutral();self.stamp=1.;self.good=True
    def fresh(self):return self.good

def setup_recorder(tmp_path,mode='poses'):
    feedback=Feedback();clock=[1.];rec=Recorder(TeachingLibrary(tmp_path),feedback,lambda:clock[0])
    rec.command(dict(op='enable',request_id='enable',mode=mode),desktop=True)
    return rec,feedback,clock

def cmd(rec,op,request_id='test',**kw):return rec.command(dict(op=op,request_id=request_id,session=rec.session,**kw),desktop=True)

def test_idempotent_point_save_and_zero_dwell(tmp_path):
    rec,f,c=setup_recorder(tmp_path)
    try:
        one=cmd(rec,'capture');two=cmd(rec,'capture')
        assert one['count']==two['count']==1
        assert rec.library.load(rec.saved_name)['frames'][0]['hold_sec']==0
        with pytest.raises(ValueError):cmd(rec,'new') # reuse id with different contents
        cmd(rec,'capture','second');assert len(rec.draft['frames'])==2
        cmd(rec,'finish','finish');assert not rec.enabled
    finally:rec.close()

def test_interruption_preserves_partial_not_executable(tmp_path):
    rec,f,c=setup_recorder(tmp_path,'trajectory')
    try:
        rec.command(dict(op='heartbeat',session=rec.session,ready=True));cmd(rec,'start')
        c[0]+=2;f.stamp=c[0];rec.heartbeat=c[0];rec.tick()
        for i in range(20):
            c[0]+=.02;f.stamp=c[0];rec.heartbeat=c[0];rec.tick()
        f.good=False;rec.tick()
        assert rec.phase=='interrupted';assert len(list(rec.pending_dir.glob('*.jsonl')))==1
        assert rec.library.records()=={}
    finally:rec.close()

def test_capture_actual_measurements_and_async_save(tmp_path):
    rec,f,c=setup_recorder(tmp_path,'trajectory')
    try:
        rec.command(dict(op='heartbeat',session=rec.session,ready=True));cmd(rec,'start')
        c[0]+=2;f.stamp=c[0];rec.heartbeat=c[0];rec.tick()
        for i in range(1,151):
            c[0]+=.02;f.stamp=c[0];rec.heartbeat=c[0]
            f.values['Joint_Right_Elbow']=neutral()['Joint_Right_Elbow']+2*np.sin(i/150*np.pi)
            rec.tick()
        cmd(rec,'stop','stop')
        end=time.monotonic()+15
        while rec.phase=='processing' and time.monotonic()<end:time.sleep(.02)
        assert rec.phase=='saved',rec.message
        assert len(rec.library.records())==1;assert not list(rec.pending_dir.glob('*.jsonl'))
        t,q=load_raw(tmp_path,rec.draft['trajectory_ref']);assert len(t)==151
        assert abs(q[-1,NAMES.index('Joint_Right_Elbow')]-f.values['Joint_Right_Elbow'])<1e-9
    finally:rec.close()

def test_reset_is_captured_in_same_recording_and_quiet_wait_compressed(tmp_path):
    from autolife_shelf_grasp_306.teaching_track import load_events,reset_timing
    rec,f,c=setup_recorder(tmp_path,'trajectory')
    def heartbeat(**data):
        rec.command(dict(op='heartbeat',session=rec.session,ready=True,**data))
    def frame():c[0]+=.02;f.stamp=c[0];rec.heartbeat=c[0];rec.tick()
    try:
        heartbeat();cmd(rec,'start');c[0]+=2;f.stamp=c[0];rec.heartbeat=c[0];rec.tick()
        for i in range(50):
            f.values['Joint_Right_Elbow']+=.12;frame()
        heartbeat(reset_held=True,reset_hold_age=0.)
        for i in range(60):frame()
        rec.command(dict(op='heartbeat',session=rec.session,ready=False,recording_ready=True,resetting=True,reset_held=True))
        for i in range(60):
            f.values['Joint_Right_Elbow']-=.1;frame()
            assert rec.phase=='recording'
        heartbeat(recording_ready=True,resetting=False,reset_held=False)
        for i in range(10):frame()
        count=len(rec.samples);cmd(rec,'stop','stop')
        end=time.monotonic()+15
        while rec.phase=='processing' and time.monotonic()<end:time.sleep(.01)
        assert rec.phase=='saved',rec.message
        t,q=load_raw(tmp_path,rec.draft['trajectory_ref']);events=load_events(tmp_path,rec.draft['trajectory_ref'],len(t))
        assert len(t)==count and len(events)==1 and events[0]['end'] is not None
        effective,trimmed=reset_timing(t,q,events)
        assert trimmed>.8 and np.all(np.diff(effective)>0)
        assert any('复位开始' in f['label'] for f in rec.draft['frames'])
        assert any('复位完成' in f['label'] for f in rec.draft['frames'])
    finally:rec.close()

def test_reset_wait_does_not_remove_real_movement():
    from autolife_shelf_grasp_306.teaching_track import reset_timing
    t,q=track();q[:,NAMES.index('Joint_Right_Elbow')]=neutral()['Joint_Right_Elbow']+t
    effective,trimmed=reset_timing(t,q,[dict(kind='reset',pressed=0,start=150,end=300)])
    assert trimmed==0 and np.array_equal(t,effective)

def test_jaw_at_exact_open_limit_survives_spline_roundoff(tmp_path):
    t,q=track();q[:,NAMES.index('Joint_Left_Gripper')]=JOINT_LIMITS['Joint_Left_Gripper'][0]
    draft=make_draft(tmp_path,'开限位',t,q)
    prepared,segs,_=plan(compile_action(draft),draft['frames'][0]['joints_deg'],tmp_path)
    assert all(s['target']['Joint_Left_Gripper']>=10 for s in segs)
