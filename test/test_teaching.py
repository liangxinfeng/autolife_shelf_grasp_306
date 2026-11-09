import json,math,time
from copy import deepcopy
import numpy as np
import pytest
from control_center_joint_control.joint_definitions import *
from autolife_shelf_grasp_306.teaching_motion import *
from autolife_shelf_grasp_306.teaching_trajectory import build,evaluate,sample
from autolife_shelf_grasp_306.teaching_library import TeachingLibrary
from autolife_shelf_grasp_306.teaching_feedback import FeedbackState

def pose(**kwargs):
    q={n:max(lo,min(hi,0.)) for n,(lo,hi) in JOINT_LIMITS.items()};q.update(kwargs);return q
def draft(poses,hold=0):
    return dict(name='取放示教',frames=[dict(label=f'姿态{i+1}',joints_deg=q,hold_sec=hold,speed_percent=100,minimum_sec=.6,ghost=True) for i,q in enumerate(poses)])
def payload(q,speed=0):
    d={}
    for group,key in FEEDBACK_KEYS.items():d[key]=dict(position=[q[n] for n in group_joint_names(group)],speed=[speed]*len(group_joint_names(group)),communication_lost=[False]*len(group_joint_names(group)))
    for n,key in GRIPPER_FEEDBACK_KEYS.items():d[key]=dict(position=[q[n]],speed=[speed],communication_lost=[False])
    return d

def test_gripper_never_changes_during_body_travel_including_first_pose():
    q=pose();a=pose(Joint_Right_Shoulder_Inner=20,Joint_Right_Gripper=180);b=pose(Joint_Right_Shoulder_Inner=40,Joint_Right_Gripper=10)
    logical=compile_action(draft([a,b],hold=.8));staged,phases=stage_action(logical,q)
    assert len(staged['keyframes'])==4
    rows=staged['keyframes'];assert rows[0]['joints_deg']['Joint_Right_Gripper']==10
    assert rows[1]['joints_deg']==a and rows[1]['hold_sec']==.8
    assert rows[2]['joints_deg']['Joint_Right_Gripper']==180
    assert rows[3]['joints_deg']==b and rows[3]['hold_sec']==.8
    assert jaws_after_body(rows[0],rows[1]) and jaws_after_body(rows[2],rows[3])
    record,segs=build(staged,q)
    for i in [0,2]:
        for u in np.linspace(0,1,31):assert evaluate(segs[i],u)['Joint_Right_Gripper']==segs[i]['start']['Joint_Right_Gripper']
    for i in [1,3]:
        for u in np.linspace(0,1,31):
            v=evaluate(segs[i],u)
            assert all(abs(v[n]-segs[i]['target'][n])<1e-9 for n in BODY_JOINTS)

def test_zero_dwell_velocity_and_acceleration_continuous():
    q=pose();poses=[pose(Joint_Right_Shoulder_Inner=v) for v in [10,20,30]]
    staged,_=stage_action(compile_action(draft(poses)),q);record,segs=build(staged,q)
    for i in [0,1]:
        a=evaluate(segs[i],1,1);b=evaluate(segs[i+1],0,1)
        assert a['Joint_Right_Shoulder_Inner']>0
        assert max(abs(a[n]-b[n]) for n in ALL_JOINTS)<1e-8
        assert max(abs(evaluate(segs[i],1,2)[n]-evaluate(segs[i+1],0,2)[n]) for n in ALL_JOINTS)<1e-8
    assert max(abs(v) for v in evaluate(segs[-1],1,1).values())<1e-8

def test_dwell_requires_stop_and_is_not_lost():
    q=pose();a=pose(Joint_Right_Shoulder_Inner=20);b=pose(Joint_Right_Shoulder_Inner=40)
    staged,_=stage_action(compile_action(draft([a,b],hold=1.7)),q);record,segs=build(staged,q)
    assert max(abs(v) for v in evaluate(segs[0],1,1).values())<1e-8
    assert sample(record,segs,segs[0]['seconds']+.9)[0]==a

@pytest.mark.parametrize('values',[[1,20,21],[20,1,-10],[2,30,2],[90,-80,10],[-2,-5,-9]])
def test_no_overshoot_limits_and_endpoint_accuracy(values):
    q=pose();staged,_=stage_action(compile_action(draft([pose(Joint_Right_Shoulder_Inner=x) for x in values])),q);record,segs=build(staged,q)
    for s in segs:
        for n in ALL_JOINTS:
            assert abs(evaluate(s,0)[n]-s['start'][n])<1e-8
            assert abs(evaluate(s,1)[n]-s['target'][n])<1e-8
        for u in np.linspace(0,1,151):
            v=evaluate(s,u)
            for n in ALL_JOINTS:assert min(s['start'][n],s['target'][n])-1e-8<=v[n]<=max(s['start'][n],s['target'][n])+1e-8
            for order in (1,2,3):
                assert all(abs(x)<=limits(n)[order-1]*1.00001 for n,x in evaluate(s,u,order).items())

def test_library_roundtrip_metadata_and_workflow_compatibility(tmp_path):
    lib=TeachingLibrary(tmp_path);lib.read();d=draft([pose(),pose(Joint_Right_Shoulder_Inner=20)]);d['frames'][1]['label']='放下瓶子';d['frames'][1]['ghost']=False
    saved=lib.save(d,lib.revision);assert lib.load(d['name'])==normalize_draft(d)
    assert lib.records()[d['name']]==saved
    assert full_record(saved)==saved
    revision=lib.revision;other=TeachingLibrary(tmp_path);d2=deepcopy(d);d2['name']='另一动作';other.save(d2)
    with pytest.raises(ValueError,match='其他窗口'):lib.save(d,revision)
    assert '另一动作' in lib.records()
    lib.delete(d['name'],lib.revision);assert list(lib.records())==['另一动作']

def test_invalid_or_moving_feedback_not_captured():
    state=FeedbackState();q=pose()
    for t in [10,10.2,10.4]:assert state.accept(payload(q),t)
    assert state.capture(10.4)==q
    with pytest.raises(ValueError):state.capture(11.)
    state.accept(payload(q,3),10.5)
    with pytest.raises(ValueError):state.capture(10.5)
    p=payload(q);p['right_arm_joint_state']['position'][0]=float('nan');assert not state.accept(p,10.6)

def test_full_body_and_limits_required():
    q=pose();q.pop(ALL_JOINTS[0])
    with pytest.raises(ValueError):compile_action(draft([q]))
    q=pose(Joint_Right_Shoulder_Inner=400)
    with pytest.raises(ValueError):compile_action(draft([q]))
