from copy import deepcopy
import numpy as np
import pytest
from test_teaching import pose,draft
from autolife_shelf_grasp_306.teaching_motion import compile_action,limits,stage_action
from autolife_shelf_grasp_306.teaching_trajectory import plan,evaluate
from autolife_shelf_grasp_306.visual_path_sampling import sample_count
from autolife_shelf_grasp_306.trajectory_extrema import peak_absolute
from control_center_joint_control.joint_definitions import ALL_JOINTS,BODY_JOINTS,GRIPPER_JOINTS


def test_feedback_jitter_does_not_insert_jaw_stop(tmp_path):
    entry=pose(Joint_Left_Gripper=112.,Joint_Right_Gripper=112.)
    targets=[{**entry,'Joint_Right_Shoulder_Inner':v,'Joint_Right_Gripper':112.+j}
             for v,j in [(10,.076),(20,-.076),(30,.438)]]
    record=compile_action(draft(targets));original=deepcopy(record)
    staged,segments,phases=plan(record,entry,tmp_path)
    assert record==original and len(segments)==3
    assert [s['continuous_boundary'] for s in segments]==[True,True,False]
    for s in segments:
        assert s['target']['Joint_Right_Gripper']==112.
    for a,b in zip(segments,segments[1:]):
        for order in (0,1,2):
            assert max(abs(evaluate(a,1,order)[n]-evaluate(b,0,order)[n]) for n in ALL_JOINTS)<1e-7
        assert evaluate(a,1,1)['Joint_Right_Shoulder_Inner']>0


def test_small_accumulated_jaw_changes_are_not_lost():
    entry=pose();qs=[pose(Joint_Right_Gripper=x) for x in [10.2,10.4,10.6,10.8,11.2]]
    r,phases=stage_action(compile_action(draft(qs)),entry)
    jaws=[f for f,p in zip(r['keyframes'],phases) if p[1]=='夹爪开合 → 停顿']
    assert [f['joints_deg']['Joint_Right_Gripper'] for f in jaws]==[10.6,11.2]


def test_real_jaw_event_preserves_body_first_and_dwell(tmp_path):
    q=pose();target=pose(Joint_Right_Shoulder_Inner=20,Joint_Right_Gripper=180)
    d=draft([target],hold=.7);r,segs,phases=plan(compile_action(d),q,tmp_path)
    assert len(segs)==2 and r['keyframes'][0]['hold_sec']==.25
    assert r['keyframes'][1]['hold_sec']==.7 and not any(s['continuous_boundary'] for s in segs)
    for u in np.linspace(0,1,51):
        assert evaluate(segs[0],u)['Joint_Right_Gripper']==q['Joint_Right_Gripper']
        assert all(abs(evaluate(segs[1],u)[n]-target[n])<1e-8 for n in BODY_JOINTS)


def test_model_sampling_units_and_upgraded_model_fallback():
    a=pose();b=pose(Joint_Right_Gripper=330);s=dict(start=a,target=b)
    assert sample_count(s,set(BODY_JOINTS))==2
    assert sample_count(s,set(ALL_JOINTS))==1280
    for n in BODY_JOINTS:
        # 函数只处理采样计数，限位由真实模型验证。
        s=dict(start=a,target={**b,n:a[n]+8})
        assert sample_count(s,set(BODY_JOINTS))==32


def test_quintic_peak_no_false_75_percent_inflation():
    assert peak_absolute([0,0,30,-60,30])==pytest.approx(1.875,rel=1e-8)
    assert peak_absolute([0,60,-180,120])==pytest.approx(10/np.sqrt(3),rel=1e-8)
    assert peak_absolute([60,-360,360])==pytest.approx(60,rel=1e-8)


@pytest.mark.parametrize('seed',range(12))
def test_exact_extrema_bounds_dense_real_curves(tmp_path,seed):
    rng=np.random.default_rng(seed)
    values=rng.uniform(-40,40,5)
    qs=[pose(Joint_Right_Shoulder_Inner=float(v)) for v in values]
    if seed%2:qs[2]['Joint_Right_Gripper']=180
    r,segs,_=plan(compile_action(draft(qs,hold=0 if seed%3 else .3)),pose(),tmp_path)
    for s in segs:
        for order in (1,2,3):
            c=s['coefficients'].copy()
            for _ in range(order):c=c[1:]*np.arange(1,len(c))[:,None]
            grid=np.linspace(0,1,4001)
            peaks=np.max(abs(sum(c[k,:,None]*grid**k for k in range(len(c)))),axis=1)/s['seconds']**order
            assert all(v<=limits(n)[order-1]*1.00001 for n,v in zip(ALL_JOINTS,peaks))
        for n in ALL_JOINTS:
            assert abs(evaluate(s,1)[n]-s['target'][n])<1e-7
    for i,(a,b) in enumerate(zip(segs,segs[1:])):
        for order in (0,1,2):
            assert max(abs(evaluate(a,1,order)[n]-evaluate(b,0,order)[n]) for n in ALL_JOINTS)<1e-7


def test_numerical_root_failure_keeps_conservative_bound(monkeypatch):
    def fail(*a,**k):raise np.linalg.LinAlgError('test')
    monkeypatch.setattr(np.polynomial.polynomial,'polyroots',fail)
    assert peak_absolute([0,0,30,-60,30])>=1.875


def test_gripper_does_not_slow_unrelated_body_by_75_percent(tmp_path):
    r,s,_=plan(compile_action(draft([pose(Joint_Right_Gripper=330),pose(Joint_Right_Shoulder_Inner=30,Joint_Right_Gripper=330)])),pose(),tmp_path)
    assert s[1]['seconds']<5.1 # 320编码单位/120，原五次曲线最小时间5秒。

