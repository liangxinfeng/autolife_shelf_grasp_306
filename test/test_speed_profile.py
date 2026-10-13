import numpy as np
import pytest
from autolife_handeye_calibration_306.verification_trajectory import JointPath
from autolife_shelf_grasp_306.motion import Follower,PathSpeedProfile,configure_grasp_path


def path():
    p=configure_grasp_path(JointPath(['a'],[{'a':0.},{'a':60.}]),{})
    p.max_speed=20.;p.acceleration=40.
    return p


@pytest.mark.parametrize('knots',[
    [],[[0,10]],[[1,10],[60,10]],[[0,10],[59,10]],[[0,10],[0,4],[60,4]],
    [[0,float('nan')],[60,4]],[[0,21],[60,4]],[[0,0],[60,4]]])
def test_invalid_profile_rejected(knots):
    with pytest.raises(ValueError):PathSpeedProfile(path(),knots)


def simulate(profile):
    p=path();f=Follower(p,0,3.,profile);q={'a':0.};history=[]
    for i in range(1,4000):
        command=f.step(q,i*.01)
        if command:q=command
        history.append((f.progress,f.velocity))
        if f.final:return i*.01,np.array(history)
    pytest.fail('did not finish')


def test_fast_far_slow_near_without_an_intermediate_stop():
    slow,_=simulate([[0,5],[60,5]])
    fast,h=simulate([[0,20],[40,5],[60,5]])
    assert fast<slow*.7
    assert np.max(h[(h[:,0]>=40)&(h[:,0]<59),1])<=5.+1e-8
    assert np.min(h[(h[:,0]>39)&(h[:,0]<41),1])>0


def test_brakes_before_lower_speed_boundary():
    p=path();profile=PathSpeedProfile(p,[[0,20],[40,5],[60,5]])
    assert 5<profile.cap(38.,.01)<20
    assert profile.cap(40.,.01)==5


def test_profile_still_stops_on_stale_motion_and_offpath():
    p=path();f=Follower(p,0,3.,[[0,20],[40,5],[60,5]])
    for i in range(1,300):f.step({'a':0.},i*.01)
    assert f.progress<=3.+1e-8
    with pytest.raises(ValueError,match='超过1°'):f.step({'a':-2.},3.1)


def test_analytic_chord_bound_contains_dense_samples_on_realistic_paths():
    rng=np.random.default_rng(12)
    for _ in range(25):
        points=np.cumsum(rng.normal(0,5,(8,7)),axis=0)
        names=[str(i) for i in range(7)]
        p=configure_grasp_path(JointPath(names,[dict(zip(names,q)) for q in points]),{})
        f=Follower(p,0,3.)
        for lo,hi in sorted(rng.uniform(0,p.length,(8,2)),key=lambda pair:pair[0]):
            lo,hi=sorted((lo,hi));u=np.linspace(0,1,1001)
            error=np.max(np.abs(p.curve(lo+(hi-lo)*u)-(p.at(lo)+(p.at(hi)-p.at(lo))*u[:,None])))
            assert error<=f.chord_bound(lo,hi)+1e-8


def test_analytic_bound_is_tighter_than_global_curvature_bound():
    p=configure_grasp_path(JointPath(['a','b'],[{'a':i,'b':b} for i,b in enumerate([0,0,0,1,1,1,1])]),{})
    f=Follower(p,0,3.);s=np.linspace(0,p.length,1001)
    old=np.max(np.abs(p.curve(s,2)))*p.length**2/8
    assert f.chord_bound(0,p.length)<old*.5


@pytest.mark.parametrize('version',[2,3])
def test_protocol_two_validates_profiles_before_execution(tmp_path,version):
    import json
    from autolife_shelf_grasp_306.demo_protocol import validate_place
    from autolife_handeye_calibration_306.core import ARMS
    for name,data in [('demo_run.json',dict(grasp_plan_id='one',product_id='drink',settings={})),
                      ('demo_grasp_result.json',dict(status='completed',plan_id='one')),
                      ('demo_base_result.json',dict(status='completed'))]:
        (tmp_path/name).write_text(json.dumps(data))
    q0=dict.fromkeys(ARMS['left'],0.);q1={**q0,ARMS['left'][0]:10.}
    segments=[dict(phase=s,waypoints=[q0,q1],path_speed_cap=5.,path_speed_profile=[[0,5],[10,5]])
              for s in ('place_approach','place_lower','place_release_retreat','place_return')]
    plan=dict(kind='place',place_protocol_version=2,table_height_checked=True,parent_plan_id='one',
              product={'id':'drink'},demo_settings={},config={},segments=segments)
    if version==3:
        plan['place_protocol_version']=3
        segments.pop(2)
    validate_place(plan,tmp_path)
    segments[1]['path_speed_profile']=[[0,10],[10,10]]
    with pytest.raises(ValueError):validate_place(plan,tmp_path)
    segments[1]['path_speed_profile']=[[0,5],[10,5]]
    del segments[1]['path_speed_profile']
    with pytest.raises(ValueError):validate_place(plan,tmp_path)
