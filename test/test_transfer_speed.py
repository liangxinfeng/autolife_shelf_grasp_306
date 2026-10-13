import numpy as np
import pytest
from autolife_handeye_calibration_306.verification_trajectory import JointPath
from autolife_shelf_grasp_306.motion import (configure_grasp_path,Follower,PathSpeedProfile,
    transfer_lead,cartesian_speed_profile)

def path():return configure_grasp_path(JointPath(['a'],[{'a':0.},{'a':80.}]),{})

@pytest.mark.parametrize('kind,phase,want',[
    ('grasp','pregrasp',8),('grasp','carry',8),('place','place_approach',8),
    ('place','place_return',8),('place','place_lower',3),('grasp','grasp',3),
    ('observe','observe_lift',3),('prepare','prepare',5)])
def test_only_transfer_lead_increased(kind,phase,want):
    assert transfer_lead({},phase,kind)==want

@pytest.mark.parametrize('value',[0,9,float('nan'),float('inf')])
def test_no_unbounded_transfer_lead(value):
    with pytest.raises(ValueError):transfer_lead({'transfer_feedback_lead_deg':value},'carry','grasp')

def test_exit_uses_measured_not_command_progress():
    p=path();f=Follower(p,0,8,transfer_from_s=20.)
    f.progress=22.;f.command=p.at(22.);f.step({'a':19.},.01)
    assert f.tracking['effective_lead_deg']<=3.
    f.step({'a':21.},.02)
    assert f.tracking['effective_lead_deg']==8.

def test_faster_transfer_still_bounded_when_motor_stalls():
    f=Follower(path(),0,8)
    for i in range(1,500):f.step({'a':0.},i*.01)
    assert not f.final and f.progress<=8.+1e-6
    with pytest.raises(ValueError,match='超过1°'):f.step({'a':-2.},5.)

def test_local_fk_caps_respect_every_checked_interval_and_contact_zone():
    p=path();s=np.linspace(0,80,801)
    # Narrow region with large Cartesian derivative; rest must not inherit it.
    ratios=np.where((s[:-1]>20)&(s[:-1]<24),.02,.002)
    x=np.r_[0.,np.cumsum(ratios*np.diff(s))]
    pos=np.c_[x,np.zeros_like(x),np.zeros_like(x)]
    limits=np.where(s>60,.03,.12)
    knots=cartesian_speed_profile(p,s,pos,limits);profile=PathSpeedProfile(p,knots)
    for i in range(len(s)-1):
        cap=profile.cap(s[i],0.)
        assert cap*ratios[i]*1.25<=min(limits[i],limits[i+1])+1e-8
    assert profile.cap(2.,0.)>profile.cap(22.,0.)*2
    assert profile.cap(65.,0.)<=.03/.002/1.25+1e-8

def test_exit_lead_does_not_change_checked_curve():
    p=path();before=p.curve(np.linspace(0,80,101)).copy()
    Follower(p,0,8,transfer_from_s=40)
    np.testing.assert_array_equal(before,p.curve(np.linspace(0,80,101)))

def test_simple_lag_model_improves_transfer_not_contact():
    def run(lead):
        p=path();f=Follower(p,0,lead);q=0.;target=0.
        for i in range(1,5000):
            c=f.step({'a':q},i*.01)
            if c:target=c['a']
            q+=(target-q)*(1-np.exp(-.01/.25))
            if f.final and abs(q-80)<.35:return i*.01
        pytest.fail('did not finish')
    assert run(8)<run(3)*.65
