import numpy as np
import pytest
from autolife_shelf_grasp_306.motion import Arrival,Follower,GoalSender,configure_grasp_path,configure_reset_path
from autolife_shelf_grasp_306.core import execution_blockers,open_tolerance,check_plan_identity
from autolife_handeye_calibration_306.verification_trajectory import JointPath


def test_reset_arrival_does_not_wait_for_goal_bookkeeping():
    detector=Arrival({'a':20.},1.)
    assert not detector.update({'a':19.4},0)
    assert detector.update({'a':19.4},.4)


def test_moving_or_outside_is_not_arrived():
    d=Arrival({'a':20},1.)
    assert not d.update({'a':19.2},0)
    assert not d.update({'a':19.8},.4)
    assert not d.update({'a':18.9},.5)
    assert not d.update({'a':19.9},.6)
    assert d.update({'a':19.9},1.0)


def test_invalid_feedback_rejected():
    with pytest.raises(ValueError):Arrival({'a':0},1).update({'a':np.nan},0)


def run_follower(lead):
    path=JointPath(['a'],[{'a':0.},{'a':40.}]);path.max_speed=min(path.max_speed,24)
    f=Follower(path,0,lead);actual=0.;target=0.;largest=0.
    for i in range(1,3000):
        t=i*.01
        command=f.step({'a':actual},t)
        if command is not None:target=command['a']
        largest=max(largest,abs(target-actual))
        if i%10==0:actual=target
        if f.final and abs(actual-40)<.01:return t,largest
    pytest.fail('follower never completed')


def test_reset_faster_without_backlog_or_excess_lead():
    slow,ls=run_follower(1.2);fast,lf=run_follower(2.5)
    assert fast<slow and ls<=1.2+1e-6 and lf<=2.5+1e-6


def test_supervised_permission_not_fake_validation():
    cfg={'hardware_execution':True,'supervised_experiment':True,'camera_tcp_validated':False}
    assert not execution_blockers(cfg) and cfg['camera_tcp_validated'] is False
    assert execution_blockers({'supervised_experiment':True})


def test_open_tolerance_is_bounded():
    assert open_tolerance({})==4.
    for value in (0.,6.,float('nan')):
        with pytest.raises(ValueError):open_tolerance({'gripper_open_tolerance_deg':value})


def test_wrong_product_run_and_stale_plan_rejected():
    p={'plan_id':'new','run_id':'window-A','product':{'id':'tea'}}
    check_plan_identity(p,'window-A','new','tea')
    for args in [('window-B','new','tea'),('window-A','old','tea'),('window-A','new','water'),('window-A',None,'tea')]:
        with pytest.raises(ValueError):check_plan_identity(p,*args)


def test_sender_skips_settled_and_duplicate_goals():
    sender=GoalSender()
    assert not sender.needed('neck',{'a':0.},{'a':0.},1.)
    assert sender.needed('left',{'a':3.},{'a':0.},1.)
    assert not sender.needed('left',{'a':3.},{'a':0.},1.)
    assert sender.needed('left',{'a':4.},{'a':1.},1.)


def test_grasp_speed_ceiling_and_curvature_acceleration():
    cfg={'max_joint_speed_deg_s':120.,'max_joint_acceleration_deg_s2':720.}
    straight=configure_grasp_path(JointPath(['a'],[{'a':0.},{'a':90.}]),cfg)
    assert straight.max_speed==120.
    path=configure_grasp_path(JointPath(['a','b'],[
        {'a':0.,'b':0.},{'a':8.,'b':1.},{'a':9.,'b':10.},{'a':20.,'b':12.}]),cfg)
    s=np.linspace(0,path.length,10000)
    v=np.abs(path.curve(s,1));a=np.abs(path.curve(s,2))
    assert np.max(v*path.max_speed)<=120.+1e-6
    assert np.max(a*path.max_speed**2+v*path.acceleration)<=720.+1e-6
    for config in ({'max_joint_speed_deg_s':1000.},{'max_joint_acceleration_deg_s2':float('nan')}):
        with pytest.raises(ValueError):configure_grasp_path(straight,config)


def test_reset_twice_previous_speed_without_changing_geometry():
    path=JointPath(['a'],[{'a':0.},{'a':110.}]);s=np.linspace(0,path.length,100)
    before=path.curve(s).copy();configure_reset_path(path,{})
    assert path.max_speed==pytest.approx(70.2)
    assert path.acceleration==pytest.approx(585./2)
    np.testing.assert_allclose(path.curve(s),before)


def test_curved_path_two_times_time_scaling():
    points=[{'a':0.,'b':0.},{'a':8.,'b':1.},{'a':9.,'b':10.},{'a':20.,'b':12.}]
    old=configure_grasp_path(JointPath(['a','b'],points),
        {'max_joint_speed_deg_s':60.,'max_joint_acceleration_deg_s2':180.})
    new=configure_grasp_path(JointPath(['a','b'],points),{})
    assert new.max_speed==pytest.approx(2*old.max_speed)
    assert new.acceleration==pytest.approx(4*old.acceleration)
    # Real feedback lag must still pause the clock, not accumulate catch-up.
    f=Follower(configure_grasp_path(JointPath(['a'],[{'a':0.},{'a':90.}]),{}),0,3.)
    for i in range(1,601):
        command=f.step({'a':0.},i/60)
        if command:assert command['a']<=3.+1e-6
    assert f.progress<=3.+1e-6 and not f.final


def test_follower_uses_available_lead_instead_of_discarding_tick():
    path=configure_grasp_path(JointPath(['a'],[{'a':0.},{'a':90.}]),{})
    f=Follower(path,0,3.);f.progress=2.8;f.command=path.at(2.8);f.velocity=60.
    command=f.step({'a':0.},.02)
    assert command is not None and 2.99<=command['a']<=3.
    assert f.limited_ticks==1 and 0<f.velocity<60.
    # Persistent stall still cannot advance beyond the same lead bound.
    for i in range(2,300):f.step({'a':0.},i*.02)
    assert f.progress<=3.+1e-6
