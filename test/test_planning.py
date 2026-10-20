"""Numerical policy tests; no ROS node or hardware commands."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from autolife_shelf_grasp_306.planning import (Planner,IKFailure,CollisionFailure,
    approach_candidates,reference_position,distance_to_polyline)


def coupled_fixture():
    # Analytic kinematic fixture: translating X necessarily changes roll.
    # Proves soft orientation works without accepting position error/jumps.
    p=Planner.__new__(Planner);p.cfg={'orientation_relax_deg':30.}
    p.names=list('abcdefg');p.bounds=np.array([[-1.]*7,[1.]*7])
    def pose(q,tcp):
        x=np.deg2rad([q[n] for n in p.names])
        return x[:3],Rotation.from_rotvec([5*x[0],0,0]).as_matrix()
    p.pose=pose
    return p,{n:0. for n in p.names}


def test_flexible_solves_position_strict_cannot():
    p,q=coupled_fixture();goal=np.array([.05,0,0])
    with pytest.raises(IKFailure):p.solve(q,np.zeros(3),goal,np.eye(3))
    result=p.solve(q,np.zeros(3),goal,np.eye(3),True)
    position,rotation=p.pose(result,None)
    assert np.linalg.norm(position-goal)<.004
    assert 3.<np.rad2deg(Rotation.from_matrix(rotation).magnitude())<30.
    assert max(abs(result[n]-q[n]) for n in q)<=8.00001


def test_flexible_not_unbounded():
    p,q=coupled_fixture()
    with pytest.raises(IKFailure):p.solve(q,None,[.13,0,0],np.eye(3),True)
    with pytest.raises(IKFailure):p.solve(q,None,[.05,0,0],np.eye(3),True,limit_deg=5.)


def test_route_retries_with_full_checks():
    p=Planner.__new__(Planner);p.cfg={};calls=[]
    def route(*args):
        calls.append(args[4])
        if not args[4]:raise IKFailure('strict unreachable')
        return {'orientation_mode':'flexible'}
    p._route=route
    assert p.plan({},None,None,{})['orientation_mode']=='flexible'
    assert calls==[False]*8+[True]
    attempted=[]
    def new_approach(*args):
        attempted.append(args[6])
        if args[6]==0:raise CollisionFailure('self collision')
        return {'orientation_mode':'horizontal'}
    p._route=new_approach
    assert p.plan({},None,None,{})['orientation_mode']=='horizontal'
    assert attempted==[0]*16+[-5]
    def collision(*args):raise ValueError('self collision')
    p._route=collision
    with pytest.raises(ValueError,match='self collision'):p.plan({},None,None,{})


def test_small_angles_and_forward_alternatives_first():
    candidates=list(approach_candidates())
    angles=[abs(c[0]) for c in candidates]
    assert angles==sorted(angles)
    assert sorted(set(angles))==list(range(0,61,5))
    assert {c[2] for c in candidates if c[0]==0}=={'direct','elbow_positive','elbow_negative','outward_4','outward_8',
        'joint_direct','joint_elbow_positive','joint_elbow_negative'}
    p=Planner.__new__(Planner);p.cfg={};attempted=[]
    def route(*args):
        attempted.append((args[6],args[7]))
        if args[7]!='outward_4':raise CollisionFailure('direct blocked')
        return {'approach_yaw_deg':args[6]}
    p._route=route
    assert p.plan({},None,None,{})['approach_yaw_deg']==0
    assert all(yaw==0 for yaw,profile in attempted)


def test_detour_preserves_endpoints_and_checks_curve_not_shortcut():
    a=np.array([0.,0.,0.]);b=np.array([.2,0.,0.]);offset=np.array([0.,-.08,.02])
    points=np.array([reference_position(a,b,t,offset) for t in np.linspace(0,1,80)])
    assert np.allclose(points[0],a) and np.allclose(points[-1],b)
    assert distance_to_polyline(reference_position(a,b,.5,offset),points)<.0001
    assert distance_to_polyline((a+b)/2,points)>.01


def test_posture_bias_retains_pose_and_step_limits():
    p,q=coupled_fixture();target=np.zeros(7);target[4]=.4
    result=p.solve(q,None,[0,0,0],np.eye(3),posture_target=target)
    assert 0<result['e']<=8.00001
    position,rotation=p.pose(result,None)
    assert np.linalg.norm(position)<.004


def test_late_failure_does_not_repeat_pregrasp_only_variants():
    p=Planner.__new__(Planner);p.cfg={};calls=[]
    def route(*args):
        calls.append((args[6],args[4],args[7]))
        if args[6]==0:
            p._candidate_stage=3
            raise IKFailure('carry endpoint infeasible')
        return {}
    p._route=route
    p.plan({},None,None,{})
    assert calls==[(0.,flex,profile) for flex in [False,True] for profile in
                  ['direct','joint_direct','joint_elbow_positive','joint_elbow_negative']]+[(-5.,False,'direct')]


def test_different_joint_branch_precedes_larger_grasp_angle():
    p=Planner.__new__(Planner);p.cfg={};calls=[]
    def route(*args):
        profile=args[7]
        calls.append((args[6],profile));p._candidate_stage=3
        if profile!='joint_direct':raise IKFailure('wrist limit on withdrawal')
        return {'approach_yaw_deg':args[6]}
    p._route=route
    result=p.plan({},None,None,{})
    assert result['approach_yaw_deg']==0.
    assert calls==[(0.,'direct'),(0.,'joint_direct')]


def test_no_tail_retry_for_grasp_failure_or_collision():
    for stage,error in [(1,IKFailure),(3,CollisionFailure)]:
        p=Planner.__new__(Planner);p.cfg={};calls=[]
        def route(*args):
            assert len(args)==8
            calls.append(args[6]);p._candidate_stage=stage
            if args[6]==0:raise error('blocked')
            return {}
        p._route=route;p.plan({},None,None,{})
        assert calls==[0.]*8+[-5.]


def test_withdraw_freedom_preserves_initial_clearance_and_is_bounded():
    from autolife_shelf_grasp_306.planning import withdrawal_freedom
    for d in [-.01,0.,.02,.06]:assert withdrawal_freedom(d)==0.
    assert withdrawal_freedom(.08)==pytest.approx(.5)
    assert withdrawal_freedom(.1)==pytest.approx(1.)
    assert withdrawal_freedom(.2)==1.


def test_retreat_does_not_relax_grasp_or_position_accuracy():
    p,q=coupled_fixture();p.cfg['up_torso']=[1.,0.,0.]
    # This fixture must rotate about UP to translate. Heading can adapt after
    # clearance; strict grasp still rejects the exact same request.
    result=p.solve(q,None,[.05,0,0],np.eye(3),retreat_freedom=1.)
    assert np.linalg.norm(p.pose(result,None)[0]-[.05,0,0])<.004
    assert max(abs(result[n]-q[n]) for n in q)<=8.00001
    with pytest.raises(IKFailure):p.solve(q,None,[.05,0,0],np.eye(3))
    with pytest.raises(IKFailure):p.solve(q,None,[1.,0,0],np.eye(3),retreat_freedom=1.)
    p.cfg['up_torso']=[0.,0.,1.]
    with pytest.raises(IKFailure):p.solve(q,None,[.13,0,0],np.eye(3),retreat_freedom=1.)
