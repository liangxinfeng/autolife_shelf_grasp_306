"""Compiled FK/Jacobians and deferred validation; no robot communication."""
import time
from types import SimpleNamespace
import numpy as np
import pinocchio as pin
import pytest
from scipy.spatial.transform import Rotation
from autolife_handeye_calibration_307.core import Kinematics
from autolife_shelf_grasp_306 import planning as m


def arm(tmp_path):
    p=m.Planner.__new__(m.Planner);p.names=[f'j{i}' for i in range(7)]
    xml=['<robot name="test"><link name="root"/>']
    axes=['0 0 1','0 1 0','1 0 0']
    for i,n in enumerate(p.names):
        parent='root' if i==0 else f'link{i-1}'
        xml.append(f'<link name="link{i}"/><joint name="{n}" type="revolute"><parent link="{parent}"/>'
                   f'<child link="link{i}"/><origin xyz=".06 .01 .02" rpy=".1 -.2 .05"/>'
                   f'<axis xyz="{axes[i%3]}"/><limit lower="-2" upper="2" effort="10" velocity="3"/></joint>')
    xml.append('</robot>');text=''.join(xml)
    path=tmp_path/'arm.urdf';path.write_text(text)
    p.pose_model=pin.buildModelFromXML(text);p.pose_data=p.pose_model.createData()
    p.pose_q=pin.neutral(p.pose_model)
    p.pose_indices=[p.pose_model.joints[p.pose_model.getJointId(n)].idx_q for n in p.names]
    p.pose_velocity_indices=[p.pose_model.joints[p.pose_model.getJointId(n)].idx_v for n in p.names]
    p.torso_frame=p.pose_model.getFrameId('root');p.wrist_frame=p.pose_model.getFrameId('link6')
    return p,Kinematics(path)


def test_compiled_fk_and_analytic_jacobian_equivalent(tmp_path):
    p,fk=arm(tmp_path);rng=np.random.default_rng(31);tcp=np.array([.23,.007,-.016])
    for _ in range(30):
        q=dict(zip(p.names,rng.uniform(-100,100,7)))
        position,R,Jp,Jw=p.pose_jacobian(q,tcp)
        T=fk.fk('root','link6',q)
        assert np.allclose(position,T[:3,3]+T[:3,:3]@tcp,atol=1e-12,rtol=0)
        assert np.allclose(R,T[:3,:3],atol=1e-12,rtol=0)
        desired=Rotation.from_rotvec([.2,-.1,.3]).as_matrix()@R
        Jr=pin.Jlog3(desired.T@R)@R.T@Jw
        for i,n in enumerate(p.names):
            a=dict(q);b=dict(q);a[n]+=np.rad2deg(1e-6);b[n]-=np.rad2deg(1e-6)
            pa,Ra=p.pose(a,tcp);pb,Rb=p.pose(b,tcp)
            numerical=(pa-pb)/2e-6
            angular=(Rotation.from_matrix(desired.T@Ra).as_rotvec()-Rotation.from_matrix(desired.T@Rb).as_rotvec())/2e-6
            assert np.allclose(Jp[:,i],numerical,atol=1e-7,rtol=0)
            assert np.allclose(Jr[:,i],angular,atol=1e-7,rtol=0)


@pytest.mark.parametrize('dense_collision',[False,True])
@pytest.mark.parametrize('profile',['direct','joint_direct'])
def test_all_four_stages_still_require_dense_validation(monkeypatch,dense_collision,profile):
    p=m.Planner.__new__(m.Planner);p.names=list('abcdefg')
    p.cfg={'arm':'left','gripper_max_width_m':.15,'approach_yaw_deg':0.,'up_torso':[0,0,1],
           'T_wrist_tool_quat_xyzw':[0,0,0,1],'pregrasp_m':.1,'lift_m':.03,'withdraw_m':.15,
           'max_joint_speed_deg_s':120.}
    p.guard=SimpleNamespace(cache=set());p.fk=SimpleNamespace(sha='test')
    p.joint_limit_policy={'version':1,'bounds_deg':{n:[-180,180] for n in p.names}}
    p.pose=lambda q,tcp:(np.array([q[n] for n in 'abc']),np.eye(3))
    p.solve=lambda source,tcp,point,*args,**kwargs:{**source,**dict(zip('abc',point))}
    p.check=lambda *args:None
    calls=[]
    class FakePath:
        def __init__(self,names,waypoints):self.waypoints=waypoints;self.max_speed=20.
        def checks(self):
            # Four stages solved first, but every interpolated state is checked
            # before a plan may return. A late dense collision must still reject.
            assert p._candidate_stage==len(calls)
            calls.append(self)
            if dense_collision and len(calls)==4:raise m.CollisionFailure('dense collision')
            yield self.waypoints[-1]
    monkeypatch.setattr(m,'JointPath',FakePath)
    monkeypatch.setattr(m,'configure_grasp_path',lambda path,cfg:path)
    source={n:0. for n in p.names}
    if dense_collision:
        with pytest.raises(m.CollisionFailure,match='dense collision'):
            p._route(source,np.zeros(3),np.array([.25,0,0]),{'diameter_m':.07},False,time.monotonic(),profile=profile)
    else:
        result=p._route(source,np.zeros(3),np.array([.25,0,0]),{'diameter_m':.07},False,time.monotonic(),profile=profile)
        assert result['self_collision_checked'] and len(result['segments'])==4
    assert len(calls)==4


def test_adaptive_withdraw_checks_full_path_and_gates_orientation(monkeypatch):
    p=m.Planner.__new__(m.Planner);p.names=list('abcdefg')
    p.cfg={'arm':'left','gripper_max_width_m':.15,'approach_yaw_deg':0.,'up_torso':[0,0,1],
           'T_wrist_tool_quat_xyzw':[0,0,0,1],'pregrasp_m':.1,'lift_m':.03,'withdraw_m':.2,
           'max_joint_speed_deg_s':120.}
    p.guard=SimpleNamespace(cache=set());p.fk=SimpleNamespace(sha='test')
    p.joint_limit_policy={'bounds_deg':{n:[-180,180] for n in p.names}}
    p.pose=lambda q,tcp:(np.array([q[n] for n in 'abc']),np.eye(3))
    solves=[];dense=[]
    def solve(source,tcp,point,*args,**kwargs):
        solves.append((p._candidate_stage,point.copy(),kwargs.get('retreat_freedom',0.)))
        return {**source,**dict(zip('abc',point))}
    p.solve=solve;p.check=lambda *args:None
    class FakePath:
        def __init__(self,names,waypoints):self.waypoints=waypoints;self.max_speed=20.
        def checks(self):
            dense.append(p._candidate_stage)
            yield self.waypoints[-1]
    monkeypatch.setattr(m,'JointPath',FakePath)
    monkeypatch.setattr(m,'configure_grasp_path',lambda path,cfg:path)
    source={n:0. for n in p.names};tcp=np.zeros(3);center=np.array([.25,0,0]);product={'diameter_m':.07}
    result=p._route(source,tcp,center,product,False,time.monotonic())
    assert dense==[0,1,2,3]
    assert result['withdraw_orientation_mode']=='adaptive'
    for index,point,freedom in solves:
        if index<3 or .25-point[0]<=.06+1e-9:assert freedom==0.
    assert any(index==3 and freedom==1. for index,point,freedom in solves)
