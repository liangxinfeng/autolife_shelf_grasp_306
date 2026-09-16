from types import SimpleNamespace
import numpy as np
import pytest
from autolife_shelf_grasp_306.sync_session import PositionSyncSession
from autolife_shelf_grasp_306.motion import transit_pose_ok,Arrival


class Publisher:
    def __init__(self):self.messages=[]
    def publish(self,msg):self.messages.append(msg.data)


class Node:
    _hardware_ready=True
    def __init__(self,success=True):self.requests=[];self.success=success
    def create_client(self,*a):return self
    def create_publisher(self,*a):return Publisher()
    def service_is_ready(self):return True
    def _is_feedback_fresh(self):return True
    def call_async(self,request):
        self.requests.append(request.data)
        return SimpleNamespace(done=lambda:True,result=lambda:SimpleNamespace(success=self.success,message='test refusal'))


def test_no_session_start_on_construction_or_idle_tick():
    node=Node();s=PositionSyncSession(node);s.tick()
    assert not node.requests and not s.heartbeat.messages and s.close()


def test_session_enters_and_explicitly_exits_holding():
    node=Node();s=PositionSyncSession(node);s.enter(lambda:None)
    assert s.active and node.requests==[True] and len(s.heartbeat.messages)==1
    s.tick();assert len(s.heartbeat.messages)==1
    assert s.close() and not s.active and node.requests==[True,False]
    assert len(s.exit_pub.messages)==3


def test_failed_entry_never_allows_motion_and_requests_cleanup():
    node=Node(False);s=PositionSyncSession(node)
    with pytest.raises(ValueError,match='拒绝'):s.enter(lambda:None)
    assert not s.active and not s.heartbeat.messages
    node.success=True;assert s.close() and node.requests==[True,False]


def test_hardware_loss_stops_lease_heartbeat():
    node=Node();s=PositionSyncSession(node);s.enter(lambda:None);node._hardware_ready=False
    node.last_lock_reason='Joint_Left_Shoulder_Outer 5秒无跟随，误差0.93°'
    with pytest.raises(ValueError,match='Shoulder_Outer.*0.93'):s.tick()
    assert len(s.heartbeat.messages)==1


def test_stale_feedback_is_not_reported_as_hardware_failure():
    node=Node();s=PositionSyncSession(node);s.enter(lambda:None)
    node._is_feedback_fresh=lambda:False
    with pytest.raises(ValueError,match='关节反馈过期'):s.tick()
    assert len(s.heartbeat.messages)==1


def test_transit_arrival_requires_cartesian_and_orientation_bounds():
    a=np.eye(4);b=np.eye(4);b[0,3]=.0063
    assert transit_pose_ok(a,b,np.zeros(3))
    b[0,3]=.0081;assert not transit_pose_ok(a,b,np.zeros(3))
    from scipy.spatial.transform import Rotation
    b=np.eye(4);b[:3,:3]=Rotation.from_euler('z',3,degrees=True).as_matrix()
    assert not transit_pose_ok(a,b,np.zeros(3))
    a[0,3]=np.nan;assert not transit_pose_ok(a,b,np.zeros(3))


def test_transit_small_static_residual_does_not_relax_grasp():
    transit=Arrival({'q':0},1.,.2);grasp=Arrival({'q':0},.35,.35)
    for t in [0.,.1,.21,.4]:
        accepted=transit.update({'q':-.96},t)
        assert not grasp.update({'q':-.96},t)
    assert accepted
