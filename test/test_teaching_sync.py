from types import SimpleNamespace as NS
import pytest
from autolife_shelf_grasp_306.teaching_sync import TeachingSyncSession
from control_center_joint_control.joint_definitions import BODY_JOINTS


def fixture(monkeypatch):
    clock=[10.];monkeypatch.setattr('time.monotonic',lambda:clock[0])
    future=NS(done=lambda:False,result=lambda:NS(success=True,message='ok'))
    calls=[];heartbeat=[];exits=[]
    node=NS(_hardware_ready=True,_is_feedback_fresh=lambda:True,_feedback_monotonic=10.,
        _current={n:0. for n in BODY_JOINTS},_details={n:dict(speed=0.) for n in BODY_JOINTS},
        _communication_lost_joints=lambda names:[])
    client=NS(service_is_ready=lambda:True,call_async=lambda req:(calls.append(req.data),future)[1])
    node.create_client=lambda *args:client
    node.create_publisher=lambda typ,topic,qos:NS(publish=lambda msg:(heartbeat if 'heartbeat' in topic else exits).append(msg.data))
    session=TeachingSyncSession(node);session.begin()
    return session,clock,future,calls,heartbeat,exits


def step(s,c,dt=.05):c[0]+=dt;s.node._feedback_monotonic=c[0];return s.poll()


def test_waits_for_ack_and_distinct_stable_feedback(monkeypatch):
    s,c,f,calls,h,_=fixture(monkeypatch)
    assert not s.poll() and calls==[True] and not h
    assert not step(s,c)
    f.done=lambda:True
    assert not step(s,c) and h
    for _ in range(5):assert not step(s,c)
    for _ in range(3):ready=step(s,c)
    assert ready and s.active and calls==[True]


@pytest.mark.parametrize('failure',['missing','timeout','reject','stale','drift','frozen','lost'])
def test_never_enters_unverified_session(monkeypatch,failure):
    s,c,f,_,_,_=fixture(monkeypatch)
    if failure=='missing':s.client.service_is_ready=lambda:False
    assert not s.poll()
    if failure=='missing':c[0]+=2.1
    elif failure=='timeout':c[0]+=12.1
    else:
        f.done=lambda:True
        if failure=='reject':f.result=lambda:NS(success=False,message='busy')
        elif failure=='stale':s.node._is_feedback_fresh=lambda:False
        elif failure=='drift':s.node._current[BODY_JOINTS[0]]=1.01
        elif failure=='lost':s.node._communication_lost_joints=lambda names:[BODY_JOINTS[0]]
    with pytest.raises(ValueError):
        for _ in range(100):
            assert not s.poll()
            c[0]+=.05
            if failure!='frozen':s.node._feedback_monotonic=c[0]


def test_exit_after_cancelled_pending_acquisition(monkeypatch):
    s,c,f,calls,h,exits=fixture(monkeypatch);s.poll()
    f.done=lambda:True
    assert s.close() and calls==[True,False] and len(exits)==3
    assert not s.active and not s.requested


def test_active_lease_heartbeat_and_feedback_guard(monkeypatch):
    s,c,f,calls,h,_=fixture(monkeypatch);s.poll();f.done=lambda:True;s.poll()
    count=len(h);c[0]+=.11;s.tick();assert len(h)==count+1
    s.node._is_feedback_fresh=lambda:False
    with pytest.raises(ValueError):s.tick()
