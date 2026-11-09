from types import SimpleNamespace as NS
import pytest
from autolife_shelf_grasp_306.sync_session import PositionSyncSession
from autolife_shelf_grasp_306.task_pause import BridgePause
from control_center_joint_control.joint_definitions import ALL_JOINTS


def fake_node(monkeypatch,loaded_available=True):
    clock=[100.];calls=[];pubs=[]
    monkeypatch.setattr('autolife_shelf_grasp_306.sync_session.time.monotonic',lambda:clock[0])
    class Client:
        def __init__(self,name):self.name=name
        def service_is_ready(self):return loaded_available or 'loaded_session' not in self.name
        def call_async(self,r):
            calls.append((self.name,r.data))
            return NS(done=lambda:True,result=lambda:NS(success=True,message='ok'))
    n=NS(_hardware_ready=True,_feedback_monotonic=100.,snapshot=lambda:{},_is_feedback_fresh=lambda:True,
         create_client=lambda _,name:Client(name),create_publisher=lambda _,name,*a:NS(publish=lambda msg:pubs.append((name,msg.data))))
    def spin():
        clock[0]+=.04;n._feedback_monotonic=clock[0]
    return n,spin,calls,pubs


@pytest.mark.parametrize('loaded',[False,True])
def test_routes_entry_and_exit_and_waits_fresh_feedback(monkeypatch,loaded):
    n,spin,calls,pubs=fake_node(monkeypatch)
    s=PositionSyncSession(n,preserve_gripper_targets=loaded);s.enter(spin)
    assert s.active and n._feedback_monotonic>100.1
    assert ('loaded_session' in calls[0][0])==loaded
    assert s.close() and len(calls)==2 and calls[-1][1] is False
    exits=[p for p in pubs if 'session_command' in p[0]]
    assert len(exits)==(0 if loaded else 3)


def test_missing_loaded_service_does_not_fallback_or_enable(monkeypatch):
    n,spin,calls,pubs=fake_node(monkeypatch,False)
    s=PositionSyncSession(n,preserve_gripper_targets=True)
    with pytest.raises(ValueError,match='入口不可用'):s.enter(spin)
    assert not calls and not pubs and not s.requested


def test_grasp_finish_does_not_send_ordinary_exit(monkeypatch):
    n,spin,calls,pubs=fake_node(monkeypatch)
    s=PositionSyncSession(n);s.enter(spin);assert s.close(preserve_grippers=True)
    assert 'grasp_finish' in calls[-1][0] and not any('session_command' in p[0] for p in pubs)


@pytest.mark.parametrize('loaded',[False,True])
def test_pause_holds_body_without_overwriting_loaded_jaws(tmp_path,loaded):
    q={name:0. for name in ALL_JOINTS};q['Joint_Right_Gripper']=160.
    writes=[];clock=[100.]
    n=NS(_hardware_ready=True,_is_feedback_fresh=lambda:True,_communication_lost_joints=lambda _:[],
        _current=q,_whole_body_pub='body',_gripper_pub='jaw',preserve_loaded_grippers=loaded,
        _publish_vendor=lambda p,m:writes.append((p,m)),_lock=lambda reason:pytest.fail(reason),
        _active_goals={},progress_watch={})
    g=BridgePause(n,tmp_path,lambda:clock[0]);(tmp_path/'task_pause').touch()
    assert g.tick() and len(writes)==(1 if loaded else 2)
    clock[0]+=10;assert g.tick()
    (tmp_path/'task_pause').unlink();assert not g.tick()
    assert g.now()==100. and len(writes)==(1 if loaded else 2)
