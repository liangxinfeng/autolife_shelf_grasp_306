from types import SimpleNamespace as S
import pytest
from autolife_shelf_grasp_306.feedback_state import NAMES,FeedbackNotReady,checked_snapshot,await_ready_snapshot
from autolife_shelf_grasp_306.pose_handoff import await_loaded_pose


def state(lost=(),fresh=True):
    return S(_current={n:0. for n in NAMES},_is_feedback_fresh=lambda:fresh,
             _communication_lost_joints=lambda names:list(lost))


@pytest.mark.parametrize('joint',NAMES)
def test_all_joints_offline_still_stop_motion(joint):
    with pytest.raises(FeedbackNotReady) as e:checked_snapshot(state([joint]))
    assert e.value.joints==[joint] and e.value.reason=='offline'


@pytest.mark.parametrize('kind',['place','observe'])
@pytest.mark.parametrize('arm',['left','right'])
def test_offline_neck_waits_before_motion_and_requires_new_stability(kind,arm):
    node=state();source=dict(node._current);ticks=[0];reports=[]
    def poll():ticks[0]+=1
    def clock():return 10.+ticks[0]*.025
    def snapshot():
        node._communication_lost_joints=lambda names: ['Joint_Neck_Pitch'] if ticks[0]<15 else []
        return checked_snapshot(node)
    q=await_loaded_pose(snapshot,poll,clock,source,'使能期间',kind,clock,reports.append,arm=arm)
    assert q==source and ticks[0]>=23
    assert any(r.get('unavailable_joints')==['Joint_Neck_Pitch'] for r in reports)
    assert reports[-1]['ok']


def test_sustained_neck_offline_is_never_ignored():
    node=state(['Joint_Neck_Pitch']);tick=[0]
    def poll():tick[0]+=1
    with pytest.raises(ValueError,match='准备期限.*Joint_Neck_Pitch'):
        await_loaded_pose(lambda:checked_snapshot(node),poll,lambda:tick[0]*.05,
            node._current,'使能期间','place',lambda:tick[0]*.05,lambda r:None)


def test_recovered_but_shifted_pose_cannot_continue():
    node=state();source=dict(node._current);ticks=[0]
    def poll():ticks[0]+=1
    def snapshot():
        if ticks[0]<10:raise FeedbackNotReady('离线',reason='offline',joints=['Joint_Neck_Pitch'])
        node._current['Joint_Neck_Pitch']=3.
        return checked_snapshot(node)
    with pytest.raises(ValueError,match='姿态改变'):
        await_loaded_pose(snapshot,poll,lambda:ticks[0]*.025,source,'使能期间','place',
            lambda:ticks[0]*.025,lambda r:None)


def test_cancel_during_prepare_is_immediate():
    node=state(['Joint_Neck_Pitch'])
    def poll():raise ValueError('用户停止')
    with pytest.raises(ValueError,match='用户停止'):
        await_ready_snapshot(lambda:checked_snapshot(node),poll,'使能期间')


def test_nan_not_treated_as_transient_offline():
    node=state();node._current[NAMES[0]]=float('nan')
    with pytest.raises(ValueError,match='非有限'):
        await_ready_snapshot(lambda:checked_snapshot(node),lambda:None,'准备')


def test_prepare_only_waits_for_typed_feedback_error():
    def bad():raise ValueError('硬件限位错误')
    with pytest.raises(ValueError,match='硬件限位错误'):await_ready_snapshot(bad,lambda:None,'准备')


@pytest.mark.parametrize('transient',[True,False])
def test_sync_transition_uses_same_offline_contract(monkeypatch,transient):
    import autolife_shelf_grasp_306.sync_session as module
    ticks=[10.];reports=[]
    class Client:
        def service_is_ready(self):return True
        def call_async(self,r):return S(done=lambda:True,result=lambda:S(success=True))
    class Node:
        _hardware_ready=True
        def create_client(self,*a):return Client()
        def create_publisher(self,*a):return S(publish=lambda m:reports.append(m))
        def _is_feedback_fresh(self):return True
        @property
        def _feedback_monotonic(self):return ticks[0]
        def snapshot(self):
            if not transient or ticks[0]<10.3:
                raise FeedbackNotReady('关节通信离线：Joint_Neck_Pitch',reason='offline',joints=['Joint_Neck_Pitch'])
            return {}
    monkeypatch.setattr(module.time,'monotonic',lambda:ticks[0])
    session=module.PositionSyncSession(Node())
    def poll():ticks[0]+=.025
    if transient:
        session.enter(poll);assert session.active and ticks[0]>=10.4
    else:
        with pytest.raises(ValueError,match='反馈未恢复'):session.enter(poll)
        assert not session.active and not reports
