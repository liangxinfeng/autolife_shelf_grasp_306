import copy
from types import SimpleNamespace
import numpy as np
import pytest
from autolife_shelf_grasp_306 import reset_recovery as m


def fixture():
    names=m.BODY+m.NECK+m.ARMS['left']+m.ARMS['right']
    q={n:0. for n in names};kin=SimpleNamespace(joint_q={n:i for i,n in enumerate(names)},
        lower=np.full(len(names),-1.),upper=np.full(len(names),1.))
    i=kin.joint_q['Joint_Neck_Pitch'];kin.lower[i]=np.deg2rad(-45);kin.upper[i]=np.deg2rad(30)
    goal=dict(q);q['Joint_Neck_Pitch']=-46.13;goal['Joint_Neck_Pitch']=-20
    return q,goal,kin


def test_small_neck_overshoot_targets_inside_but_source_is_not_clipped():
    q,goal,kin=fixture();before=copy.deepcopy(q)
    target,changes=m.recovery_target(q,goal,kin)
    assert q==before and target['Joint_Neck_Pitch']==pytest.approx(-43)
    assert set(changes)=={'Joint_Neck_Pitch'}
    q['Joint_Neck_Pitch']=31.
    target,changes=m.recovery_target(q,goal,kin)
    assert target['Joint_Neck_Pitch']==pytest.approx(28.)


@pytest.mark.parametrize('case',['large_neck','arm','body','bad_goal','nan'])
def test_unapproved_violations_reject(case):
    q,goal,kin=fixture()
    if case=='large_neck':q['Joint_Neck_Pitch']=-48
    elif case=='arm':q[m.ARMS['left'][0]]=-60
    elif case=='body':q[m.BODY[0]]=-60
    elif case=='bad_goal':goal['Joint_Neck_Pitch']=-46
    else:q['Joint_Neck_Pitch']=float('nan')
    with pytest.raises(ValueError):m.recovery_target(q,goal,kin)


@pytest.mark.parametrize('fail',[False,True])
def test_real_geometry_checked_and_strict_bounds_restored(fail):
    q,goal,kin=fixture();target,changes=m.recovery_target(q,goal,kin)
    low=kin.lower.copy();high=kin.upper.copy();seen=[]
    def check(values):
        seen.append(values.copy())
        assert values['Joint_Neck_Pitch']>=np.rad2deg(kin.lower[kin.joint_q['Joint_Neck_Pitch']])-1e-8
        if fail:raise ValueError('碰撞')
    guard=SimpleNamespace(kin=kin,cache=set(),check=check)
    path=SimpleNamespace(checks=lambda:[q,target])
    if fail:
        with pytest.raises(ValueError,match='碰撞'):m.check_recovery_geometry(guard,q,path,changes)
    else:assert m.check_recovery_geometry(guard,q,path,changes)==2
    assert seen[0]['Joint_Neck_Pitch']==-46.13
    assert np.array_equal(kin.lower,low) and np.array_equal(kin.upper,high) and not guard.cache


def test_recovery_cannot_move_farther_out_or_move_arms():
    q,goal,kin=fixture();target,changes=m.recovery_target(q,goal,kin)
    with pytest.raises(ValueError):m.check_recovery_sample(q,{**q,'Joint_Neck_Pitch':-46.2},changes)
    with pytest.raises(ValueError):m.check_recovery_sample(q,{**q,m.ARMS['left'][0]:.1},changes)


@pytest.mark.parametrize('mode',['ok','stuck','outward','other_arm','rejected','stale_goal','clamped'])
def test_execution_recovery_is_neck_only_and_restores_tolerance(mode):
    q,goal,kin=fixture();target,changes=m.recovery_target(q,goal,kin);t=[0.];held=[]
    class Node:
        def __init__(self):
            self.params={'goal_tolerance_deg':1.,'max_single_step_deg':5.};self.calls=[];self._active_goals={}
            if mode=='stale_goal':self._active_goals['neck']={'targets':{n:target[n] for n in m.NECK},'speed_scale':.1}
        def get_parameter(self,n):return SimpleNamespace(value=self.params[n])
        def set_parameters(self,values):
            self.params.update({v.name:v.value for v in values});return [SimpleNamespace(successful=True) for _ in values]
        def apply(self,group,values,speed_scale):
            self.calls.append(group);assert group=='neck' and speed_scale==.1
            assert speed_scale*self.params['max_single_step_deg']==pytest.approx(.1)
            assert values['Joint_Neck_Pitch']>-45
            if mode not in ('rejected','stale_goal'):
                accepted=dict(values)
                if mode=='clamped':accepted['Joint_Neck_Pitch']=-42.
                self._active_goals[group]={'targets':accepted,'speed_scale':speed_scale}
    node=Node()
    def checked():
        t[0]+=.1;v=dict(q)
        if mode=='ok':v['Joint_Neck_Pitch']=min(target['Joint_Neck_Pitch'],q['Joint_Neck_Pitch']+t[0]*2)
        if mode=='outward' and t[0]>.2:v['Joint_Neck_Pitch']-=1
        if mode=='other_arm' and t[0]>.2:v[m.ARMS['left'][0]]=1
        return v
    segment={'waypoints':[q,target],'neck_recovery':changes}
    if mode=='ok':m.execute_recovery(node,segment,checked,lambda n,g:held.extend(g),lambda:t[0])
    else:
        with pytest.raises(ValueError):m.execute_recovery(node,segment,checked,lambda n,g:held.extend(g),lambda:t[0])
    assert node.params=={'goal_tolerance_deg':1.,'max_single_step_deg':5.}
    if mode in ('rejected','stale_goal','clamped'):assert t[0]<.2
    assert held==(['neck'] if mode=='ok' else [])


@pytest.mark.parametrize('mode',['ok','invalid_old_speed','locked','no_subscriber','busy'])
def test_recovery_real_bridge_acceptance_without_ros_or_publishers(mode,monkeypatch):
    """Use actual parser/apply chain, without initializing ROS or any hardware."""
    from autolife_handeye_calibration_306.bridge import CalibrationBridge
    q,goal,kin=fixture();target,changes=m.recovery_target(q,goal,kin);clock=[0.];held=[]
    class OfflineBridge(CalibrationBridge):
        def __init__(self):
            self.gui_call=False;self.gate=lambda group:'';self.sampling=False;self.conflict=''
            self._current=dict(q);self._preview={};self._active_goals={};self.progress_watch={}
            self._goal_sent_groups=set();self._last_event='';self.manual_speed_scale=1.
            self._hardware_enabled=mode!='locked';self._hardware_ready=True
            self._action_state={} if mode=='busy' else None
            # No publish method exists: this test cannot issue motor commands.
            self._whole_body_pub=SimpleNamespace(get_subscription_count=lambda:0 if mode=='no_subscriber' else 1)
            self.params={'goal_tolerance_deg':1.,'max_single_step_deg':5.}
        def _is_feedback_fresh(self):return True
        def _communication_lost_joints(self,names):return []
        def _event(self,message,level='info'):self._last_event=message
        def _lock(self,reason):self._hardware_ready=False;self._last_event=reason
        def get_parameter(self,n):return SimpleNamespace(value=self.params[n])
        def set_parameters(self,values):
            self.params.update({v.name:v.value for v in values})
            return [SimpleNamespace(successful=True) for _ in values]
    node=OfflineBridge()
    if mode=='invalid_old_speed':monkeypatch.setattr(m,'RECOVERY_SPEED_SCALE',.02)
    def checked():
        clock[0]+=.1
        record=node._active_goals.get('neck')
        if record:node._current.update(record['targets'])
        return dict(node._current)
    segment={'waypoints':[q,target],'neck_recovery':changes}
    if mode=='ok':
        m.execute_recovery(node,segment,checked,lambda n,g:held.extend(g),lambda:clock[0])
        assert held==['neck'] and node._active_goals['neck']['targets']=={n:target[n] for n in m.NECK}
    else:
        with pytest.raises(ValueError,match='指令未被接受'):
            m.execute_recovery(node,segment,checked,lambda n,g:held.extend(g),lambda:clock[0])
        assert clock[0]<.2 and not held and not node._active_goals
        if mode=='invalid_old_speed':assert 'speed_scale must be between' in node._last_event
    assert node.params=={'goal_tolerance_deg':1.,'max_single_step_deg':5.}
