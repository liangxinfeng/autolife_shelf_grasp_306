"""Opening-only terminal settling for taught actions; no relaxed arm targets."""
import math

OPENING_RESIDUAL_DEG = 5.0
OPENING_SETTLE_SEC = .5


def opening_modes(state, targets, jaw):
    """Classify release and its unchanged continuation, never ordinary closing."""
    from control_center_joint_control.joint_definitions import GRIPPER_JOINTS
    refs=state.setdefault('opening_references',{})
    result={}
    for n in GRIPPER_JOINTS:
        if n not in targets:continue
        goal=float(targets[n]);start=float(state['start_values'][n])
        ref=refs.get(n)
        if ref and (ref['loop']!=state['loop_index'] or abs(goal-ref['goal'])>jaw):
            refs.pop(n,None);ref=None
        # Ten degrees distinguishes a deliberate opening from measured-keyframe
        # quantization. Continuations stay anchored to the original release
        # target, not the last measured pose: no accumulating drift allowance.
        if start-goal>=10.:result[n]='release'
        elif ref and goal<=start+jaw:result[n]='hold_open'
    return result


def remember_opening_arrival(bridge, targets):
    state=bridge._action_state
    if state is None:return
    jaw=float(bridge.get_parameter('gripper_goal_tolerance_deg').value)
    modes=opening_modes(state,targets,jaw)
    refs=state.setdefault('opening_references',{})
    for n,mode in modes.items():
        if mode=='release':refs[n]=dict(goal=float(targets[n]),loop=state['loop_index'])


def settled_opening_arrival(bridge, targets):
    from control_center_joint_control.joint_definitions import GRIPPER_JOINTS
    state=bridge._action_state
    if state is None:return False
    def reject():
        state.pop('opening_arrival',None)
        return False
    if not bridge._is_feedback_fresh():return reject()
    body=float(bridge.get_parameter('goal_tolerance_deg').value)
    jaw=float(bridge.get_parameter('gripper_goal_tolerance_deg').value)
    modes=opening_modes(state,targets,jaw)
    observed={};exception=False
    for n,goal in targets.items():
        actual=float(bridge._current.get(n,float('nan')))
        detail=bridge._details.get(n,{})
        speed=float(detail.get('speed',float('nan')))
        if not all(math.isfinite(v) for v in (actual,goal,speed)) or detail.get('communication_lost',True):return reject()
        if abs(speed)>2.:return reject()
        observed[n]=actual;error=abs(actual-goal)
        if n not in GRIPPER_JOINTS:
            if error>body+1e-9:return reject()
            continue
        if error<=jaw+1e-9:continue
        # Require actual opening progress, or an unchanged target inherited
        # from a previously confirmed release. No closing/contact success.
        start=float(state['start_values'][n]);command=float(state['commanded_values'][n])
        if not math.isfinite(start) or not math.isfinite(command):return reject()
        travel=start-goal
        mode=modes.get(n)
        if not mode or error>OPENING_RESIDUAL_DEG+1e-9 or abs(command-goal)>.05:return reject()
        if mode=='release' and start-actual<max(2.,travel*.8):return reject()
        exception=True
    if not exception:return reject()
    stamp=float(bridge._feedback_monotonic)
    if not math.isfinite(stamp):return reject()
    key=(state['loop_index'],state['frame_index'])
    stable=state.get('opening_arrival')
    if (stable is None or stable['key']!=key or stamp<stable['last'] or stamp-stable['last']>.25
            or any(abs(v-stable['anchor'][n])>.4 for n,v in observed.items())):
        state['opening_arrival']=dict(key=key,start=stamp,last=stamp,count=1,anchor=observed)
        return False
    if stamp>stable['last']:
        stable['last']=stamp;stable['count']+=1
    reached=stable['count']>=4 and stamp-stable['start']>=OPENING_SETTLE_SEC
    if reached:remember_opening_arrival(bridge,targets)
    return reached
