"""Opening-only terminal settling for taught actions; no relaxed arm targets."""
import math

OPENING_RESIDUAL_DEG = 5.0
OPENING_SETTLE_SEC = .5


def open_endpoint_observation(bridge,targets):
    """Eligible stationary open-region feedback, otherwise None; no mutation."""
    from control_center_joint_control.joint_definitions import GRIPPER_JOINTS, JOINT_LIMITS
    state=bridge._action_state
    if state is None:return None
    if not bridge._is_feedback_fresh():return None
    body=float(bridge.get_parameter('goal_tolerance_deg').value)
    jaw=float(bridge.get_parameter('gripper_goal_tolerance_deg').value)
    observed={};exception=False
    for n,goal in targets.items():
        actual=float(bridge._current.get(n,float('nan')))
        detail=getattr(bridge,'_details',{}).get(n,{})
        speed=float(detail.get('speed',float('nan')))
        command=float(state['commanded_values'].get(n,float('nan')))
        if not all(math.isfinite(v) for v in (actual,goal,speed,command)) or detail.get('communication_lost',True):return None
        error=abs(actual-goal);observed[n]=actual
        low=JOINT_LIMITS[n][0] if n in GRIPPER_JOINTS else None
        full_open=low is not None and low<=goal<=low+OPENING_RESIDUAL_DEG
        if full_open:
            start=float(state.get('start_values',{}).get(n,float('nan')))
            # A small movement entirely inside the open-end region is encoder
            # alignment, not proof of grasp/contact. Never accept an invalid
            # start, a yet-unissued goal, or feedback outside the endpoint band.
            if not math.isfinite(start) or start<low-1.e-6 or abs(command-goal)>.05:return None
            if not low-jaw<=actual<=low+OPENING_RESIDUAL_DEG or error>OPENING_RESIDUAL_DEG or abs(speed)>5.:return None
            exception=exception or error>jaw or abs(speed)>2.
        elif error>(jaw if n in GRIPPER_JOINTS else body) or abs(speed)>2.:
            return None
    return observed if exception else None


def settled_full_open_arrival(bridge, targets, *, channel='preparation'):
    """Confirm the open-end region using distinct encoder samples.

    Small initial opening corrections need not contain a ten-degree release.
    DM gripper speed telemetry is quantized (the observed stationary grippers
    reported up to 4.41 deg/s), so full-open jaws additionally use a bounded
    position window. Closing and intermediate openings retain normal bounds.
    Recorded goals are measured encoder values, not ideal endpoint commands:
    both 10 -> 12.89 and 12.89 -> 10 can represent the same open endpoint.
    Apply the same bounded open-region contract in preparation and final
    arrival. Both goal AND actual must lie in that region; partial/closing
    targets outside it still require ordinary precision. No force/contact
    completion is inferred. This function never modifies targets, commands,
    or trajectory time.
    """
    state=bridge._action_state
    if state is None:return False
    slot='full_open_arrival_'+channel
    def reject():
        state.pop(slot,None)
        return False
    observed=open_endpoint_observation(bridge,targets)
    if observed is None:return reject()
    stamp=float(getattr(bridge,'_feedback_monotonic',float('nan')))
    if not math.isfinite(stamp):return reject()
    key=(state['loop_index'],state['frame_index'],tuple(sorted(targets.items())))
    stable=state.get(slot)
    if (stable is None or stable['key']!=key or stamp<stable['last'] or stamp-stable['last']>.25
            or any(abs(v-stable['anchor'][n])>.4 for n,v in observed.items())):
        state[slot]=dict(key=key,start=stamp,last=stamp,count=1,anchor=observed)
        return False
    if stamp>stable['last']:
        stable['last']=stamp;stable['count']+=1
    return stable['count']>=4 and stamp-stable['start']>=OPENING_SETTLE_SEC


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
