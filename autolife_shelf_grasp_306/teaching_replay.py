"""Measured-time replay. No automatic retiming or gripper/body reordering."""
from copy import deepcopy
import math
import time
import numpy as np
from control_center_joint_control.joint_definitions import (
    BODY_JOINTS,GRIPPER_JOINTS,GROUP_ORDER,JOINT_LIMITS)
from .teaching_track import NAMES,load_raw,validate_arrays,evaluate_curve


def edited_samples(record,directory):
    meta=record['trajectory'];t,original=load_raw(directory,meta['sha256'])
    indices=meta['indices'];frames=record['keyframes']
    if len(indices)!=len(frames) or indices[0]!=0 or indices[-1]!=len(t)-1:
        raise ValueError('连续轨迹的编辑节点与原始采样不一致')
    q=original.copy()
    for j,n in enumerate(NAMES):
        delta=np.array([f['joints_deg'][n]-original[i,j] for i,f in zip(indices,frames)])
        for k in range(len(indices)-1):
            a,b=indices[k:k+2];u=(t[a:b+1]-t[a])/(t[b]-t[a]);s=u**3*(10-15*u+6*u*u)
            q[a:b+1,j]=original[a:b+1,j]+delta[k]+(delta[k+1]-delta[k])*s
    validate_arrays(t,q)
    # The runtime carry handoff may retain an already-held object. Keep that
    # opening until the recording explicitly opens beyond it; never tighten it
    # just to match the first sample.
    for n,value in meta.get('entry_gripper_hold',{}).items():
        j=NAMES.index(n);opening=np.flatnonzero(q[:,j]<value)
        end=int(opening[0]) if len(opening) else len(t)
        q[:end,j]=value
    warped=t.copy()
    for k in range(1,len(indices)):
        a,b=indices[k-1:k+1]
        factor=100./meta['speeds'][k]*100./record['speed_percent']
        warped[a:b+1]=warped[a]+(t[a:b+1]-t[a])*factor
    # Only explicit user dwell changes add time. Do not compress A+X waits,
    # impose per-marker minimums, or insert automatic gripper delays.
    holds={i:f['hold_sec'] for i,f in zip(indices,frames) if f['hold_sec']>0}
    times=[];values=[];extra=0.
    for i,(stamp,row) in enumerate(zip(warped,q)):
        times.append(float(stamp+extra));values.append(row.copy())
        if i in holds:
            extra+=holds[i];times.append(float(stamp+extra));values.append(row.copy())
    return np.array(times),np.array(values),float(t[-1])


def plan(record,entry,directory):
    from scipy.interpolate import PchipInterpolator
    from .teaching_motion import checked_joints,duration
    from .teaching_trajectory import build
    from .task_flow import full_record
    t,q,recorded=edited_samples(record,directory)
    # Shape-preserving interpolation keeps each measured sample at its original
    # time and cannot overshoot a joint limit between two valid samples.
    spline=PchipInterpolator(t,q,axis=0)
    curve=dict(times=spline.x,coeff=spline.c)
    rows=[];segments=[];phases=[];current=checked_joints(entry)
    first=dict(zip(NAMES,map(float,q[0])))
    def prepare(target,hold,label):
        nonlocal current
        seconds=duration(current,target,100.,.6)
        one,seg=build(dict(groups=list(GROUP_ORDER),loop_count=1,speed_percent=100,
            keyframes=[dict(joints_deg=target,transition_sec=seconds,hold_sec=hold,easing='smooth')]),current)
        rows.extend(one['keyframes']);segments.extend(seg);phases.append((0,label,0));current=dict(target)
    body=dict(first);body.update({n:current[n] for n in GRIPPER_JOINTS})
    body_change=any(abs(current[n]-body[n])>1e-6 for n in BODY_JOINTS)
    jaw_change=any(abs(current[n]-first[n])>1e-6 for n in GRIPPER_JOINTS)
    if body_change or jaw_change:prepare(body,.25 if jaw_change else 0.,'准备：对齐录制起点（另计）')
    if jaw_change:prepare(first,0.,'准备：起点夹爪（另计）')
    # Split for the existing <=30s action schema; the execution clock does not
    # restart or wait for arrival at these internal blocks.
    cuts=list(np.linspace(0.,float(t[-1]),max(1,int(math.ceil(t[-1]/8.)))+1))
    peak=float(t[int(np.argmax(np.max(np.abs(q-q[0]),axis=1)))])
    if .2<=peak<=t[-1]-.2 and min(abs(peak-c) for c in cuts)>=.2:cuts.append(peak)
    cuts.sort()
    for a,b in zip(cuts,cuts[1:]):
        seg=dict(curve=curve,curve_start=float(a),curve_end=float(b),scale=1.,seconds=float(b-a),
                 recorded_timing=True,timeline_offset=float(a),timeline_duration=float(t[-1]),recorded_sec=recorded)
        seg.update(start=evaluate_curve(seg,0),target=evaluate_curve(seg,1),
                   v0=evaluate_curve(seg,0,1),v1=evaluate_curve(seg,1,1))
        rows.append(dict(joints_deg=checked_joints(seg['target']),transition_sec=seg['seconds'],hold_sec=0.,easing='linear'))
        segments.append(seg);phases.append((0,'按录制时序播放',0))
    result=full_record(dict(groups=list(GROUP_ORDER),loop_count=1,speed_percent=100,keyframes=rows))
    return result,segments,phases


def advance(node):
    """Return True when this tick belongs to the recorded-time controller.

    Retain native ownership, communication and feedback-lead guards. A missed
    deadline or untrackable target stops with a reason, never stretches time.
    """
    state=node._action_state;segments=node.teaching_segments
    if not state or not segments:return False
    segment=segments[state['frame_index']]
    if not segment.get('recorded_timing'):return False
    if not getattr(node,'teaching_sync_verified',False):
        node._lock('原时序回放未确认位置SYNC会话，已停止');return True
    if not node._hardware_enabled or not node._hardware_ready or not node._is_feedback_fresh():
        node._lock('原时序回放的使能或反馈失效，已停止');return True
    if not node._leg_waist_enabled:
        node._cancel_action('腰腿解锁已关闭，原时序回放停止');return True
    if node._communication_lost_joints(state['joint_names']):
        node._lock('原时序回放检测到关节失联，已停止');return True
    wall=time.monotonic();pause=getattr(node,'task_pause',None);now=pause.now() if pause else wall
    node._last_operator_activity=wall
    if 'recorded_epoch' not in state:
        state['recorded_epoch']=now-segment['timeline_offset'];state['recorded_last_tick']=now
    if now-state['recorded_last_tick']>.15:
        node._cancel_action('原时序回放执行循环延迟超过150ms，已停止；未跳过轨迹追赶');return True
    state['recorded_last_tick']=now
    elapsed=max(0.,now-state['recorded_epoch']);total=segment['timeline_duration']
    while state['frame_index']<len(segments)-1 and elapsed>=segment['timeline_offset']+segment['seconds']:
        state['frame_index']+=1;segment=segments[state['frame_index']]
    state['progress']=min(1.,max(0.,(elapsed-segment['timeline_offset'])/segment['seconds']))
    state['recorded_elapsed']=min(total,elapsed);state['recorded_duration']=total
    state['stage_label']=f'原时序播放 {min(total,elapsed):.2f} / {total:.2f} 秒'
    desired=evaluate_curve(segment,state['progress'])
    state['replay_desired']=dict(desired)
    from .teaching_output import command_lead,ARMS
    clipped=state.setdefault('recorded_clipped_since',{})
    for n,target in desired.items():
        measured=float(node._current[n]);limit=command_lead(node,n)
        error=abs(target-measured)
        if not math.isfinite(measured):
            node._lock('原时序回放关节反馈数值无效');return True
        if n in GRIPPER_JOINTS:continue
        if error>limit+1e-6:
            # The streaming publisher uses this same per-joint envelope.
            # Permit brief clipping, but never skip forward through a stall.
            since=clipped.setdefault(n,now)
            abort=min(22.,2.*limit) if n in ARMS else (3. if n in GRIPPER_JOINTS else 2.)*limit
            if error>abort or now-since>=.15:
                node._cancel_action(f'原时序回放跟随不足：{n} 偏差{error:.2f}；已有指令限幅持续或偏差过大，已停止，未自动放慢');return True
        else:clipped.pop(n,None)
    from .teaching_gripper_follow import check as check_grippers
    jaw_failure=check_grippers(node,state,desired,now,tracking_required=False)
    if jaw_failure:
        node._cancel_action(jaw_failure);return True
    node._preview.update(desired)
    try:
        node._publish_action_targets(state,desired,state['record']['keyframes'][state['frame_index']])
    except (ValueError,KeyError,TypeError) as exc:
        node._cancel_action('原时序指令校验失败，已停止：'+str(exc));return True
    if elapsed>=total:
        node._record_action_event('recorded_timeline_complete',f'原时序{total:.6f}秒，实发{elapsed:.6f}秒')
        # Gripper position tracking is advisory for recorded playback, including
        # its final sample. Do not extend the timeline to wait for jaw arrival.
        # Arm/body arrival, feedback validity and communication remain required.
        okay=all(abs(node._current[n]-v)<=float(node.get_parameter(
            'goal_tolerance_deg').value) for n,v in desired.items() if n not in GRIPPER_JOINTS)
        if not okay:
            node._cancel_action('录制时间轴已结束，但身体关节实测末姿未到位；已停止，未延长轨迹');return True
        node._record_action_event('recorded_gripper_arrival_unverified',
            '夹爪按原时序发送；位置偏差不拦停，夹取或释放结果未核验')
        name=state['name'];node._action_state=None
        node.outcome=(name,True,f'原时序轨迹完成：{total:.2f}秒，等待身体停稳；夹爪到位未核验')
        node._event(node.outcome[2])
    return True
