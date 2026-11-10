"""姿态串沿一条共同曲线推进：反馈限制减慢整段，不把各轴单独裁切。"""
import math,time
from control_center_joint_control.joint_definitions import GROUPS,GRIPPER_JOINTS,JOINT_LIMITS,group_joint_names
from .teaching_trajectory import evaluate


def body_names(state):
    return [n for g in ('neck','left_arm','right_arm','leg_waist') if g in state['record']['groups'] for n in group_joint_names(g)]


def before_tick(node):
    state=node._action_state;segments=node.teaching_segments
    if not state or not segments or state['stage']!='transition':return
    seg=segments[state['frame_index']]
    if seg.get('recorded_timing') or 'curve' in seg:return
    now=time.monotonic();key=(state['loop_index'],state['frame_index'])
    clock=state.get('synchronized_clock')
    if clock is None or clock['key']!=key:
        clock=dict(key=key,progress=0.,last=now);state['synchronized_clock']=clock
    # GUI暂停/调度延迟不能形成一下子追赶到远处的目标。
    dt=max(0.,min(.04,now-clock['last']));clock['last']=now
    duration=seg['seconds']/(state['record']['speed_percent']/100.)
    request=min(1.,clock['progress']+dt/duration)
    state['frame_started']=now-request*duration
    if now>state['frame_deadline']:
        node._cancel_action('同步姿态超过原有到位期限，已停止；未跳过关节跟随检查')


def progress(node,state,segment,requested):
    clock=state.setdefault('synchronized_clock',dict(progress=0.))
    previous=float(clock['progress']);requested=max(previous,min(1.,float(requested)))
    lead=max(.1,float(node.get_parameter('max_single_step_deg').value))*state['record']['speed_percent']/100.
    names=body_names(state)
    if not all(math.isfinite(float(node._current[n])) for n in names):raise ValueError('同步姿态反馈无效，未下发目标')
    def within(u):
        q=evaluate(segment,u)
        return all(math.isfinite(float(node._current[n])) and abs(q[n]-node._current[n])<=lead+1e-9 for n in names)
    if within(requested):accepted=requested
    elif not within(previous):accepted=previous
    else:
        lo,hi=previous,requested
        for _ in range(24):
            mid=(lo+hi)/2
            if within(mid):lo=mid
            else:hi=mid
        accepted=lo
    clock['progress']=accepted;state['progress']=accepted
    return evaluate(segment,accepted)


def publish_body(node,state,desired,frame):
    """仅供普通姿态共用进度输出；夹爪由既有有界目标/接触等待处理。"""
    names=body_names(state);command={n:float(desired[n]) for n in names}
    for n,v in command.items():
        lo,hi=JOINT_LIMITS[n]
        if not math.isfinite(v) or not lo<=v<=hi:raise ValueError('同步姿态目标无效：'+n)
    payload={}
    for group in ('neck','left_arm','right_arm','leg_waist'):
        values=[command[n] if n in command else float(node._controller_target[n]) for n in group_joint_names(group)]
        if not all(map(math.isfinite,values)):raise ValueError('保持目标无效：'+group)
        payload[GROUPS[group]['command_key']]=values
    if names:node._publish_vendor(node._whole_body_pub,payload)
    state['commanded_values'].update(command)
