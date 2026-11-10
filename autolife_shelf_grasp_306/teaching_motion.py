"""Endpoint-preserving, rest-to-rest teaching curves; no motor interfaces."""
import math
from copy import deepcopy
from control_center_joint_control.joint_definitions import ALL_JOINTS,JOINT_LIMITS,JOINT_TO_GROUP,GROUP_ORDER,GRIPPER_JOINTS,BODY_JOINTS
from .task_flow import full_record


def finite(value,lo,hi,label):
    value=float(value)
    if not math.isfinite(value) or not lo<=value<=hi:raise ValueError(f'{label}应在{lo:g}～{hi:g}之间')
    return value


def checked_joints(values):
    if not isinstance(values,dict) or set(values)!=set(ALL_JOINTS):raise ValueError('姿态缺少完整的23个关节，不能自动补零')
    return {n:finite(values[n],*JOINT_LIMITS[n],n) for n in ALL_JOINTS}


def normalize_draft(draft):
    rows=draft.get('frames',[])
    if not isinstance(rows,list) or len(rows)>100:raise ValueError('动作串最多100个姿态；较长任务请拆成多个动作串')
    frames=[]
    for i,f in enumerate(rows):
        label=str(f.get('label',f'姿态 {i+1}')).strip()[:48] or f'姿态 {i+1}'
        frames.append(dict(label=label,joints_deg=checked_joints(f['joints_deg']),
            hold_sec=finite(f.get('hold_sec',.3),0.,30.,'停顿时间'),
            speed_percent=finite(f.get('speed_percent',100),25.,150.,'姿态速度'),
            minimum_sec=finite(f.get('minimum_sec',.6),.2,30.,'最短到达时间'),
            ghost=bool(f.get('ghost',True))))
        if 'sample_index' in f:frames[-1]['sample_index']=int(f['sample_index'])
        if f.get('visual_binding'):
            from .visual_reference import binding
            frames[-1]['visual_binding']=binding(f['visual_binding'])
            frames[-1]['visual_enabled']=f.get('visual_enabled',True) is not False
    result=dict(version=1,name=str(draft.get('name','')).strip()[:32],frames=frames)
    if draft.get('kind')=='trajectory':
        from .teaching_track import check_hash
        indices=[f.get('sample_index',-1) for f in frames]
        if not indices or indices!=sorted(set(indices)) or indices[0]!=0 or indices[-1]>15000:
            raise ValueError('连续轨迹节点必须按录制顺序排列并保留首尾；请用微调修改姿态')
        result.update(kind='trajectory',trajectory_ref=check_hash(draft['trajectory_ref']),
                      recorded_sec=finite(draft['recorded_sec'],.2,300.1,'录制时长'),notes=str(draft.get('notes',''))[:300])
        timing=draft.get('timing','recorded')
        if timing not in ('recorded','interaction'):raise ValueError('连续轨迹时序模式无效')
        result['timing']=timing
    return result


def limits(name):
    # Conservative reference limits, never a claim about motor capabilities.
    group=JOINT_TO_GROUP[name]
    if group=='leg_waist':return 12.,24.,90.
    if group=='neck':return 20.,40.,150.
    if group=='grippers':return 120.,240.,900.
    return 40.,90.,360.


def duration(start,target,speed_percent=100.,minimum=.6):
    start=checked_joints(start);target=checked_joints(target)
    scale=finite(speed_percent,25.,150.,'姿态速度')/100.
    required=float(minimum)
    for n in ALL_JOINTS:
        delta=abs(target[n]-start[n]);v,a,j=limits(n)
        # Quintic s=10u^3-15u^4+6u^5: max derivatives 1.875, 10/sqrt(3),60.
        required=max(required,1.875*delta/(v*scale),
                     math.sqrt((10/math.sqrt(3))*delta/(a*scale)),
                     (60.*delta/(j*scale))**(1/3))
    if required>30.:raise ValueError('相邻姿态跨度过大或速度过低，单段超过30秒；请增加中间姿态')
    return max(.2,required)


def curve(start,target,progress):
    u=max(0.,min(1.,float(progress)));s=u*u*u*(10.+u*(-15.+6.*u))
    return {n:float(start[n])+(float(target[n])-float(start[n]))*s for n in target}


def compile_action(draft,entry=None):
    draft=normalize_draft(draft)
    if not draft['frames']:raise ValueError('请先记录至少一个姿态')
    previous=checked_joints(entry) if entry is not None else draft['frames'][0]['joints_deg']
    keyframes=[]
    for f in draft['frames']:
        body=dict(f['joints_deg']);body.update({n:previous[n] for n in GRIPPER_JOINTS})
        seconds=duration(previous,body,f['speed_percent'],f['minimum_sec'])
        keyframes.append(dict(joints_deg=dict(f['joints_deg']),transition_sec=seconds,
                             hold_sec=f['hold_sec'],easing='smooth'))
        previous=f['joints_deg']
    record=dict(groups=list(GROUP_ORDER),loop_count=1,speed_percent=100,keyframes=keyframes)
    if draft.get('kind')=='trajectory':
        record['trajectory']=dict(sha256=draft['trajectory_ref'],indices=[f['sample_index'] for f in draft['frames']],
                                  speeds=[f['speed_percent'] for f in draft['frames']],minimums=[f['minimum_sec'] for f in draft['frames']],timing=draft['timing'])
    bindings={str(i):f['visual_binding'] for i,f in enumerate(draft['frames']) if f.get('visual_binding') and f.get('visual_enabled',True)}
    if bindings:record['visual_bindings']=bindings
    return full_record(record)


def stage_action(record,entry):
    """Body arrival -> verified settle -> jaws -> dwell, for preview and execution.

    The persisted record remains logical poses. Expand only with the actual entry
    state, so the very first move also holds the current gripper opening.
    """
    record=full_record(record);previous=checked_joints(entry);rows=[];phases=[]
    for repeat in range(record['loop_count']):
        for i,f in enumerate(record['keyframes']):
            target=dict(f['joints_deg']);body=dict(target)
            body.update({n:previous[n] for n in GRIPPER_JOINTS})
            jaw_change=any(abs(target[n]-previous[n])>.001 for n in GRIPPER_JOINTS)
            rows.append(dict(joints_deg=body,transition_sec=duration(previous,body,100,
                f['transition_sec']/(record['speed_percent']/100.)),hold_sec=.25 if jaw_change else f['hold_sec'],easing='smooth'))
            phases.append((i,'关节到位并停稳',repeat))
            if jaw_change:
                rows.append(dict(joints_deg=target,transition_sec=duration(body,target,100,.3),
                                 hold_sec=f['hold_sec'],easing='smooth'))
                phases.append((i,'夹爪开合 → 停顿',repeat))
            previous=target
    if len(rows)>200:raise ValueError('展开夹爪步骤后超过200段，请拆分动作串或减少循环次数')
    return full_record(dict(groups=list(GROUP_ORDER),loop_count=1,speed_percent=100,keyframes=rows)),phases


def jaws_after_body(current_frame,next_frame):
    if next_frame is None:return False
    a=current_frame['joints_deg'];b=next_frame['joints_deg']
    return all(abs(a[n]-b[n])<1e-8 for n in BODY_JOINTS) and any(abs(a[n]-b[n])>.001 for n in GRIPPER_JOINTS)


def sample_action(record,entry,elapsed):
    record=full_record(record);previous=checked_joints(entry);elapsed=max(0.,float(elapsed))
    for i,f in enumerate(record['keyframes']):
        seconds=f['transition_sec']/(record['speed_percent']/100.)
        if elapsed<seconds:return curve(previous,f['joints_deg'],elapsed/seconds),i,False
        elapsed-=seconds
        if elapsed<f['hold_sec']:return dict(f['joints_deg']),i,False
        elapsed-=f['hold_sec'];previous=f['joints_deg']
    return dict(previous),len(record['keyframes'])-1,True


def total_time(record):
    return sum(f['transition_sec']/(record['speed_percent']/100.)+f['hold_sec'] for f in record['keyframes'])
