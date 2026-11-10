"""夹爪目标是夹持意图，不是物体受压后的必达开口；不宣称检测到抓牢。"""
import math,time
from control_center_joint_control.joint_definitions import GRIPPER_JOINTS,JOINT_LIMITS


def stable_closing(node,state,targets,commands,starts):
    if not node._is_feedback_fresh():return False
    observed={};has_closing=False
    for n,target in targets.items():
        actual=float(node._current[n]);detail=node._details.get(n,{})
        if detail.get('communication_lost',True) or not all(math.isfinite(float(v)) for v in (target,actual,commands.get(n,float('nan')))):return False
        if n not in GRIPPER_JOINTS:
            if abs(actual-target)>float(node.get_parameter('goal_tolerance_deg').value):return False
            continue
        if abs(commands[n]-target)>.05:return False
        if target>starts[n]+.1 or target>actual+2.:
            has_closing=True
        elif abs(actual-target)>5.:return False  # 张开仍须确认，避免未释放就收手。
        observed[n]=actual
    if not has_closing:return False
    stamp=float(node._feedback_monotonic);old=state.get('closing_settle')
    key=tuple(sorted(targets.items()))
    if old is None or old['key']!=key or stamp<old['last'] or stamp-old['last']>.2 or any(abs(v-old['q'][n])>1. for n,v in observed.items()):
        state['closing_settle']=dict(key=key,q=observed,start=stamp,last=stamp,count=1);return False
    if stamp>old['last']:old['last']=stamp;old['count']+=1
    return old['count']>=4 and stamp-old['start']>=.4


def publish_pose(node,base_publish,state,desired,frame):
    """身体输出委托所选轨迹发布器；夹爪输出有界目标，不被接触残差钳住。"""
    jaws={n:float(desired[n]) for n in GRIPPER_JOINTS} if 'grippers' in state['record']['groups'] else {}
    for n,v in jaws.items():
        lo,hi=JOINT_LIMITS[n]
        if not math.isfinite(v) or not lo<=v<=hi:raise ValueError('夹爪目标超限：'+n)
    proxy=dict(state,record=dict(state['record'],groups=[g for g in state['record']['groups'] if g!='grippers']))
    base_publish(proxy,desired,frame)
    if jaws:
        node._publish_vendor(node._gripper_pub,dict(left_gripper_target_joints_position=[jaws['Joint_Left_Gripper']],right_gripper_target_joints_position=[jaws['Joint_Right_Gripper']]))
        state['commanded_values'].update(jaws)


def manual_tick(engine):
    """空载/持物均结束等待，保持原目标；只接管本次选中的夹爪，不发送身体目标。"""
    j=engine.job;b=engine.bridge
    if not b._hardware_ready or not b._is_feedback_fresh():raise ValueError('夹爪反馈或保持会话失效')
    if b._communication_lost_joints(list(j['targets'])):raise ValueError('夹爪通信中断')
    if 'jaw_started' not in j:
        from .teaching_motion import duration
        target=dict(b._current);target.update(j['targets'])
        j.update(jaw_started=time.monotonic(),jaw_start=dict(b._current),jaw_seconds=duration(b._current,target,j['speed']*100.,.4))
    from .teaching_motion import curve
    u=min(1.,(time.monotonic()-j['jaw_started'])/j['jaw_seconds'])
    wanted=curve(j['jaw_start'],{**j['jaw_start'],**j['targets']},u)
    # 未选的另一只爪保留既有夹持目标，不能用实际开口覆盖它。
    held=getattr(engine,'gripper_targets',{})
    commands={n:wanted[n] if n in j['targets'] else held.get(n,b._current[n]) for n in GRIPPER_JOINTS}
    b._publish_vendor(b._gripper_pub,dict(left_gripper_target_joints_position=[commands['Joint_Left_Gripper']],right_gripper_target_joints_position=[commands['Joint_Right_Gripper']]))
    b._last_operator_activity=time.monotonic()
    if u<1:return
    closing=any(j['targets'][n]>j['jaw_start'][n]+.1 or j['targets'][n]>b._current[n]+2 for n in j['targets'])
    if closing:done=stable_closing(b,j,j['targets'],commands,j['jaw_start'])
    else:done=all(abs(b._current[n]-v)<=5. for n,v in j['targets'].items())
    if not done:return
    engine.gripper_targets={**held,**j['targets']};engine.job=None;engine.panel.setEnabled(True)
    engine.completed.emit(j['token'],True,'夹爪目标已发送并保持；已结束等待。实际开口可能由物体限制，未据此判定抓牢。' if closing else '夹爪已张开到位')
