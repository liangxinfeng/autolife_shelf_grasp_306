"""全身复位使用姿态执行共用动作桥；不叠加Follower或分组apply平滑。"""
import uuid
from .core import atomic_json
from .teaching_trajectory import plan as trajectory_plan
from control_center_joint_control.joint_definitions import BODY_JOINTS

def execute_pose(node,context,plan,checked,now,run):
    q=checked();source=plan['pose_source_q']
    if any(abs(q[n]-v)>(8. if 'Gripper' in n else .7) for n,v in source.items()):
        raise ValueError('复位开始姿态与已检查路径不一致，请重新复位')
    # 原厂动作接口会拒绝完全不变的目标；已在目标处只核验停稳，不重复发送。
    start=now();stable=None
    while max(abs(q[n]-v) for n,v in plan['end_q'].items())<.2:
        quiet=all(abs(float(node._details.get(n,{}).get('speed',float('inf'))))<=2. for n in BODY_JOINTS)
        if not quiet:stable=None
        elif stable is None:stable=now()
        elif now()-stable>=.3:return dict(q)
        if now()-start>5.:raise ValueError('复位目标处未确认停稳')
        q=checked()
    if any(abs(q[n]-v)>(8. if 'Gripper' in n else .7) for n,v in source.items()):
        raise ValueError('复位开始前姿态改变，请重新复位')
    record,segments,phases=trajectory_plan(plan['pose_action'],source,None)
    context.teaching_phases=phases
    node.teaching_segments=segments;node.teaching_sync_verified=True;node.outcome=None
    token='workspace-reset-'+uuid.uuid4().hex
    node.start_workspace_reset(token,record)
    if not node._action_state:raise ValueError('复位动作未被接收：'+node._last_event)
    start=now();last=-1.;stable=None
    try:
        while True:
            q=checked();stamp=now()
            if stamp-start>300.:raise ValueError('全身复位动作超时')
            if node.outcome and node.outcome[0]==token:
                if not node.outcome[1]:raise ValueError(node.outcome[2])
                quiet=all(abs(float(node._details.get(n,{}).get('speed',float('inf'))))<=2. for n in BODY_JOINTS)
                if not quiet:stable=None
                elif stable is None:stable=stamp
                elif stamp-stable>=.3:return dict(q)
            elif not node._action_state:
                raise ValueError('复位动作中断，未收到完成确认')
            if stamp-last>=.2:
                state=node._action_state or {}
                atomic_json(run/'motion_progress.json',dict(stage='全身复位（与姿态执行共用）',
                    frame_index=state.get('frame_index',0),progress=state.get('progress',1.),
                    elapsed_sec=stamp-start,speed_percent=plan['config']['workspace_pose_speed_percent']))
                last=stamp
    finally:
        if node._action_state:node._cancel_action('全身复位停止，保持当前位置')
        node.teaching_sync_verified=False
