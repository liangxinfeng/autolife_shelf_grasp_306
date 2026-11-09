"""姿态工作区是抓取复位目标的唯一用户设置入口；不写原厂复位文件。"""
import json,time,hashlib
from pathlib import Path
from .teaching_motion import checked_joints
from .core import atomic_json

def reset_path(root):return Path(root)/'data/fixed_actions/manual_reset_pose.json'

def save(root,cfg,q,pose_name=None):
    q=checked_joints(q);path=reset_path(root);path.parent.mkdir(parents=True,exist_ok=True)
    data=dict(version=1,robot_id=int(cfg['robot_id']),urdf_sha256=hashlib.sha256(Path(cfg['urdf_path']).read_bytes()).hexdigest(),
              joints_deg=q,saved_at=time.time(),reference=dict(source='shared pose execution profile'))
    if pose_name is not None:
        data['reference']['pose_name']=str(pose_name)
    atomic_json(path,data);return data

def sync_named_pose(root,cfg,name,q):
    """覆盖正在使用的命名姿态时，同步复位快照；已生成计划由摘要校验作废。"""
    path=reset_path(root)
    if path.exists() and json.loads(path.read_text('utf-8')).get('reference',{}).get('pose_name')==name:
        save(root,cfg,q,pose_name=name)
        return True
    return False

def resolve(root,cfg):
    path=reset_path(root);result=dict(cfg)
    for key in ('workspace_reset_joints','workspace_reset_path','workspace_reset_sha256','workspace_pose_speed_percent'):
        result.pop(key,None)
    if not path.exists():return result
    raw=path.read_bytes();d=json.loads(raw)
    if d.get('version')!=1 or d.get('robot_id')!=int(cfg['robot_id']):raise ValueError('复位姿态不属于本机，请重新设置')
    if d['urdf_sha256']!=hashlib.sha256(Path(cfg['urdf_path']).read_bytes()).hexdigest():raise ValueError('复位模型已改变，请重新设置复位姿态')
    q=checked_joints(d['joints_deg'])
    from control_center_joint_control.joint_definitions import group_joint_names
    result['initial_pose_deg']={**cfg['initial_pose_deg'],**{g:[q[n] for n in group_joint_names(g)] for g in ('left_arm','right_arm','neck')}}
    from .pose_execution_profile import read_speed
    result['workspace_pose_speed_percent']=read_speed(root)
    result.update(workspace_reset_joints=q,workspace_reset_path=str(path),workspace_reset_sha256=hashlib.sha256(raw).hexdigest(),use_position_sync_session=True)
    return result

def reset_record(cfg,source):
    """直接用一条全身姿态轨迹复位；不添加低速抬头或颈部归正阶段。"""
    from .pose_execution_profile import pose_record
    goal=checked_joints(cfg['workspace_reset_joints']);source=checked_joints(source)
    return pose_record(goal,source,cfg['workspace_pose_speed_percent'],'全身复位')


def validate_plan(plan):
    cfg=plan['config'];path=Path(cfg['workspace_reset_path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=cfg['workspace_reset_sha256']:raise ValueError('复位姿态已经改变，请重新点击复位')
    expected=checked_joints(json.loads(path.read_text())['joints_deg'])
    if expected!=cfg['workspace_reset_joints'] or plan.get('reset_protocol_version')!=4 or not cfg.get('use_position_sync_session'):raise ValueError('全身复位数据不一致')
    if plan['segments'][-1]['waypoints'][-1]!=expected:raise ValueError('全身复位终点不匹配')
    if plan.get('pose_execution_version')!=1 or plan.get('pose_action')!=reset_record(cfg,plan['pose_source_q']):
        raise ValueError('全身复位执行轨迹不一致，请重新点击复位')
    return expected

def build_plan(planner,source):
    from control_center_joint_control.joint_definitions import BODY_JOINTS
    from autolife_handeye_calibration_306.verification_trajectory import JointPath
    from .teaching_trajectory import plan
    cfg=planner.cfg;goal=checked_joints(cfg['workspace_reset_joints']);source=checked_joints(source)
    planner.guard.begin_fixed_neck(source,goal)
    record=reset_record(cfg,source)
    staged,curves,_=plan(record,source,None)
    # 单姿态的五次曲线在关节空间是一条直线；逐段检查同一几何路径，
    # 执行只使用共用动作桥一次平滑，不再附加Follower或重新插值。
    planner.guard.cache.clear();count=0;neck_notes=[]
    from .reset_collision_scope import reset_path_scope
    for curve in curves:
        # 每段独立登记；首段偏角归正未满足兼容条件，必须完整检查。
        with reset_path_scope(planner.guard,curve['start'],curve['target']) as notes:
            path=JointPath(list(BODY_JOINTS),[curve['start'],curve['target']])
            for q in path.checks():
                planner.guard.check({**curve['start'],**q});count+=1
            planner.guard.check(curve['target']);count+=1
            neck_notes.extend(p for p in notes if p not in neck_notes)
    return dict(kind='prepare',source_q=source,pose_source_q=source,end_q=goal,checked_states=count,
                segments=[dict(name='执行工作区全身复位姿态',waypoints=[source]+[f['joints_deg'] for f in record['keyframes']])],
                reset_protocol_version=4,workspace_reset=True,pose_execution_version=1,pose_action=record,
                pose_expected_sec=sum(f['transition_sec']+f['hold_sec'] for f in staged['keyframes']),
                urdf_sha256=planner.fk.sha,self_collision_checked=True,table_height_checked=False,
                config=dict(cfg),neck_model_notes=list(neck_notes),created_monotonic=time.monotonic())

