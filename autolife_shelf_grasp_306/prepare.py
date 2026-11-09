"""Precheck the original VR reset in the worker, never in the motor executor."""
import math
import time
from autolife_handeye_calibration_306.core import digest,ARMS,NECK
from control_center_joint_control.joint_definitions import GROUPS


def ready_after_place(cfg,q):
    """Only measured full preparation pose can mint another reset receipt."""
    try:
        if cfg.get('workspace_reset_joints'):
            return all(abs(q[n]-v)<=(8. if 'Gripper' in n else 1.) for n,v in cfg['workspace_reset_joints'].items())
        for group in ('left_arm','right_arm','neck'):
            joints=GROUPS[group]['joints'];values=cfg['initial_pose_deg'][group]
            if len(joints)!=len(values):return False
            for (name,*_),value in zip(joints,values):
                if not math.isfinite(q[name]) or not math.isfinite(value) or abs(q[name]-value)>1.:return False
        from .core import open_tolerance
        return all(math.isfinite(q[n]) and abs(q[n]-cfg['gripper_open_deg'])<=open_tolerance(cfg)
                   for n in ('Joint_Left_Gripper','Joint_Right_Gripper'))
    except (KeyError,TypeError,ValueError):return False


def reset_request(cfg,source):
    for group in ('left_arm','right_arm','neck'):
        targets=cfg['initial_pose_deg'][group];joints=GROUPS[group]['joints']
        if len(targets)!=len(joints):raise ValueError('复位关节数量不匹配：'+group)
        for (name,_,lo,hi),v in zip(joints,targets):
            if not math.isfinite(v) or not lo<=v<=hi:raise ValueError('复位目标越界：'+name)
    if not source or not all(math.isfinite(v) for v in source.values()):raise ValueError('当前关节反馈无效')
    return {'kind':'prepare','prepare_mode':'hold_then_vr','reset_protocol_version':2,
            'config':dict(cfg),'source_q':dict(source),'created_monotonic':time.monotonic(),
            'urdf_sha256':digest(cfg['urdf_path']),'self_collision_checked':False,'segments':[]}


def checked_reset_plan(planner,cfg,source,*,operator_supported=False):
    if cfg.get('workspace_reset_joints') and (
            getattr(planner,'cfg',{}).get('workspace_reset_joints')!=cfg['workspace_reset_joints'] or
            getattr(planner,'cfg',{}).get('workspace_reset_sha256')!=cfg.get('workspace_reset_sha256') or
            getattr(planner,'cfg',{}).get('workspace_pose_speed_percent')!=cfg.get('workspace_pose_speed_percent')):
        raise ValueError('复位规划器不是当前保存姿态，请重新生成复位计划')
    request=reset_request(cfg,source)
    try:
        result=planner.reset_plan(source)
    except ValueError as exc:
        if cfg.get('workspace_reset_joints'):
            raise ValueError('保存姿态的直接复位路径未通过检查，未开始运动：'+str(exc)) from exc
        # 区分当前姿态就有模型碰撞与复位中途发生碰撞；诊断不改变放行条件。
        # 不裁剪实测角度、不删除碰撞对，也不为消除报错自动移动颈部。
        if '机器人自碰撞：' not in str(exc):
            raise
        current_error=None
        guard=getattr(planner,'guard',None)
        if guard is not None:
            try:
                guard.check(dict(source))
            except ValueError as current:
                current_error=str(current)
        if current_error and '机器人自碰撞：' in current_error:
            if cfg.get('supported_neck_recovery_enabled') is True:
                from .supported_neck_recovery import build_plan
                result=build_plan(planner,cfg,source,operator_supported=operator_supported)
                result.update(prepare_mode=request['prepare_mode'],reset_protocol_version=result.get('reset_protocol_version',3))
                return result
            angle=source.get('Joint_Neck_Pitch')
            neck=('颈部俯仰实测 %.1f°。'%angle if angle is not None and
                  ('Link_Neck_' in current_error or 'Head' in current_error) else '')
            raise ValueError('当前姿态未通过自碰撞检查，尚未开始复位或抓取。'+neck+
                             '请先核对实际部件间隙与模型；反复点击抓取不会解除此状态。\n'+current_error) from exc
        raise ValueError('复位路径未通过自碰撞检查，尚未开始运动。\n'+str(exc)) from exc
    if not result['self_collision_checked']:raise ValueError('VR准备姿态路径未通过检查')
    result.update(prepare_mode=request['prepare_mode'],reset_protocol_version=result.get('reset_protocol_version',request['reset_protocol_version']))
    return result
