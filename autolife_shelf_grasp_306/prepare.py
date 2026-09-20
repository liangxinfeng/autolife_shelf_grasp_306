"""Precheck the original VR reset in the worker, never in the motor executor."""
import math
import time
from autolife_handeye_calibration_306.core import digest,ARMS,NECK
from control_center_joint_control.joint_definitions import GROUPS


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


def checked_reset_plan(planner,cfg,source):
    request=reset_request(cfg,source)
    result=planner.reset_plan(source)
    if not result['self_collision_checked']:raise ValueError('VR准备姿态路径未通过检查')
    result.update(prepare_mode=request['prepare_mode'],reset_protocol_version=request['reset_protocol_version'])
    return result
