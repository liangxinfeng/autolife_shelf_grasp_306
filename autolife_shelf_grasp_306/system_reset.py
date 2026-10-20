"""Read this robot's selected system HOME without writing vendor configuration.

Resolve once when the worker starts, so reset and loaded-arm return share one
immutable pose snapshot for the entire planning session.
"""
import copy
import json
import math
import re
import socket
from pathlib import Path
from control_center_joint_control.joint_definitions import GROUPS
from control_center_joint_control.vr_reset_pose import DEFAULT_CONFIG_PATHS, DEFAULT_STATE_PATH


def apply_system_reset(cfg, state_path=None, config_paths=None):
    if cfg.get('reset_pose_source', 'package') == 'package':
        return cfg
    if cfg['reset_pose_source'] != 'robot_system':
        raise ValueError('未知的抓取复位位姿来源')
    expected = str(cfg['robot_id'])
    match = re.search(r'(\d+)$', socket.gethostname())
    if not match or match.group(1) != expected:
        raise ValueError('抓取配置与本机机器人编号不匹配')
    path = Path(state_path or DEFAULT_STATE_PATH)
    state = json.loads(path.read_text(encoding='utf-8'))
    if state.get('version') != 1 or str(state.get('robot_id')) != expected:
        raise ValueError('系统复位配置版本或机器人编号不匹配')
    mode = state.get('mode')
    keys = {'left_arm': ('LEFT_ARM_MOTOR_HOME_ANGLE', 'standing', 7),
            'right_arm': ('RIGHT_ARM_MOTOR_HOME_ANGLE', 'standing', 7),
            'neck': ('NECK_MOTOR_HOME_ANGLE', 'zero', 3)}
    def from_home(data):
        pose = {}
        for group, (key, posture, count) in keys.items():
            values = data[key][posture]
            if len(values) != count + 4:
                raise ValueError('系统HOME关节数量不匹配：' + group)
            pose[group] = [float(v) for v in values[4:]]
        return pose
    if mode == 'custom':
        joints = state['custom']['joints_deg']
        pose = {g: [float(joints[j[0]]) for j in GROUPS[g]['joints']] for g in keys}
    elif mode == 'original':
        originals = state['original']['configs']
        if not originals:
            raise ValueError('系统复位原始位姿缺失')
        pose = from_home(next(iter(originals.values())))
    else:
        raise ValueError('系统复位模式无效，禁止回退到旧抓取位姿')
    for group, values in pose.items():
        for (name, _, low, high), value in zip(GROUPS[group]['joints'], values):
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError('系统复位目标越界：' + name)
    paths = [Path(p) for p in (DEFAULT_CONFIG_PATHS if config_paths is None else config_paths) if Path(p).is_file()]
    if not paths:
        raise ValueError('未找到本机系统HOME配置')
    for vendor_path in paths:
        vendor = from_home(json.loads(vendor_path.read_text(encoding='utf-8')))
        for group, values in pose.items():
            if any(not math.isfinite(b) or abs(a-b) > 1e-5 for a, b in zip(values, vendor[group])):
                raise ValueError('系统保存位姿与厂商HOME不一致：' + str(vendor_path))
    result = copy.deepcopy(cfg)
    result['initial_pose_deg'] = pose
    result['carry_pose_deg'] = list(pose[cfg.get('arm', 'left') + '_arm'])
    result['system_reset_snapshot'] = {'robot_id': expected, 'mode': mode,
        'state_path': str(path), 'saved_at': (state.get('custom') or {}).get('saved_at', '') if mode == 'custom' else state.get('restored_at', ''),
        'groups': list(keys), 'body_policy': 'keep_measured', 'vendor_paths': [str(p) for p in paths]}
    return result
