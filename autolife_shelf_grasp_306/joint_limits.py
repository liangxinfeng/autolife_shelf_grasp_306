"""Grasp targets must be reachable before the vendor's joint-limit guard stops.

No ROS/hardware access. Keep the vendor guard intact and shrink IK's domain.
"""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def load_arm_policy(cfg, names, ui_limits):
    model_path = Path(cfg['urdf_path'])
    vendor_path = Path(cfg['vendor_control_config']) if cfg.get('vendor_control_config') else (
        Path(cfg['vendor_python_path']) / 'autolife_robot_arm/configs/robot_v2_2.json')
    model_bytes = model_path.read_bytes()
    vendor_bytes = vendor_path.read_bytes()
    model = ET.fromstring(model_bytes)
    dead = json.loads(vendor_bytes)['MOTOR_DEAD_ZONE']
    prefix = cfg['arm'].upper() + '_ARM_'
    lower, upper = dead[prefix+'LOWER_DEAD_ZONE'], dead[prefix+'UPPER_DEAD_ZONE']
    # Fixed extra clearance is above feedback quantization/arrival tolerances.
    # Not an operator setting to enlarge the manufacturer's motion range.
    margin = .5
    if len(lower) != len(names) or len(upper) != len(names):
        raise ValueError('厂商关节保护区维度不匹配，不能规划')
    bounds = {}
    for i, name in enumerate(names):
        limit = model.find(f".//joint[@name='{name}']/limit")
        if limit is None:
            raise ValueError('模型缺少关节限位：'+name)
        hard_lo, hard_hi = (math.degrees(float(limit.get(k))) for k in ('lower', 'upper'))
        dl, du = float(lower[i]), float(upper[i])
        ui_lo, ui_hi = ui_limits[name]
        if not all(math.isfinite(v) for v in (hard_lo, hard_hi, dl, du, ui_lo, ui_hi)) or min(dl, du) < 0:
            raise ValueError('无效的厂商关节限位：'+name)
        lo, hi = max(ui_lo, hard_lo+dl+margin), min(ui_hi, hard_hi-du-margin)
        if lo >= hi:
            raise ValueError('厂商保护区与模型限位无交集：'+name)
        bounds[name] = [lo, hi]
    return {'version': 1, 'bounds_deg': bounds, 'clearance_deg': margin,
            'vendor_config_sha256': hashlib.sha256(vendor_bytes).hexdigest(),
            'urdf_sha256': hashlib.sha256(model_bytes).hexdigest()}


def check_pose(policy, q):
    for name, (lo, hi) in policy['bounds_deg'].items():
        value = float(q[name])
        if not math.isfinite(value) or not lo-1e-7 <= value <= hi+1e-7:
            raise ValueError(f'{name} 目标{value:.2f}°进入厂商限位保护区'
                             f'（抓取可用{lo:.2f}～{hi:.2f}°），请重新规划')


def validate_grasp_plan(plan, names, ui_limits):
    policy = load_arm_policy(plan['config'], names, ui_limits)
    if plan.get('joint_limit_policy') != policy:
        raise ValueError('旧规划或厂商限位配置已改变，请重新规划')
    check_pose(policy, plan['source_q'])
    for segment in plan['segments']:
        for q in segment['waypoints']:
            check_pose(policy, q)
    return policy
