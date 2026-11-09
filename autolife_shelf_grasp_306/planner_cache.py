"""只复用模型和纯计算进程，绝不复用路径、关节起点或碰撞结果。"""
import json
from pathlib import Path

DYNAMIC={'up_torso','target_offset_torso_mm','target_offsets_by_arm_torso_mm',
    'initial_pose_deg','workspace_reset_joints','workspace_reset_path','workspace_reset_sha256',
    'workspace_pose_speed_percent','grasp_speed_percent','semantic_prompt_overrides'}

def cache_key(cfg):
    static={k:v for k,v in cfg.items() if k not in DYNAMIC}
    vendor=Path(cfg.get('vendor_control_config') or Path(cfg['vendor_python_path'])/'autolife_robot_arm/configs/robot_v3_1.json')
    collision=Path(__file__).resolve().parents[1]/'urdf/robot_v3_1_simplified.urdf'
    files=[Path(cfg['urdf_path']),vendor,collision,collision.with_name('robot_v3_1.srdf')]
    if cfg.get('gripper_width_calibration'):files.append(Path(cfg['gripper_width_calibration']))
    stamp=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in files]
    return json.dumps([static,stamp],sort_keys=True,allow_nan=False)
