"""Explicit URDF geometry profiles for machines without measured hand-eye data.

Stored under data/system_geometry, never as a measured calibration session.
No measured accuracy, marker transform, or camera intrinsics are fabricated.
"""
import copy
import json
from pathlib import Path
import numpy as np
from autolife_handeye_calibration_306.core import atomic_json, MOUNT, TORSO


def system_profile(cfg, fk, side):
    zero = {joint[0]: 0. for joint in fk.joints.values()}
    wrist = 'Link_' + side.title() + '_Wrist_Lower_to_Gripper'
    # The system URDF has no explicit TCP link. Its two finger-frame origins
    # define a nominal central reference; this is not a calibrated contact tip.
    fingers = ['Link_' + side.title() + '_Gripper_' + jaw + '_Finger' for jaw in ('Left', 'Right')]
    tcp = np.mean([fk.fk(wrist, frame, zero)[:3, 3] for frame in fingers], axis=0)
    fk.fk(TORSO, MOUNT, zero)  # Reject a different/incomplete model explicitly.
    if tcp.shape != (3,) or not np.isfinite(tcp).all():
        raise ValueError('系统夹爪中心几何无效')
    return {'geometry_source': 'system_urdf', 'schema_version': 1,
        '_verification_camera_source': 'urdf', 'urdf_sha256': fk.sha,
        'config': {'robot_id': cfg['robot_id'], 'arm': side,
                   'head_offsets_deg': list(cfg.get('head_offsets_deg', [0., 0., 0.]))},
        'report': {'quality_grade': 'system_nominal', 'calibrated': False},
        'system_geometry': {'tcp_m': tcp.tolist(), 'tcp_source': 'urdf_finger_frame_midpoint',
            'tcp_reference_frames': fingers, 'tcp_offset_mm_applied': False,
            'camera_mount_frame': MOUNT, 'camera_optical_convention': 'existing_sdk_mount_rz90',
            'accuracy': 'unvalidated'}}


def resolve_missing_calibration(cfg, fk, root):
    result = copy.deepcopy(cfg)
    if not cfg.get('allow_system_geometry_fallback', True):
        return result
    primary = cfg.get('arm', 'left')
    def available(value):
        return bool(value) and Path(value).is_file()
    missing_primary = not available(cfg.get('calibration'))
    for side in ('left', 'right'):
        key = side + '_calibration'
        path = cfg.get(key)
        # An absent optional other-arm profile keeps the measured primary-arm
        # transfer contract; explicit missing paths get their own system profile.
        needed = (not available(path) if path else missing_primary)
        if not needed:
            continue
        profile = system_profile(cfg, fk, side)
        destination = Path(root)/'data/system_geometry'/('robot%s_%s_%s.json' % (cfg['robot_id'],side,fk.sha[:16]))
        text = json.dumps(profile, ensure_ascii=False, sort_keys=True, allow_nan=False)
        if not destination.exists() or json.loads(destination.read_text(encoding='utf-8')) != profile:
            atomic_json(destination, profile)
        result[key] = str(destination)
        if side == primary and missing_primary:
            result['calibration'] = str(destination)
    return result
