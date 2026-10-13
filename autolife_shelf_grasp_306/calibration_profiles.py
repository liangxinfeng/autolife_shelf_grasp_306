"""Immutable, arm-specific camera/TCP calibration snapshots. No robot I/O."""
import copy
import hashlib
import json
from pathlib import Path


class CalibrationProfiles:
    def __init__(self, cfg, fk):
        self.base = copy.deepcopy(cfg)
        self.fk = fk
        self.cache = {}

    def resolve(self, cfg):
        from autolife_handeye_calibration_306.verification_core import load_geometry
        path = str(Path(cfg['calibration']).resolve())
        raw = Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if path in self.cache:
            data, previous = self.cache[path]
            if previous != digest:
                raise ValueError('标定文件已改变，请重开抓取程序')
        else:
            data = json.loads(raw)
            self.cache[path] = (data, digest)
        # Validate robot, URDF and solution even for a needs_validation profile.
        load_geometry(data, self.fk, cfg['tcp_offset_mm'], robot_id=cfg.get('robot_id'))
        return data, digest

    def for_arm(self, side):
        if side not in ('left', 'right'):
            raise ValueError('无效抓取手臂')
        cfg = copy.deepcopy(self.base)
        explicit = cfg.get(side + '_calibration')
        if explicit:
            cfg['calibration'] = explicit
        data, digest = self.resolve(cfg)
        if explicit and data.get('config', {}).get('arm') != side:
            raise ValueError('独立标定文件与抓取手臂不一致：' + side)
        if side != self.base.get('arm', 'left'):
            cfg['target_offset_torso_mm'] = list(cfg.get(side + '_target_offset_torso_mm', [0., 0., 0.]))
        cfg['calibration_profile'] = {
            'arm': side, 'source_arm': data.get('config', {}).get('arm', 'left'),
            'session': Path(cfg['calibration']).parent.name, 'sha256': digest,
            'quality_grade': data.get('report', {}).get('quality_grade', 'unknown'),
            'test_only': data.get('report', {}).get('quality_grade') == 'needs_validation',
        }
        return cfg, data, digest

    def for_held_plan(self, plan):
        from .arm_selection import plan_arm
        cfg, data, digest = self.for_arm(plan_arm(plan))
        saved = plan['config']
        if (Path(saved['calibration']).resolve() != Path(cfg['calibration']).resolve()
                or plan.get('calibration_sha256') != digest
                or plan.get('tcp_offset_mm') != cfg['tcp_offset_mm']):
            raise ValueError('抓取后该手臂标定或夹取中心参数改变，不能沿用持物记录')
        return data, digest
