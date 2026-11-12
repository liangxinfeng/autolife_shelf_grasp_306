"""抓取统一几何入口：默认URDF名义坐标，不读取任何手眼会话文件。

保留标定模式供将来显式启用；不伪造标定结果或内参样本。
相机图像的内参仍由实时RGB-D驱动提供。名义安装关系不代表实测精度。
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def uses_urdf(cfg):
    source=cfg.get('geometry_source','urdf')
    if source not in ('urdf','calibrated'):raise ValueError('未知几何来源：'+str(source))
    return source=='urdf'


def is_nominal(data):return data.get('geometry_source')=='urdf'


def nominal_profile(cfg,fk):
    from .pose_geometry import ModelGeometry
    side=cfg.get('arm','left')
    if side not in ('left','right'):raise ValueError('无效抓取手臂')
    path=Path(cfg['urdf_path'])
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha!=fk.sha:raise ValueError('本机URDF已改变，请重新打开抓取程序')
    tool=ModelGeometry(path).tool_transform(side)
    # CAD安装面+X为镜头朝前轴；图像光学系X向右、Y向下、Z朝前。
    # 因此光学[X,Y,Z]分别对应CAD[-Y,-Z,+X]，不是绕Z旋转90°。
    optical=np.array([[0.,0.,1.,0.],[-1.,0.,0.,0.],[0.,-1.,0.,0.],[0.,0.,0.,1.]])
    return dict(geometry_source='urdf',geometry_version=2,tool_axes_version=2,tool_axes_source='urdf_mount_transverse_axis',urdf_sha256=sha,
                config={'robot_id':cfg['robot_id'],'arm':side,'head_offsets_deg':[0.,0.,0.]},
                T_mount_camera=optical.tolist(),T_wrist_tool=tool.tolist())


def same_nominal_camera(a,b):
    if not (is_nominal(a) and is_nominal(b)):return False
    keys=('geometry_version','urdf_sha256','T_mount_camera')
    fields=('robot_id','head_offsets_deg')
    return (all(k in a and k in b and a[k]==b[k] for k in keys)
            and all(k in a.get('config',{}) and k in b.get('config',{})
                    and a['config'][k]==b['config'][k] for k in fields))


def profile_digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def check_nominal(data,fk,robot_id=None):
    if data['urdf_sha256']!=fk.sha or (robot_id is not None and int(data['config']['robot_id'])!=int(robot_id)):
        raise ValueError('抓取几何与本机模型不一致，请重新规划')


def load_geometry(data,fk,offset_mm=15.,robot_id=None):
    if not is_nominal(data):
        from autolife_handeye_calibration_306.verification_core import load_geometry as calibrated
        return calibrated(data,fk,offset_mm,robot_id)
    check_nominal(data,fk,robot_id)
    # tcp_offset_mm是旧标定板到夹取中心的偏移，URDF模式不叠加它。
    return np.asarray(data['T_wrist_tool'],float)[:3,3].copy()


def camera_transform(data,fk,q,frame,robot_id=None):
    if not is_nominal(data):
        from autolife_handeye_calibration_306.verification_core import camera_transform as calibrated
        return calibrated(data,fk,q,frame,robot_id)
    from autolife_handeye_calibration_306.core import TORSO,MOUNT
    check_nominal(data,fk,robot_id)
    # 用图像配对时刻的真实颈部/身体姿态做FK，不用历史标定的头部零偏。
    return fk.fk(TORSO,MOUNT,q)@np.asarray(data['T_mount_camera'],float)


def validate_plan_geometry(plan,cfg):
    from .calibration_profiles import CalibrationProfiles
    from autolife_handeye_calibration_306.core import Kinematics
    _,digest=CalibrationProfiles(cfg,Kinematics(cfg['urdf_path'])).resolve(cfg)
    if digest!=plan.get('calibration_sha256') or (not uses_urdf(cfg) and plan.get('tcp_offset_mm')!=cfg.get('tcp_offset_mm')):
        raise ValueError('相机/TCP几何来源已改变，请重新规划')
