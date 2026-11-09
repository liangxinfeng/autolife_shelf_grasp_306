"""345 DM Type1 measured width mapping; never reuse the S2 95 mm model."""
import math
import hashlib
import json
from pathlib import Path
import numpy as np


def load_curve(cfg):
    path=Path(cfg.get('gripper_width_calibration','/home/ubuntu/ros2_ws/calibration345/gripper_width.json'))
    if not path.is_file():
        raise ValueError('345 DM夹爪尚未实测毫米开度；请完成左右夹爪宽度标定，当前可识别和手动位置控制')
    raw=path.read_bytes();data=json.loads(raw)
    if data.get('robot_id')!=345 or data.get('hardware')!='dm_type1' or data.get('measured') is not True:
        raise ValueError('夹爪标定必须为345本机DM Type1实测数据')
    points=np.array(data.get('arms',{}).get(cfg['arm'],[]),dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.all(np.isfinite(points)):
        raise ValueError('每侧夹爪至少需要3组[开口毫米,原厂位置量]实测数据')
    widths,positions=points.T
    if (np.any(widths<0) or np.any(positions<10) or np.any(positions>360)
            or not np.all(np.diff(widths)>0) or not np.all(np.diff(positions)<0)):
        raise ValueError('夹爪数据应按开口递增、位置量递减排序，位置量范围10～360')
    return widths,positions,hashlib.sha256(raw).hexdigest()


def calibrated_max_width(cfg):
    widths,_,_=load_curve(cfg)
    return float(widths[-1])/1000.

def checked_margin(value):
    if isinstance(value,bool):raise ValueError('抓取收紧量应为0～50毫米')
    value=float(value)
    if not math.isfinite(value) or not 0<=value<=50:raise ValueError('抓取收紧量应为0～50毫米')
    return value

def closing_command(width_m,margin_mm,cfg):
    widths,positions,sha=load_curve(cfg)
    width=float(width_m);margin=checked_margin(margin_mm)
    if not math.isfinite(width) or not widths[0]<width*1000<=widths[-1]:
        raise ValueError(f'夹取部位应大于{widths[0]:g}毫米且不超过{widths[-1]:g}毫米；不能用最小开口夹紧更薄物品')
    requested=max(float(widths[0]),width*1000-margin)
    maximum=float(cfg.get('gripper_closed_deg',330.));minimum=float(cfg.get('gripper_open_deg',10.))
    if not 10<=minimum<=maximum<=360:raise ValueError('DM夹爪位置端点无效')
    angle=max(minimum,min(maximum,float(np.interp(requested,widths,positions))))
    actual=float(np.interp(angle,positions[::-1],widths[::-1]))
    return dict(estimated_width_mm=width*1000,tightening_mm=margin,
                requested_width_mm=requested,nominal_target_width_mm=actual,
                target_deg=angle,mapping='robot345_dm_type1_measured_v1',calibration_sha256=sha)

def command_from_plan(plan):
    command=plan.get('grip_command')
    if command is None:raise ValueError('旧抓取计划缺少345 DM夹爪标定信息，请重新规划')
    expected=closing_command(plan['grasp_width_m'],command['tightening_mm'],plan['config'])
    for key in ('mapping','calibration_sha256'):
        if command.get(key)!=expected[key]:raise ValueError('夹爪标定已变化，请重新规划')
    for key in ('estimated_width_mm','requested_width_mm','nominal_target_width_mm','target_deg'):
        if not math.isclose(float(command[key]),expected[key],abs_tol=1e-6):raise ValueError('夹爪宽度计划不一致，请重新规划')
    return expected['target_deg']
