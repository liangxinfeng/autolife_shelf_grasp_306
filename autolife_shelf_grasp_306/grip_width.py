"""Nominal 95 mm linkage mapping shared with VR; not a force controller."""
import math

def checked_margin(value):
    if isinstance(value,bool):raise ValueError('抓取收紧量应为0～50毫米')
    value=float(value)
    if not math.isfinite(value) or not 0<=value<=50:raise ValueError('抓取收紧量应为0～50毫米')
    return value

def closing_command(width_m,margin_mm,cfg):
    width=float(width_m);margin=checked_margin(margin_mm)
    if not math.isfinite(width) or not 0<width<=float(cfg.get('gripper_max_width_m',.095)):
        raise ValueError('夹取部位宽度无效，请重新规划')
    requested=max(0.,width*1000-margin)
    maximum=min(360.,float(cfg.get('gripper_closed_deg',330.)))
    minimum=max(10.,float(cfg.get('gripper_open_deg',10.)))
    if not math.isfinite(maximum+minimum) or maximum<minimum:raise ValueError('夹爪端点无效')
    angle=max(minimum,min(maximum,360.-requested/95.*350.))
    return dict(estimated_width_mm=width*1000,tightening_mm=margin,
                requested_width_mm=requested,nominal_target_width_mm=(360.-angle)/350.*95.,
                target_deg=angle,mapping='nominal_95mm_v1')

def command_from_plan(plan):
    command=plan.get('grip_command')
    if command is None:return float(plan['config']['gripper_closed_deg'])
    expected=closing_command(plan['grasp_width_m'],command['tightening_mm'],plan['config'])
    for key in ('estimated_width_mm','requested_width_mm','nominal_target_width_mm','target_deg'):
        if not math.isclose(float(command[key]),expected[key],abs_tol=1e-6):raise ValueError('夹爪宽度计划不一致，请重新规划')
    return expected['target_deg']
