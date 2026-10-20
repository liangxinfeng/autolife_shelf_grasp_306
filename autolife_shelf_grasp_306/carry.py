"""Fixed carrying endpoint contract; no hardware access."""
import numpy as np


def carry_goal(cfg, source, names):
    values=np.asarray(cfg['initial_pose_deg'][cfg.get('arm','left')+'_arm']
                      if cfg.get('carry_return_mode')=='continuous_reset' else cfg['carry_pose_deg'],float)
    if values.shape!=(len(names),) or not np.isfinite(values).all():
        raise ValueError('固定持物姿态配置无效')
    return {**source,**dict(zip(names,map(float,values)))}


def validate_carry_plan(plan,names):
    cfg=plan['config']
    if not cfg.get('fixed_carry_enabled',False):return
    segments=plan.get('segments',[])
    continuous=cfg.get('carry_return_mode')=='continuous_reset'
    phases=['pregrasp','grasp','carry'] if continuous else ['pregrasp','grasp','lift','withdraw','carry']
    if (plan.get('carry_protocol_version')!=(2 if continuous else 1) or
            [s.get('phase') for s in segments]!=phases):
        raise ValueError('缺少已检查的固定持物路径，请重新规划')
    goal=carry_goal(cfg,plan['source_q'],names)
    endpoint=segments[-1]['waypoints'][-1]
    if any(not np.isfinite(endpoint[n]) or abs(endpoint[n]-goal[n])>1e-6 for n in names):
        raise ValueError('持物终点与固定姿态不一致，请重新规划')
    start=segments[-2]['waypoints'][-1]
    first=segments[-1]['waypoints'][0]
    if any(not np.isfinite(first[n]) or not np.isfinite(start[n]) or abs(first[n]-start[n])>1e-6 for n in names):
        raise ValueError('固定持物路径不连续，请重新规划')
