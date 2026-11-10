"""放置与抓取共用原厂整段位置轨迹；只在落点确认后允许张爪。

移动和收回沿用抓取速度滑块，最后下降仍受桌面接近速度约束。
本文件不导入规划器或 Pinocchio，不新建硬件控制实例。
"""
import time
import numpy as np
from autolife_handeye_calibration_306.core import BODY, ARMS
from autolife_handeye_calibration_306.verification_trajectory import JointPath
from .vendor_grasp_trajectory import timed_samples, NativeTrajectoryAdapter, validate_reset_reference
from .grasp_speed import checked_percent


def build_blocks(plan,cfg):
    names=plan['controlled_joints']; source=plan['source_q']
    scale=checked_percent(cfg.get('grasp_speed_percent',100.))/100.
    arm=float(cfg.get('vendor_grasp_arm_speed_deg_s',60.))
    waist=float(cfg.get('vendor_grasp_waist_speed_deg_s',20.))
    if not 0<arm<=120 or not 0<waist<=30:raise ValueError('放置关节速度配置无效')
    caps=np.array([waist if n in BODY else arm for n in names])*scale
    accels=np.array([80. if n in BODY else 240. for n in names])*scale**2
    result=[]
    for segment in plan['segments']:
        path=JointPath(names,segment['waypoints'])
        profile=np.asarray(segment['path_speed_profile'],float)
        if (profile.ndim!=2 or profile.shape[1]!=2 or len(profile)<2 or
                not np.isfinite(profile).all() or np.any(profile[:,1]<=0) or
                np.any(np.diff(profile[:,0])<=0) or abs(profile[0,0])>1e-6 or
                abs(profile[-1,0]-path.length)>1e-6):
            raise ValueError('放置阶段的末端速度曲线无效')
        def limit(s):
            # 相邻速度取较小值，不能通过插值提前加速进入下降/退离区域。
            i=np.clip(np.searchsorted(profile[:,0],s,side='right')-1,0,len(profile)-2)
            return np.minimum(profile[i,1],profile[i+1,1])*scale
        speed_limit=limit
        if plan.get('placement_speed_policy')=='grasp_transfer_v2':
            if segment['phase']=='place_approach':
                # 自由移入与抓取相同：仅使用共同的关节速度/加速度定时。
                speed_limit=None
            elif segment['phase']=='place_return':
                transfer=float(segment.get('transfer_from_s',path.length))
                if not np.isfinite(transfer) or not 0<=transfer<=path.length:
                    raise ValueError('放置退出边界无效')
                def speed_limit(s):
                    # 尚未退出水瓶时保持原局部约束，退出后与抓取相同。
                    return np.where(s<transfer,limit(s),1.e9)
        sampled,duration=timed_samples(path,caps,accels,speed_limit)
        result.append(dict(name=segment['name'],phase=segment['phase'],
            carrying=segment['phase']!='place_return',release_after=segment['phase']=='place_lower',
            duration_sec=float(duration),
            waypoints=[{**source,**dict(zip(names,map(float,row)))} for row in sampled]))
    return result


def validate_blocks(plan):
    if (plan.get('kind')!='place' or plan.get('place_protocol_version')!=4 or
            plan.get('execution_backend')!='vendor_place_trajectory_v1' or
            plan.get('ik_backend')!='autolife_robot_sdk.KdlSolver.nr_jl' or
            not plan.get('native_checked_states') or not plan.get('table_height_checked')):
        raise ValueError('放置缺少原厂腰臂轨迹及桌面预检')
    validate_reset_reference(plan['config'])
    names=BODY[2:]+ARMS[plan['config']['arm']];source=plan['source_q']
    if plan.get('controlled_joints')!=names:raise ValueError('放置腰臂顺序无效')
    if [s['phase'] for s in plan['segments']]!=['place_approach','place_lower','place_return']:
        raise ValueError('放置缺少接近、落点或退离阶段')
    expected=build_blocks(plan,plan['config'])
    if expected!=plan.get('vendor_blocks'):raise ValueError('放置时序与已检查路径不一致，请重新规划')
    bounds=plan['waist_bounds_deg']
    if set(bounds)!=set(BODY[2:]):raise ValueError('放置腰部边界不完整')
    previous=source
    for segment,block in zip(plan['segments'],expected):
        if any(abs(segment['waypoints'][0][n]-previous[n])>1e-6 for n in source):
            raise ValueError('放置阶段不连续')
        for q in segment['waypoints']+block['waypoints']:
            if any(not np.isfinite(q[n]) for n in source):raise ValueError('放置关节值无效')
            if any(abs(q[n]-source[n])>1e-7 for n in source if n not in names):
                raise ValueError('放置意外改变固定关节')
            for n,(lo,hi) in bounds.items():
                if not np.isfinite([lo,hi]).all() or not lo<=source[n]<=hi or not lo-1e-7<=q[n]<=hi+1e-7:
                    raise ValueError('放置腰部超出预检范围')
        previous=segment['waypoints'][-1]
    if any(abs(previous[n]-source[n])>1e-6 for n in BODY):
        raise ValueError('放置收回未恢复开始时腰腿姿态')


def execute_blocks(node,plan,checked,clock,release_gripper,hold,report,endpoint_ok,trajectory_function):
    adapter=NativeTrajectoryAdapter(node,plan,checked,clock)
    names=plan['controlled_joints']
    for block in plan['vendor_blocks']:
        start=clock();last_report=[-float('inf')]
        def progress(actual,command):
            now=clock()
            if now-last_report[0]<.2:return
            report(dict(stage=block['name'],progress=min(1.,(now-start)/block['duration_sec']),
                measured_deg={n:actual[n] for n in names},commanded_deg={n:command[n] for n in names},
                planned_sec=block['duration_sec'],elapsed_sec=now-start,
                execution_backend='vendor_place_trajectory_v1',position_sync_active=True))
            last_report[0]=now
        adapter.on_sample=progress
        trajectory_function(adapter,[[q[n] for n in BODY+ARMS['left']+ARMS['right']] for q in block['waypoints']],
            duration=block['duration_sec'],excluded_joints_name=BODY[:2],rate_hz=60.)
        endpoint=block['waypoints'][-1];deadline=clock()+5.;stable=None
        while True:
            q=checked();now=clock();error=max(abs(q[n]-endpoint[n]) for n in names)
            valid=error<=(.35 if block['release_after'] else .7) and endpoint_ok(q,endpoint,block['release_after'])
            if valid:
                if stable is None:stable=now
                if now-stable>=.15:break
            else:stable=None
            progress(q,endpoint)
            if now>deadline:raise ValueError(f'放置终点尚未到位，保持夹持：{block["name"]}，关节偏差{error:.2f}°')
        if block['release_after']:
            # 唯一松爪边界；取消、通信失败、未到位均不能越过这里。
            release_gripper()
    hold(node,['leg_waist',plan['config']['arm']+'_arm'])
