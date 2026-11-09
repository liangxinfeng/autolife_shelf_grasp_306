"""整段原厂轨迹适配：离线准备时序，执行时不再用反馈推进小目标。

仅在用户点击执行后由 execution 调用。所有通信、停用和界面心跳
检查仍由现有执行器提供；闭爪前和收回结束核对实际到位。
"""
import time
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from autolife_handeye_calibration_306.core import BODY, ARMS, NECK
from autolife_handeye_calibration_306.verification_trajectory import JointPath
from control_center_joint_control.joint_definitions import GROUPS


def timed_samples(path, caps, accels):
    """离线前后扫描定时：只在曲率大的局部减速，不把整段一同拖慢。

    q'=dq/ds；关节加速度=q''·路径速度²+q'·路径加速度。
    两项各分配40%上限，余量用于离散采样；首尾路径速度为零。
    结果是原厂插值器需要的等时间间隔关节姿态，不读取实机反馈。
    """
    s=np.linspace(0,path.length,max(3,int(np.ceil(path.length/.1))+1))
    d1=np.abs(path.curve(s,1));d2=np.abs(path.curve(s,2))
    speeds=np.minimum(np.min(caps/np.maximum(d1,1e-9),axis=1),
        np.sqrt(np.min(.4*accels/np.maximum(d2,1e-9),axis=1)))
    acceleration=np.min(.4*accels/np.maximum(d1,1e-9),axis=1)
    speeds[0]=speeds[-1]=0.
    for i in range(1,len(s)):
        a=min(acceleration[i-1],acceleration[i])
        speeds[i]=min(speeds[i],np.sqrt(speeds[i-1]**2+2*a*(s[i]-s[i-1])))
    for i in range(len(s)-2,-1,-1):
        a=min(acceleration[i+1],acceleration[i])
        speeds[i]=min(speeds[i],np.sqrt(speeds[i+1]**2+2*a*(s[i+1]-s[i])))
    intervals=2*np.diff(s)/np.maximum(speeds[:-1]+speeds[1:],1e-9)
    stamps=np.r_[0.,np.cumsum(intervals)]
    duration=max(.3,float(stamps[-1]))
    for _ in range(6):
        count=max(3,int(np.ceil(duration*60))+1)
        t=np.linspace(0,stamps[-1],count)
        index=np.minimum(np.searchsorted(stamps,t,side='right')-1,len(intervals)-1)
        tau=t-stamps[index]
        a=(speeds[index+1]-speeds[index])/intervals[index]
        at=s[index]+speeds[index]*tau+.5*a*tau*tau
        sampled=path.curve(np.clip(at,0,path.length))
        sampled[0]=path.at(0);sampled[-1]=path.at(path.length)
        dt=duration/(count-1)
        velocity=np.diff(sampled,axis=0)/dt
        accel=np.diff(velocity,axis=0)/dt
        factor=max(float(np.max(np.abs(velocity)/caps)),
                   float(np.sqrt(np.max(np.abs(accel)/accels))),1.)
        if factor<=1.001:return sampled,duration
        duration*=factor*1.02
    raise ValueError('原厂整段轨迹定时未收敛')


def build_blocks(plan, cfg):
    """预抓/抓点连续通过；闭爪后抬起/收回连续通过，不在路径采样点等待。"""
    names = plan['controlled_joints']
    source = plan['source_q']
    speed = float(cfg.get('vendor_grasp_arm_speed_deg_s', 60.))
    waist_speed = float(cfg.get('vendor_grasp_waist_speed_deg_s', 20.))
    if not 0 < speed <= 120 or not 0 < waist_speed <= 30:
        raise ValueError('原厂抓取轨迹速度配置无效')
    from .grasp_speed import checked_percent
    scale=checked_percent(cfg.get('grasp_speed_percent',100.))/100.
    speed*=scale;waist_speed*=scale
    caps = np.array([waist_speed if n in BODY else speed for n in names])
    accels = np.array([80. if n in BODY else 240. for n in names])*scale*scale
    pending = []; blocks = []
    for segment in plan['segments']:
        pending.append(segment)
        if segment['phase'] != 'grasp' and segment is not plan['segments'][-1]:
            continue
        points=[]
        for stage in pending:
            if points:
                if max(abs(stage['waypoints'][0][n]-points[-1][n]) for n in names)>1e-6:
                    raise ValueError('原厂轨迹阶段不连续')
                points.extend(stage['waypoints'][1:])
            else:points.extend(stage['waypoints'])
        sampled,duration=timed_samples(JointPath(names,points),caps,accels)
        blocks.append(dict(name='接近并夹取' if not blocks else '抬起并收回',
            carrying=bool(blocks), close_after=segment['phase']=='grasp',
            duration_sec=float(duration),
            waypoints=[{**source, **dict(zip(names, map(float, row)))} for row in sampled]))
        pending = []
    if not blocks or sum(b['close_after'] for b in blocks) != 1:
        raise ValueError('原厂抓取轨迹缺少唯一闭爪边界')
    return blocks


def validate_blocks(plan):
    """在使能前校验控制范围、整段时序和模型已检查凭据。"""
    validate_reset_reference(plan['config'])
    side = plan['config']['arm']
    names = BODY[2:]+ARMS[side]
    if (plan.get('controlled_joints') != names or not plan.get('native_checked_states') or
            plan.get('ik_backend') != 'autolife_robot_sdk.KdlSolver.nr_jl'):
        raise ValueError('原厂抓取计划缺少腰臂范围或路径检查')
    expected = build_blocks(plan, plan['config'])
    if plan.get('vendor_blocks') != expected:
        raise ValueError('整段轨迹与预览路径不一致，请重新规划')
    source = plan['source_q']
    for segment in plan['segments']:
        for q in segment['waypoints']:
            if any(not np.isfinite(q[n]) or abs(q[n]-source[n])>1e-7
                   for n in source if n not in names):
                raise ValueError('原路径改变了固定关节，不能通过重采样掩盖')
    for block in expected:
        for q in block['waypoints']:
            if any(not np.isfinite(q[n]) for n in source):
                raise ValueError('轨迹包含无效关节值')
            if any(abs(q[n]-source[n]) > 1e-7 for n in source if n not in names):
                raise ValueError('抓取轨迹改变了未授权的关节')
            for n, (lo, hi) in plan['waist_bounds_deg'].items():
                if not lo-1e-7 <= q[n] <= hi+1e-7:
                    raise ValueError('抓取腰部超出预检范围')
    if plan['config'].get('fixed_carry_enabled') and any(
            abs(expected[-1]['waypoints'][-1][n]-source[n]) > 1e-6 for n in BODY):
        raise ValueError('收回终点没有恢复准备时的腰部角度')


def validate_reset_reference(cfg):
    """复位姿态仅作为抓取臂回程参照；保存发生变化须重算整条路径。"""
    path = cfg.get('workspace_reset_path')
    if not path:
        return
    try:
        current = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError('保存的复位姿态已不可读取，请重新规划抓取') from exc
    if current != cfg.get('workspace_reset_sha256'):
        raise ValueError('保存的复位姿态已经改变，请重新规划抓取')



class NativeTrajectoryAdapter:
    """只实现原厂插值需要的四个接口，绝不实例化电机/CAN控制器。"""
    def __init__(self, node, plan, checked, clock):
        self.node = node; self.plan = plan; self.checked = checked
        self.realtime_clock = SimpleNamespace(now=clock)
        self.last = dict(plan['source_q']); self.side = plan['config']['arm']
        self.on_sample=lambda actual,command:None

    def vectors(self, q):
        return ([q[n] for n in BODY+ARMS['left']],
                [q[n] for n in BODY+ARMS['right']], [q[n] for n in NECK])

    def _get_current_body_joints_deg(self):
        return self.vectors(self.checked())

    def _get_current_target_joints_deg(self):
        return self.vectors(self.last)

    def _set_target_body_joints_deg(self, left, right):
        actual=self.checked()  # 停止在每次原厂写目标之前生效；异常直接退出同步调用。
        if len(left) != 11 or len(right) != 11 or not np.allclose(left[:4], right[:4], atol=1e-9):
            raise ValueError('原厂轨迹返回的腰腿/手臂顺序无效')
        q = {**self.last, **dict(zip(BODY+ARMS['left']+ARMS['right'],
                                   list(left)+list(right[4:])))}
        if not all(np.isfinite(v) for v in q.values()):
            raise ValueError('原厂轨迹返回了无效目标')
        for n in BODY[:2]+ARMS['right' if self.side=='left' else 'left']:
            if abs(q[n]-self.plan['source_q'][n]) > 1e-6:
                raise ValueError('原厂插值意外改变固定关节')
        payload = {GROUPS[g]['command_key']: [q[n] for n,*_ in GROUPS[g]['joints']]
                   for g in ('leg_waist', self.side+'_arm')}
        self.node._publish_vendor(self.node._whole_body_pub, payload)
        self.node._last_operator_activity = time.monotonic()
        self.last = q
        self.on_sample(actual,q)


def load_trajectory_function(cfg):
    # 执行端只加载原厂轨迹函数，不能从规划模块间接引入 Pinocchio。
    import sys
    for path in (cfg['vendor_python_path'], '/usr/lib/python3/dist-packages'):
        if path not in sys.path:sys.path.append(path)
    from autolife_robot_arm.arm_task_control_base import ArmTaskControlBase
    return ArmTaskControlBase.move_joints_trajectory


def execute_blocks(node, plan, checked, clock, close_gripper, hold, report, trajectory_function=None):
    if trajectory_function is None:trajectory_function=load_trajectory_function(plan['config'])
    adapter = NativeTrajectoryAdapter(node, plan, checked, clock)
    names = plan['controlled_joints']; contact = None
    for block in plan['vendor_blocks']:
        print('原厂整段轨迹：'+block['name'], flush=True)
        # 回调直接输出原厂位置目标，不创建JointBridge的小目标推进器。
        traj = [[q[n] for n in BODY+ARMS['left']+ARMS['right']] for q in block['waypoints']]
        start = clock()
        last_sample_report=[-float('inf')]
        def progress(actual,command):
            now=clock()
            if now-last_sample_report[0]<.2:return
            report(dict(stage=block['name'],progress=min(1.,(now-start)/block['duration_sec']),
                elapsed_sec=now-start,planned_sec=block['duration_sec'],
                measured_deg={n:actual[n] for n in names},commanded_deg={n:command[n] for n in names},
                execution_backend='vendor_trajectory_v1',position_sync_active=True))
            last_sample_report[0]=now
        adapter.on_sample=progress
        trajectory_function(adapter, traj,
            duration=block['duration_sec'], excluded_joints_name=BODY[:2], rate_hz=60.)
        endpoint = block['waypoints'][-1]
        # 闭爪前必须到抓点；不以仍在追赶的手臂去夹物。中间采样点不等反馈。
        deadline = clock()+5.; stable_since = None; last_report = -float('inf')
        while True:
            q = checked(); now = clock()
            error = max(abs(q[n]-endpoint[n]) for n in names)
            if now-last_report >= .2:
                report(dict(stage=block['name'], elapsed_sec=now-start,
                    planned_sec=block['duration_sec'], max_error_deg=error,
                    execution_backend='vendor_trajectory_v1', position_sync_active=True))
                last_report = now
            if error <= .7:
                if stable_since is None: stable_since = now
                if now-stable_since >= .15: break
            else: stable_since = None
            if now > deadline:
                worst = max(names, key=lambda n: abs(q[n]-endpoint[n]))
                raise ValueError(f'整段轨迹终点未到位：{worst}，偏差{abs(q[worst]-endpoint[worst]):.2f}°')
        if block['close_after']:
            contact = close_gripper()
    hold(node, ['leg_waist', plan['config']['arm']+'_arm'])
    return contact
