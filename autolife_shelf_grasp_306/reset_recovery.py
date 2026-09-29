"""Bounded inward-only recovery from measured NECK overshoot, reset only.

No hardware limits or URDF files are changed. Arm/body violations still reject.
"""
import math
import numpy as np
from autolife_handeye_calibration_307.core import ARMS,BODY,NECK

MAX_NECK_OVERSHOOT_DEG=2.0
INSIDE_MARGIN_DEG=2.0
RECOVERY_SPEED_SCALE=0.1  # The control-centre command parser accepts 0.1..2.0.
RECOVERY_STEP_DEG=1.0     # Combined feedback lead stays 0.1deg, not 0.5deg.


def recovery_target(source,goal,kin):
    target=dict(source);changes={}
    for name in BODY+NECK+ARMS['left']+ARMS['right']:
        i=kin.joint_q[name];lo,hi=map(float,np.rad2deg([kin.lower[i],kin.upper[i]]))
        value=float(source[name]);final=float(goal[name])
        if not math.isfinite(value) or not math.isfinite(final):raise ValueError('关节数据无效：'+name)
        if final<lo-1e-5 or final>hi+1e-5:
            raise ValueError(f'复位目标越界：{name} {final:.2f}°，允许{lo:.2f}～{hi:.2f}°')
        over=max(lo-value,value-hi,0.)
        if over<=1e-5:continue
        if name not in NECK or over>MAX_NECK_OVERSHOOT_DEG:
            raise ValueError(f'当前{name}={value:.2f}°，超出URDF {lo:.2f}～{hi:.2f}°约{over:.2f}°；'
                             '不允许自动复位，请检查关节零位及硬件状态')
        target[name]=lo+INSIDE_MARGIN_DEG if value<lo else hi-INSIDE_MARGIN_DEG
        changes[name]={'source_deg':value,'target_deg':target[name],'lower_deg':lo,'upper_deg':hi}
    return target,changes


def check_recovery_sample(source,values,changes,tolerance=1e-6):
    for name in BODY+NECK+ARMS['left']+ARMS['right']:
        value=float(values[name])
        if not math.isfinite(value):raise ValueError('回退反馈无效：'+name)
        if name in changes:
            a,b=source[name],changes[name]['target_deg']
            if not min(a,b)-tolerance<=value<=max(a,b)+tolerance:
                raise ValueError('颈部回退偏离已检查范围：'+name)
        elif abs(value-source[name])>tolerance:
            raise ValueError('颈部回退期间其他关节发生变化：'+name)


def check_recovery_geometry(guard,source,path,changes):
    """Check REAL out-of-range starting geometry, never clip measured angles.

    The temporary envelope only contains the known initial neck value and an
    inward-only segment. Restore strict bounds and clear cache even on failure.
    """
    low=guard.kin.lower.copy();high=guard.kin.upper.copy()
    try:
        for name,row in changes.items():
            i=guard.kin.joint_q[name];r=math.radians(row['source_deg'])
            guard.kin.lower[i]=min(low[i],r);guard.kin.upper[i]=max(high[i],r)
        guard.cache.clear();count=0;previous=dict(source)
        for partial in path.checks():
            values={**source,**partial};check_recovery_sample(source,values,changes)
            for name,row in changes.items():
                direction=1 if row['target_deg']>row['source_deg'] else -1
                if direction*(values[name]-previous[name]) < -1e-7:raise ValueError('回退轨迹非单调')
            guard.check(values);previous=values;count+=1
        return count
    finally:
        guard.kin.lower[:]=low;guard.kin.upper[:]=high;guard.cache.clear()


def execute_recovery(node,segment,checked,hold,now):
    """A small neck-only target inside normal limits; never chase a bad sensor."""
    source=segment['waypoints'][0];target=segment['waypoints'][-1]
    changes=segment['neck_recovery'];start=now()
    if not changes or any(n not in NECK for n in changes):raise ValueError('无效的颈部回退计划')
    for name,row in changes.items():
        a,b=float(source[name]),float(target[name]);lo,hi=row['lower_deg'],row['upper_deg']
        if (max(lo-a,a-hi,0.)>MAX_NECK_OVERSHOOT_DEG or not lo+1.<=b<=hi-1.
                or b!=row['target_deg'] or a!=row['source_deg']):raise ValueError('颈部回退计划范围无效')
    # No full-body reset, gripper opening or arm motion before this finishes.
    q=checked();check_recovery_sample(source,q,changes,tolerance=.3)
    from rclpy.parameter import Parameter
    previous={n:float(node.get_parameter(n).value) for n in
              ('goal_tolerance_deg','max_single_step_deg')}
    try:
        result=node.set_parameters([Parameter('goal_tolerance_deg',value=.25),
                                    Parameter('max_single_step_deg',value=RECOVERY_STEP_DEG)])
        if not all(v.successful for v in result):raise ValueError('无法设置颈部回退到位检查')
        targets={n:target[n] for n in NECK}
        previous_goal=node._active_goals.get('neck')
        node.apply('neck',targets,speed_scale=RECOVERY_SPEED_SCALE)
        # apply() returns None even when the real bridge rejects a command.
        # Verify a NEW accepted goal synchronously, before waiting for motion.
        accepted=node._active_goals.get('neck')
        if (accepted is None or accepted is previous_goal or accepted.get('targets')!=targets
                or accepted.get('speed_scale')!=RECOVERY_SPEED_SCALE):
            raise ValueError('颈部回退指令未被接受：'+str(getattr(node,'_last_event','请检查控制接口')))
        _wait_recovery(node,source,target,changes,checked,hold,now,start)
    finally:
        node.set_parameters([Parameter(n,value=v) for n,v in previous.items()])


def _wait_recovery(node,source,target,changes,checked,hold,now,start):
    stable=None;best=dict(source)
    while True:
        q=checked();stamp=now()
        check_recovery_sample(source,q,changes,tolerance=.7)
        for name,row in changes.items():
            direction=1 if row['target_deg']>source[name] else -1
            if direction*(q[name]-best[name])<-.25:raise ValueError('颈部未向限位内回退，已停止')
            if direction*(q[name]-best[name])>0:best[name]=q[name]
        reached=all(abs(q[n]-target[n])<=.3 for n in changes)
        stable=stamp if reached and stable is None else stable if reached else None
        if stable is not None and stamp-stable>=.2:
            hold(node,['neck']);return
        if stamp-start>8.:
            detail='，'.join(f'{n} 实测{q[n]:.2f}°/目标{target[n]:.2f}°' for n in changes)
            raise ValueError('颈部回退未到位：'+detail+'；未开始双臂复位')
