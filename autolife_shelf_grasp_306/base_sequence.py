"""Bounded odometry-relative base steps; no map, ROS or actuator dependencies."""
import math
import numpy as np

LINEAR_SPEED_MAX_CM_S=30.
ANGULAR_SPEED_MAX_DEG_S=60.
LINEAR_ACCEL=.25
ANGULAR_ACCEL=.6
# Receiver admits scans up to 0.5s old; reserve another 0.1s for processing/stop transport.
REACTION_TIME=.6

MOVES={'forward':'前进','backward':'后退','left':'左移','right':'右移','turn_left':'左转','turn_right':'右转'}

def normalize(value):
    value=value or {};enabled=value.get('enabled',False)
    if not isinstance(enabled,bool):raise ValueError('底盘启用状态无效')
    rows=value.get('moves',[])
    if not isinstance(rows,list) or len(rows)>30 or (enabled and not rows):raise ValueError('启用底盘动作需填写1～30项')
    result=[]
    for row in rows:
        kind=row.get('kind');amount=float(row.get('amount',0));speed=float(row.get('speed',20. if str(kind).startswith('turn_') else 10.))
        if kind not in MOVES or not math.isfinite(amount+speed):raise ValueError('底盘动作参数无效')
        turn=kind.startswith('turn_')
        if not (0.5<=amount<=360. if turn else 1.<=amount<=300.):raise ValueError('转向范围0.5～360°，移动范围1～300cm')
        if not (5<=speed<=ANGULAR_SPEED_MAX_DEG_S if turn else 3<=speed<=LINEAR_SPEED_MAX_CM_S):raise ValueError('转速5～60°/s，移动速度3～30cm/s')
        result.append(dict(kind=kind,amount=amount,speed=speed))
    return dict(enabled=enabled,moves=result)

def summary(value):
    v=normalize(value)
    if not v['enabled']:return '未启用'
    return ' → '.join(f'{MOVES[r["kind"]]}{r["amount"]:g}'+('°' if r['kind'].startswith('turn_') else 'cm') for r in v['moves'])

def wrap(x):return math.atan2(math.sin(x),math.cos(x))

class Controller:
    def __init__(self,value):
        self.config=normalize(value);self.index=0;self.done=not self.config['enabled']
        self.origin=None;self.last=None;self.last_time=None;self.started=None;self.yaw_total=0.
        self.stable=None;self.velocity=np.zeros(3)

    def update(self,pose,twist,now,paused=False):
        pose=np.asarray(pose,float);twist=np.asarray(twist,float)
        if pose.shape!=(3,) or twist.shape!=(3,) or not np.isfinite(np.r_[pose,twist,now]).all():raise ValueError('底盘反馈无效')
        if self.done:return [0.,0.,0.]
        if self.last_time is not None:
            dt=now-self.last_time
            if not 0<dt<=.5:raise ValueError('底盘反馈中断')
            dyaw=wrap(pose[2]-self.last[2])
            if np.linalg.norm(pose[:2]-self.last[:2])>.12 or abs(dyaw)>.35:raise ValueError('里程计突变，已停止底盘动作')
            self.yaw_total+=dyaw
        else:dt=.02
        self.last=pose.copy();self.last_time=now
        if self.origin is None:
            self.origin=pose.copy();self.started=now;self.yaw_total=0.;self.stable=None
        if paused:
            # Keep odometry continuity while excluding an obstacle hold from the motion timeout.
            self.started+=dt;self.velocity[:]=0.;self.stable=None
            return [0.,0.,0.]
        row=self.config['moves'][self.index];turn=row['kind'].startswith('turn_')
        speed=row['speed']*(math.pi/180 if turn else .01)
        target=row['amount']*(math.pi/180 if turn else .01)
        sign=-1 if row['kind'] in ('backward','right','turn_right') else 1
        delta=pose[:2]-self.origin[:2];yaw=pose[2];heading=wrap(yaw-self.origin[2])
        if turn:
            if np.linalg.norm(delta)>.08:raise ValueError('原地转向发生过大位置漂移')
            error=target-sign*self.yaw_total;tol=min(target*.2,math.radians(1.));axis=2;acc=ANGULAR_ACCEL
        else:
            axis=1 if row['kind'] in ('left','right') else 0
            c,s=math.cos(self.origin[2]),math.sin(self.origin[2]);local=np.array([[c,s],[-s,c]])@delta
            error=target-sign*local[axis];tol=min(target*.2,.008);acc=LINEAR_ACCEL
            if abs(local[1-axis])>.08 or abs(heading)>.15:raise ValueError('底盘偏离指定直线方向')
        if now-self.started>target/speed*3+8:raise ValueError('底盘动作超时，未自动重复该段移动')
        if error < -max(3*tol,.04 if not turn else .08):raise ValueError('底盘超出本段目标，已停止')
        desired=np.zeros(3)
        if error<=tol:
            self.velocity[:]=0
            still=math.hypot(*twist[:2])<.015 and abs(twist[2])<.025
            self.stable=(self.stable if self.stable is not None else now) if still else None
            if self.stable is not None and now-self.stable>=.3:
                self.index+=1;self.done=self.index==len(self.config['moves']);self.origin=None
            return [0.,0.,0.]
        self.stable=None
        desired[axis]=sign*min(speed,math.sqrt(2*acc*max(0,error-tol)),max(.015 if not turn else .035,error*1.5))
        if not turn:desired[2]=max(-.15,min(.15,-heading*1.5))
        limits=np.array([LINEAR_ACCEL,LINEAR_ACCEL,ANGULAR_ACCEL])*dt
        self.velocity+=np.clip(desired-self.velocity,-limits,limits)
        return self.velocity.tolist()

def scan_points(ranges,angle_min,increment,range_min,range_max,front):
    r=np.asarray(ranges,float)
    if r.ndim!=1 or len(r)<30 or not math.isfinite(angle_min+increment+range_min+range_max) or increment==0:
        raise ValueError('激光数据无效')
    valid=(np.isfinite(r)&(r>=max(.03,range_min))&(r<=range_max))|np.isposinf(r)
    if valid.mean()<.2:raise ValueError('有效激光数据不足')
    a=angle_min+np.arange(len(r))*increment;keep=np.isfinite(r)&valid
    x=r[keep]*np.cos(a[keep]);y=r[keep]*np.sin(a[keep])
    # Same fixed mounting as robot306 bringup_localization.launch.py.
    points=np.column_stack([.215-x,y]) if front else np.column_stack([-.215+x,-y])
    # Match control_center_navigation/config/navigation_self_filter.yaml:
    # strictly inside the chassis, NOT the padded 50cm collision footprint.
    # Discard self hits only; do not convert them to free-space observations.
    inside=(np.abs(points[:,0])<.18)&(np.abs(points[:,1])<.16)
    return points[~inside]

def clearance(points,velocity):
    """50x50cm base + 3cm margin, reaction then bounded-deceleration sweep.

    Do not erase isolated returns: thin real objects also produce few points.
    Outside this swept body, observations do not prevent a quantitative move.
    """
    v=np.asarray(velocity,float);p=np.asarray(points,float).reshape(-1,2)
    if v.shape!=(3,) or not np.isfinite(np.r_[v,p.ravel()]).all():raise ValueError('底盘/障碍数据无效')
    if not np.any(v) or not len(p):return True
    linear=float(np.linalg.norm(v[:2]));angular=abs(float(v[2]))
    horizon=REACTION_TIME+max(linear/LINEAR_ACCEL,angular/ANGULAR_ACCEL)
    steps=max(1,int(math.ceil(horizon/.02)));dt=horizon/steps
    # Pad by maximum within-sample travel so discretization cannot miss a narrow return.
    half=.28+(linear+angular*math.sqrt(2)*.28)*dt/2
    center=np.zeros(2);yaw=0.
    for i in range(steps+1):
        c,s=math.cos(yaw),math.sin(yaw)
        q=(p-center)@np.array([[c,-s],[s,c]])
        if np.any((np.abs(q[:,0])<half)&(np.abs(q[:,1])<half)):return False
        elapsed=max(0.,(i+.5)*dt-REACTION_TIME)
        speed=max(0.,linear-LINEAR_ACCEL*elapsed)
        turn=math.copysign(max(0.,angular-ANGULAR_ACCEL*elapsed),v[2])
        mid=yaw+turn*dt/2;c,s=math.cos(mid),math.sin(mid)
        body=v[:2]*(speed/linear) if linear else np.zeros(2)
        center+=np.array([[c,-s],[s,c]])@body*dt
        yaw+=turn*dt
    return True


class ObstacleHold:
    """Immediate stop; resume only after a fresh clear, stationary interval."""
    def __init__(self,timeout=30.,clear_seconds=.6):
        self.timeout=timeout;self.clear_seconds=clear_seconds
        self.since=None;self.clear_since=None;self.probe=None

    @property
    def active(self):return self.since is not None

    def update(self,points,command,measured,now):
        probe=self.probe if self.active else command
        clear=clearance(points,probe) and clearance(points,measured)
        if not clear:
            if not self.active:
                self.since=now;self.probe=list(command if any(command) else measured)
            self.clear_since=None
        elif self.active:
            stationary=math.hypot(*measured[:2])<.015 and abs(measured[2])<.025
            self.clear_since=(self.clear_since if self.clear_since is not None else now) if stationary else None
            if self.clear_since is not None and now-self.clear_since>=self.clear_seconds:
                self.since=None;self.clear_since=None;self.probe=None
                return [0.,0.,0.]
        if self.active:
            if now-self.since>=self.timeout:raise ValueError('运动范围内障碍持续30秒，已停止；清理后可重试本节点')
            return [0.,0.,0.]
        return list(command)
