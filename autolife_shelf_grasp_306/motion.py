"""Feedback-bounded progress and stable arrival; no hardware access."""
import numpy as np


def configure_grasp_path(path,cfg):
    """Analytically bound every PCHIP interval, not just the waypoint slopes.

    Half of the acceleration budget is reserved for geometric curvature and
    half for path-clock acceleration/braking. No change to checked geometry.
    """
    speed=float(cfg.get('max_joint_speed_deg_s',120.))
    accel=float(cfg.get('max_joint_acceleration_deg_s2',720.))
    if not (np.isfinite(speed) and 0<speed<=120 and np.isfinite(accel) and 0<accel<=720):
        raise ValueError('抓取速度/加速度超出已配置范围')
    d1=1.;d2=1e-6
    for i,h in enumerate(np.diff(path.s)):
        a,b,c,_=path.curve.c[:,i,:]
        d1=max(d1,float(np.max(np.abs(c))),float(np.max(np.abs(3*a*h*h+2*b*h+c))))
        d2=max(d2,float(np.max(np.abs(2*b))),float(np.max(np.abs(6*a*h+2*b))))
        roots=np.zeros_like(a);nz=np.abs(a)>1e-15;roots[nz]=-b[nz]/(3*a[nz])
        valid=nz&(roots>0)&(roots<h)
        if np.any(valid):d1=max(d1,float(np.max(np.abs((3*a*roots*roots+2*b*roots+c)[valid]))))
    path.max_speed=min(speed/d1,np.sqrt((accel/2)/d2))
    path.acceleration=(accel/2)/d1
    return path


def configure_reset_path(path,cfg):
    # 2x time scaling of the previous VR-derived profile: velocity x2,
    # acceleration x4. Synchronize arms/neck without changing shared profiles.
    return configure_grasp_path(path,{
        'max_joint_speed_deg_s':cfg.get('reset_joint_speed_deg_s',70.2),
        'max_joint_acceleration_deg_s2':cfg.get('reset_joint_acceleration_deg_s2',585.)})


class GoalSender:
    """Do not continually re-arm already settled groups or identical targets."""
    def __init__(self):self.last={}

    def needed(self,group,targets,actual,tolerance):
        if max(abs(actual[n]-v) for n,v in targets.items())<=tolerance:return False
        previous=self.last.get(group)
        if previous is not None and max(abs(previous[n]-v) for n,v in targets.items())<.015:return False
        self.last[group]=dict(targets)
        return True


class Follower:
    def __init__(self,path,now,lead=1.2):
        self.path=path;self.progress=0.;self.velocity=0.;self.last=now
        self.command=path.at(0);self.lead=float(lead)
        self.limited_ticks=0;self.total_ticks=0

    @property
    def final(self):return self.progress>=self.path.length-1e-6

    def step(self,q,now):
        actual=np.array([q[n] for n in self.path.names],float)
        if not np.isfinite(actual).all():raise ValueError('关节反馈无效')
        dt=min(.1,max(0.,now-self.last));self.last=now
        self.total_ticks+=1
        if self.velocity==0.:dt=min(dt,.05)
        samples=np.linspace(max(0,self.progress-max(4.,self.lead*2)),self.progress,51)
        if np.min(np.max(np.abs(self.path.curve(samples)-actual),axis=1))>1.:
            raise ValueError('实测关节偏离预检轨迹超过1°')
        if self.final:return None
        remaining=self.path.length-self.progress
        speed=min(self.path.max_speed,np.sqrt(max(0,2*self.path.acceleration*remaining)))
        self.velocity=min(speed,self.velocity+self.path.acceleration*dt)
        proposed=min(self.path.length,self.progress+self.velocity*dt);target=self.path.at(proposed)
        if np.max(np.abs(target-actual))>self.lead:
            self.limited_ticks+=1
            # Old all-or-nothing rejection discarded useful remaining lead,
            # reset speed to zero and repeatedly restarted acceleration. Take
            # only the feasible part of THIS tick; never bank elapsed time.
            low=self.progress;high=proposed
            if np.max(np.abs(self.path.at(low)-actual))>self.lead:
                self.velocity=0.;return None
            for _ in range(14):
                middle=(low+high)/2
                if np.max(np.abs(self.path.at(middle)-actual))<=self.lead:low=middle
                else:high=middle
            proposed=low;target=self.path.at(proposed)
            self.velocity=min(self.velocity,(proposed-self.progress)/max(dt,1e-6))
            if proposed-self.progress<1e-6:return None
        self.progress=proposed
        if np.max(np.abs(target-self.command))<.015 and not self.final:return None
        self.command=target
        return dict(zip(self.path.names,map(float,target)))


class Arrival:
    def __init__(self,goal,tolerance,duration=.35):
        self.goal=goal;self.tolerance=tolerance;self.duration=duration;self.samples=[]

    def update(self,q,now):
        values=np.array([q[n] for n in self.goal],float)
        if not np.isfinite(values).all():raise ValueError('到位反馈无效')
        if max(abs(q[n]-v) for n,v in self.goal.items())>self.tolerance:
            self.samples=[];return False
        self.samples.append((now,values))
        while len(self.samples)>1 and now-self.samples[1][0]>=self.duration:self.samples.pop(0)
        if now-self.samples[0][0]<self.duration:return False
        return bool(np.max(np.ptp([v for _,v in self.samples],axis=0))<min(.25,self.tolerance))


def transit_pose_ok(current_pose,planned_pose,tcp,position_limit_m=.008,angle_limit_deg=2.):
    """Only pregrasp/withdraw: joint arrival must ALSO pass Cartesian checks.

    This is not used to relax grasp/lift accuracy or the no-follow watchdog.
    """
    from scipy.spatial.transform import Rotation
    current=np.asarray(current_pose,float);goal=np.asarray(planned_pose,float);tcp=np.asarray(tcp,float)
    if not np.isfinite(np.r_[current.ravel(),goal.ravel(),tcp]).all():return False
    position=np.linalg.norm(current[:3,3]+current[:3,:3]@tcp-goal[:3,3]-goal[:3,:3]@tcp)
    angle=Rotation.from_matrix(goal[:3,:3].T@current[:3,:3]).magnitude()
    return bool(position<=position_limit_m and angle<=np.deg2rad(angle_limit_deg))
