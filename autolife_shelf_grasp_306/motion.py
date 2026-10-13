"""Feedback-bounded progress and stable arrival; no hardware access."""
import numpy as np


def feedback_lead(cfg,phase='',kind=None):
    """Bounded per-stage lead; never speed up precision placement/observation."""
    if kind=='prepare':value=cfg.get('reset_feedback_lead_deg',5.);ceiling=5.
    elif kind not in ('place','observe') and phase in ('pregrasp','carry'):
        value=cfg.get('grasp_transit_feedback_lead_deg',cfg.get('grasp_feedback_lead_deg',3.));ceiling=5.
    else:value=cfg.get('grasp_feedback_lead_deg',3.);ceiling=3.
    value=float(value)
    if not np.isfinite(value) or not 0<value<=ceiling:raise ValueError('反馈领先量超出范围')
    return value


def transfer_lead(cfg,phase,kind):
    """Only free transfer uses larger lead; precision and reset stay unchanged.

    Eight degrees is the VR controller's *minimum* dynamic lead, not its
    18-degree maximum. Follower still enforces the 0.15-degree chord tube,
    adaptive tracking slowdown, one-degree stop and no stalled-clock catchup.
    """
    base=feedback_lead(cfg,phase,kind)
    if kind not in ('prepare','observe') and phase in ('pregrasp','carry','place_approach','place_return'):
        value=float(cfg.get('transfer_feedback_lead_deg',8.))
        if not np.isfinite(value) or not base<=value<=8.:raise ValueError('搬运反馈领先量超出范围')
        return value
    return base


def cartesian_speed_profile(path,samples,positions,limits):
    """Convert already checked FK samples to local, conservative speed caps.

    Do not let one short high-Jacobian region limit the whole transfer. Every
    two path-degrees take the worst ratio/speed in that cell, with the same
    25% conversion reserve as before. The follower brakes before lower caps.
    """
    s=np.asarray(samples,float);p=np.asarray(positions,float);v=np.asarray(limits,float)
    if (s.ndim!=1 or len(s)<2 or p.shape!=(len(s),3) or v.shape!=s.shape or
        not np.isfinite(np.r_[s,p.ravel(),v]).all() or np.any(np.diff(s)<=0) or
        abs(s[0])>1e-8 or abs(s[-1]-path.length)>1e-6 or np.any(v<=0)):
        raise ValueError('局部末端速度采样无效')
    ratios=np.linalg.norm(np.diff(p,axis=0),axis=1)/np.diff(s)
    caps=np.minimum(path.max_speed,np.minimum(v[:-1],v[1:])/np.maximum(1.25*ratios,1e-6))
    boundaries=np.unique(np.r_[0,np.flatnonzero(np.diff(np.floor(s[:-1]/2.)))+1,len(caps)])
    profile=[[float(s[a]),float(np.min(caps[a:b]))] for a,b in zip(boundaries[:-1],boundaries[1:])]
    profile.append([path.length,profile[-1][1]])
    PathSpeedProfile(path,profile)
    return profile


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


def intermediate_tolerance(final_tolerance,preparing=False):
    """Do not let reset's 1deg terminal tolerance swallow its path lookahead.

    Follower can reduce its lookahead to 0.6deg. A 1deg bridge/sender deadband
    then stops the motors while the follower waits for measured progress.
    Keep this deadband below the minimum lookahead, leaving arrival unchanged.
    """
    value=float(final_tolerance)
    if not np.isfinite(value) or value<=0:raise ValueError('到位容差无效')
    return min(value,.15) if preparing else value


class PathSpeedProfile:
    """A checked path-clock cap, with braking before a slower interval."""
    def __init__(self,path,knots):
        values=np.asarray(knots,float)
        if (values.ndim!=2 or values.shape[1]!=2 or len(values)<2 or
            not np.isfinite(values).all() or np.any(np.diff(values[:,0])<=0) or
            abs(values[0,0])>1e-8 or abs(values[-1,0]-path.length)>1e-6 or
            np.any(values[:,1]<=0) or np.any(values[:,1]>path.max_speed+1e-6)):
            raise ValueError('分段速度曲线无效，请重新规划')
        self.positions=values[:,0];self.caps=values[:,1];self.path=path

    def cap(self,s,dt):
        i=min(len(self.caps)-1,max(0,int(np.searchsorted(self.positions,s,side='right'))-1))
        # A full control-tick reserve prevents stepping past a slower boundary.
        remaining=np.maximum(self.positions[i+1:]-s-self.path.max_speed*dt,0.)
        future=np.sqrt(self.caps[i+1:]**2+2*self.path.acceleration*remaining)
        return min(self.caps[i],float(np.min(future)) if len(future) else self.caps[i])


class Follower:
    def __init__(self,path,now,lead=1.2,speed_profile=None,transfer_from_s=None):
        self.path=path;self.progress=0.;self.velocity=0.;self.last=now
        self.command=path.at(0);self.lead=float(lead)
        if transfer_from_s is not None and (not np.isfinite(transfer_from_s) or not 0<=transfer_from_s<=path.length):
            raise ValueError('退出阶段速度切换位置无效')
        self.transfer_from_s=transfer_from_s
        self.measured_rate=0.;self.last_measured_s=0.
        self.limited_ticks=0;self.total_ticks=0
        self.tracking={}
        self.speed_profile=PathSpeedProfile(path,speed_profile) if speed_profile is not None else None

    @property
    def final(self):return self.progress>=self.path.length-1e-6

    def step(self,q,now):
        actual=np.array([q[n] for n in self.path.names],float)
        if not np.isfinite(actual).all():raise ValueError('关节反馈无效')
        dt=min(.1,max(0.,now-self.last));self.last=now
        self.total_ticks+=1
        if self.velocity==0.:dt=min(dt,.05)
        samples=np.linspace(max(0,self.progress-max(4.,self.lead*2)),self.progress,51)
        # Refine projection so a coarse grid cannot itself cause a 1-degree stop.
        for _ in range(3):
            errors=np.max(np.abs(self.path.curve(samples)-actual),axis=1)
            nearest=int(np.argmin(errors));near_s=float(samples[nearest])
            low=float(samples[max(0,nearest-1)]);high=float(samples[min(len(samples)-1,nearest+1)])
            samples=np.linspace(low,high,11)
        delta=actual-self.path.at(near_s);distance=float(np.max(np.abs(delta)))
        self.tracking=dict(path_error_deg=distance,measured_path_s=near_s,command_path_s=self.progress,
            worst_joint=self.path.names[int(np.argmax(np.abs(delta)))],
            measured_deg=dict(zip(self.path.names,map(float,actual))),
            nearest_checked_deg=dict(zip(self.path.names,map(float,self.path.at(near_s)))),
            command_deg=dict(zip(self.path.names,map(float,self.command))))
        if distance>1.:
            raise ValueError(f'实测关节偏离预检轨迹超过1°：{self.tracking["worst_joint"]}，偏差{distance:.2f}°')
        if self.final:return None
        remaining=self.path.length-self.progress
        speed=min(self.path.max_speed,np.sqrt(max(0,2*self.path.acceleration*remaining)))
        if self.speed_profile is not None:speed=min(speed,self.speed_profile.cap(self.progress,dt))
        # Motor lag cuts corners when a far-ahead target spans curved IK knots.
        # Keep the original hard tube, but bound lookahead BEFORE reaching it.
        # Use MEASURED progress: a far-ahead command must not prematurely
        # accelerate open fingers while they still surround the placed object.
        lead=min(self.lead,3.) if self.transfer_from_s is not None and near_s<self.transfer_from_s else self.lead
        effective_lead=min(lead,max(.6,lead*min(1.,max(0.,(.65-distance)/.45))))
        # Approach the lead boundary continuously rather than repeatedly
        # accelerating into it and being clipped to zero. This affects the
        # path clock only: geometry, hard lead and collision tube are unchanged.
        rate=max(0.,(near_s-self.last_measured_s)/max(dt,1e-6))
        alpha=1.-np.exp(-dt/.12)
        self.measured_rate+=alpha*(rate-self.measured_rate);self.last_measured_s=near_s
        derivative=max(1.,float(np.max(np.abs(self.path.curve(near_s,1)))),
                       float(np.max(np.abs(self.path.curve(self.progress,1)))))
        desired_gap=.85*effective_lead/derivative
        # Anticipate the same geometric lookahead constraint on bends too;
        # leave margin before the unchanged 0.15-degree hard chord bound.
        gap_end=min(self.path.length,near_s+desired_gap)
        if self.chord_bound(near_s,gap_end)>.12:
            low=near_s;high=gap_end
            for _ in range(8):
                mid=(low+high)/2
                if self.chord_bound(near_s,mid)<=.12:low=mid
                else:high=mid
            desired_gap=min(desired_gap,low-near_s)
        following_cap=max(0.,self.measured_rate+6.*(desired_gap-(self.progress-near_s)))
        desired=min(speed,following_cap)
        self.velocity=min(speed,float(np.clip(desired,self.velocity-self.path.acceleration*dt,
                                             self.velocity+self.path.acceleration*dt)))
        proposed=min(self.path.length,self.progress+self.velocity*dt);target=self.path.at(proposed)
        def feasible(s):
            if np.max(np.abs(self.path.at(s)-actual))>effective_lead:return False
            return self.chord_bound(near_s,s)<=.15
        self.tracking['effective_lead_deg']=effective_lead
        self.tracking['measured_path_rate_deg_s']=self.measured_rate
        self.tracking['following_speed_cap_deg_s']=following_cap
        self.tracking['lookahead_chord_bound_deg']=self.chord_bound(near_s,proposed)
        if not feasible(proposed):
            self.limited_ticks+=1
            # Old all-or-nothing rejection discarded useful remaining lead,
            # reset speed to zero and repeatedly restarted acceleration. Take
            # only the feasible part of THIS tick; never bank elapsed time.
            low=self.progress;high=proposed
            if not feasible(low):
                self.velocity=0.;return None
            for _ in range(14):
                middle=(low+high)/2
                if feasible(middle):low=middle
                else:high=middle
            proposed=low;target=self.path.at(proposed)
            self.velocity=min(self.velocity,(proposed-self.progress)/max(dt,1e-6))
            if proposed-self.progress<1e-6:return None
        self.progress=proposed
        if np.max(np.abs(target-self.command))<.015 and not self.final:return None
        self.command=target
        return dict(zip(self.path.names,map(float,target)))

    def chord_bound(self,start,end):
        """Bound chord deviation, refining loose curvature bounds analytically.

        Evaluate endpoints and every derivative root of cubic-minus-chord on
        every crossed interval. The SAME 0.15 degree lookahead tube is retained.
        """
        if end<=start:return 0.
        s=self.path.s;indices=np.flatnonzero((s[:-1]<end)&(s[1:]>start))
        if not len(indices):return 0.
        lo=np.maximum(start,s[indices])-s[indices];hi=np.minimum(end,s[indices+1])-s[indices]
        a=self.path.curve.c[0,indices,:];b=self.path.curve.c[1,indices,:]
        curvature=float(max(np.max(np.abs(6*a*lo[:,None]+2*b)),np.max(np.abs(6*a*hi[:,None]+2*b))))
        coarse=curvature*(end-start)**2/8.
        if coarse<=.15:return coarse
        c=self.path.curve.c[2,indices,:];d=self.path.curve.c[3,indices,:]
        slope=(self.path.at(end)-self.path.at(start))/(end-start)
        cc=c-slope;dd=d-self.path.at(start)-(s[indices]-start)[:,None]*slope
        def error(t):return np.abs(((a*t+b)*t+cc)*t+dd)
        result=max(float(np.max(error(lo[:,None]))),float(np.max(error(hi[:,None]))))
        A=3*a;B=2*b;C=cc;quadratic=np.abs(A)>1e-12
        discriminant=B*B-4*A*C
        for sign in (-1.,1.):
            roots=np.divide(-B+sign*np.sqrt(np.maximum(discriminant,0.)),2*A,
                            out=np.zeros_like(A),where=quadratic)
            valid=quadratic&(discriminant>=0)&(roots>lo[:,None])&(roots<hi[:,None])
            if np.any(valid):result=max(result,float(np.max(error(roots)[valid])))
        linear=(~quadratic)&(np.abs(B)>1e-12)
        roots=np.divide(-C,B,out=np.zeros_like(C),where=linear)
        valid=linear&(roots>lo[:,None])&(roots<hi[:,None])
        if np.any(valid):result=max(result,float(np.max(error(roots)[valid])))
        return min(coarse,result+1e-9)


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
