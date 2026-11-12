"""C2 joint-space curves with zero-dwell through points and bounded retiming."""
from copy import deepcopy
import numpy as np
from control_center_joint_control.joint_definitions import ALL_JOINTS
from .teaching_motion import checked_joints,limits


def coefficients(start,target,v0,v1,seconds):
    a=np.array([start[n] for n in ALL_JOINTS]);d=np.array([target[n]-start[n] for n in ALL_JOINTS])
    x=np.array([v0[n]*seconds for n in ALL_JOINTS]);y=np.array([v1[n]*seconds for n in ALL_JOINTS])
    return np.array([a,x,np.zeros_like(a),10*d-6*x-4*y,-15*d+8*x+7*y,6*d-3*x-3*y])


def evaluate(segment,progress,order=0):
    if 'curve' in segment:
        from .teaching_track import evaluate_curve
        return evaluate_curve(segment,progress,order)
    c=segment['coefficients'];u=max(0.,min(1.,float(progress)))
    for _ in range(order):c=c[1:]*np.arange(1,len(c))[:,None]
    value=sum(c[i]*u**i for i in range(len(c)))/(segment['seconds']**order)
    return dict(zip(ALL_JOINTS,map(float,value)))


def build(record,entry):
    record=deepcopy(record);points=[checked_joints(entry)]+[f['joints_deg'] for f in record['keyframes']]
    zero={n:0. for n in ALL_JOINTS}
    def make():
        times=[f['transition_sec'] for f in record['keyframes']];vel=[dict(zero) for _ in points]
        for k in range(1,len(points)-1):
            if record['keyframes'][k-1]['hold_sec']>0:continue
            for n in ALL_JOINTS:
                a=(points[k][n]-points[k-1][n])/times[k-1];b=(points[k+1][n]-points[k][n])/times[k]
                # This bound also guarantees nonnegative Bernstein coefficients
                # of the derivative: no joint overshoot between taught poses.
                if a*b>0:vel[k][n]=np.sign(a)*min(abs(a),abs(b))
        return [dict(start=points[i],target=points[i+1],v0=vel[i],v1=vel[i+1],seconds=t,
                     coefficients=coefficients(points[i],points[i+1],vel[i],vel[i+1],t)) for i,t in enumerate(times)]
    from .trajectory_extrema import peak_absolute
    segments=make();factor=1.
    for s in segments:
        c=s['coefficients'];t=s['seconds']
        for order in (1,2,3):
            c=c[1:]*np.arange(1,len(c))[:,None]
            # 五次曲线导数为低阶多项式，求真实峰值而不是粗Lipschitz上界。
            # 原上界会把一个合法夹爪动作夸大为1.75倍，再拖慢整段双臂动作。
            bound=np.array([peak_absolute(c[:,j]) for j in range(len(ALL_JOINTS))])/t**order
            for j,n in enumerate(ALL_JOINTS):factor=max(factor,(bound[j]/limits(n)[order-1])**(1/order))
    if factor>1.000001:
        for f in record['keyframes']:
            f['transition_sec']*=float(factor)*1.001
            if f['transition_sec']>30.:raise ValueError('平滑限速后单段超过30秒，请添加中间姿态')
        segments=make()
    return record,segments


def sample(record,segments,elapsed):
    elapsed=max(0.,float(elapsed))
    for i,(f,s) in enumerate(zip(record['keyframes'],segments)):
        if elapsed<s['seconds']:return evaluate(s,elapsed/s['seconds']),i,False
        elapsed-=s['seconds']
        if elapsed<f['hold_sec']:return dict(s['target']),i,False
        elapsed-=f['hold_sec']
    return dict(segments[-1]['target']),len(segments)-1,True


def plan(record,entry,directory):
    if 'trajectory' in record:
        from .teaching_track import plan as plan_track
        return plan_track(record,entry,directory)
    from .teaching_motion import stage_action
    staged,phases=stage_action(record,entry)
    staged,segments=build(staged,entry)
    # 身体段保持上一阶段夹持意图；不要把持物开口当作新的松爪目标。
    for segment,phase in zip(segments,phases):
        segment['hold_grippers']=phase[1]=='关节到位并停稳'
        segment['gripper_command_timing']=True
    for i,segment in enumerate(segments):
        # 反向点速度必须过零，但不等于需要到点后停留。夹爪交互点除外。
        segment['continuous_boundary']=(i+1<len(segments) and staged['keyframes'][i]['hold_sec']==0
            and segment['hold_grippers'] and segments[i+1]['hold_grippers'])
    return staged,segments,phases
