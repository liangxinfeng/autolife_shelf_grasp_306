"""Immutable measured tracks and C2 replay planning; no ROS/motor imports."""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import uuid
import zipfile
import numpy as np
from control_center_joint_control.joint_definitions import ALL_JOINTS, BODY_JOINTS, GRIPPER_JOINTS, JOINT_LIMITS, GROUP_ORDER

NAMES = list(ALL_JOINTS)
JAWS = [NAMES.index(n) for n in GRIPPER_JOINTS]
BODY = [NAMES.index(n) for n in BODY_JOINTS]


def check_hash(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-f0-9]{64}', value):
        raise ValueError('连续轨迹校验码无效')
    return value


def validate_arrays(t, q):
    t, q = np.asarray(t, dtype=float), np.asarray(q, dtype=float)
    if t.ndim != 1 or not 3 <= len(t) <= 15001 or q.shape != (len(t), len(NAMES)):
        raise ValueError('连续轨迹采样数量或关节数量不正确')
    if not np.isfinite(t).all() or not np.isfinite(q).all() or abs(t[0]) > 1e-6:
        raise ValueError('连续轨迹包含无效时间或角度')
    if np.any(np.diff(t) <= 0) or t[-1] > 300.1 or t[-1] < .2 or np.max(np.diff(t)) > .25:
        raise ValueError('轨迹时间中断、过短或超过5分钟，请保留草稿后重新录制')
    for j, n in enumerate(NAMES):
        if np.any(q[:, j] < JOINT_LIMITS[n][0]) or np.any(q[:, j] > JOINT_LIMITS[n][1]):
            raise ValueError('录制关节超出执行范围：' + n)
    return t, q


def checked_events(events,count):
    if not isinstance(events,list) or len(events)>100:raise ValueError('复位记录数量无效')
    result=[]
    for e in events:
        a=e.get('pressed');b=e.get('start');end=e.get('end')
        if e.get('kind')!='reset' or type(a) is not int or type(b) is not int or not 0<=a<=b<count:
            raise ValueError('复位记录索引无效')
        if end is not None and (type(end) is not int or not b<=end<count):raise ValueError('复位结束索引无效')
        result.append(dict(kind='reset',pressed=a,start=b,end=end))
    return result


def reset_timing(t,q,events):
    """Compress only measured quiet waits associated with an explicit reset.

    No body samples or manual dwell settings are discarded. Dynamic portions
    remain recorded, and the normal derivative bounds retime the final curve.
    """
    effective=t.copy();trimmed=0.;last_end=-1
    for event in checked_events(list(events),len(t)):
        a=max(event['pressed'],last_end+1);b=event['start']
        if b<=a:continue
        k=a;best=None
        for i in range(a+1,b+1):
            while k<i and (np.max(np.ptp(q[k:i+1,BODY],axis=0))>.15 or np.max(np.ptp(q[k:i+1,JAWS],axis=0))>.5):k+=1
            if t[i]-t[k]>.5 and (best is None or t[i]-t[k]>t[best[1]]-t[best[0]]):best=(k,i)
        if best:
            k,i=best;amount=float(t[i]-t[k]-.15)
            effective[k:i+1]=effective[k]+(t[k:i+1]-t[k])*.15/(t[i]-t[k])
            effective[i+1:]-=amount;trimmed+=amount;last_end=i
    return effective,trimmed


def save_raw(directory, t, q, events=()):
    t, q = validate_arrays(t, q)
    stream = io.BytesIO()
    events=checked_events(list(events),len(t))
    np.savez_compressed(stream, times=t, joints=q, names=np.array(NAMES),events=np.array(json.dumps(events)))
    raw = stream.getvalue(); sha = hashlib.sha256(raw).hexdigest()
    root = Path(directory) / 'trajectories'; root.mkdir(parents=True, exist_ok=True)
    target = root / (sha + '.npz')
    temp = root / (uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('wb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return sha


def load_raw(directory, sha):
    path = Path(directory) / 'trajectories' / (check_hash(sha) + '.npz')
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError('轨迹文件过大')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('轨迹内容校验失败，未执行')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if sum(x.file_size for x in z.infolist()) > 12 * 1024 * 1024:
            raise ValueError('轨迹展开数据过大')
    with np.load(io.BytesIO(raw), allow_pickle=False) as a:
        if list(a['names']) != NAMES:
            raise ValueError('轨迹关节顺序不匹配')
        return validate_arrays(a['times'], a['joints'])


def load_events(directory,sha,count):
    # load_raw must validate the immutable file's digest and size first.
    path=Path(directory)/'trajectories'/(check_hash(sha)+'.npz')
    with np.load(path,allow_pickle=False) as a:
        return checked_events(json.loads(str(a['events'])) if 'events' in a else [],count)


def jaw_events(t, q):
    """Jointly group both grippers; small encoder jitter is not an interaction."""
    events = []; baseline = q[0, JAWS].copy(); stable = 0; active = None
    for i in range(1, len(t)):
        if active is None:
            if np.max(np.abs(q[i, JAWS] - baseline)) > 3.:
                active = stable
            elif np.max(np.abs(q[i, JAWS] - baseline)) < .6:
                stable = i
        else:
            k = int(np.searchsorted(t, t[i] - .25))
            if i > k + 2 and np.max(np.ptp(q[k:i+1, JAWS], axis=0)) < 1.:
                events.append((active, i)); baseline = q[i, JAWS].copy()
                stable = i; active = None
    if active is not None:
        events.append((active, len(t)-1))
    return events


def make_draft(directory, name, t, q, markers=(), reset_events=()):
    t, q = validate_arrays(t, q);reset_events=checked_events(list(reset_events),len(t))
    sha = save_raw(directory, t, q,reset_events)
    events = jaw_events(t, q)
    indices = {0, len(t)-1, *(int(i) for i in markers if 0 <= int(i) < len(t))}
    automatic=[int(np.searchsorted(t,sec)) for sec in np.arange(5.,t[-1],5.)]
    indices.update(i for i in automatic if not any(e['pressed']<i<e['start'] for e in reset_events))
    indices.update(a for a, _ in events)
    reset_starts={e['start'] for e in reset_events};reset_ends={e['end'] for e in reset_events if e['end'] is not None}
    indices.update(reset_starts|reset_ends)
    if len(indices) > 100:
        raise ValueError('交互及标记点超过100个；原始轨迹已保留，请分段录制')
    frames = []
    interaction = {a for a, _ in events}
    for k, i in enumerate(sorted(indices)):
        label = ('起点' if i == 0 else '终点' if i == len(t)-1 else
                 '复位开始' if i in reset_starts else '复位完成' if i in reset_ends else
                 '夹爪交互' if i in interaction else '标记' if i in markers else '轨迹段')
        frames.append(dict(label=f'{label} · {t[i]:.1f}s', joints_deg=dict(zip(NAMES, map(float, q[i]))),
                           sample_index=i, hold_sec=0., speed_percent=100., minimum_sec=.6 if i==0 else .2, ghost=True))
    reordered = sum(bool(np.max(np.ptp(q[a:b+1, BODY], axis=0)) > 1.) for a, b in events if b > a)
    return dict(name=name, version=1, kind='trajectory', trajectory_ref=sha, recorded_sec=float(t[-1]),
                timing='recorded',notes=f'{len(t)}个采样；按原始时序；{len(events)}次夹爪交互，{reordered}处同步开合保留；{len(reset_events)}次复位', frames=frames)


def evaluate_curve(segment, progress, order=0):
    x = segment['curve']; u = min(1., max(0., float(progress)))
    t = segment['curve_start'] + u * (segment['curve_end'] - segment['curve_start'])
    i = min(len(x['times'])-2, max(0, int(np.searchsorted(x['times'], t, side='right')-1)))
    dt = t - x['times'][i]; c = x['coeff'][:, i, :]
    for _ in range(order):
        c = c[:-1] * np.arange(len(c)-1, 0, -1)[:, None]
    value = sum(c[j] * dt**(len(c)-1-j) for j in range(len(c))) / segment['scale']**order
    if order==0:
        # Polynomial arithmetic can produce 9.999999999999998 at a 10° limit.
        # Correct roundoff only; true spline excursions are rejected by _curve.
        for j,n in enumerate(NAMES):
            lo,hi=JOINT_LIMITS[n]
            if lo-1e-7<=value[j]<lo:value[j]=lo
            elif hi<value[j]<=hi+1e-7:value[j]=hi
    return dict(zip(NAMES, map(float, value)))


def _curve(t, q, anchor_times=()):
    from scipy.interpolate import make_interp_spline, PPoly
    from .teaching_motion import limits
    # Remove sub-frame encoder ripple, then fit a C2 path at 10 Hz. Preserve
    # endpoints exactly; raw data remains immutable for inspection/reprocessing.
    smooth = q.copy()
    if len(t) >= 7:
        kernel = np.array([1., 2., 3., 2., 1.]) / 9.
        for j in BODY:
            smooth[:, j] = np.convolve(np.pad(q[:, j], (2, 2), mode='edge'), kernel, mode='valid')
    smooth[0], smooth[-1] = q[0], q[-1]
    for at in anchor_times:
        i=int(np.argmin(np.abs(t-at)));smooth[i]=q[i]
    candidates=sorted(np.r_[t[0],np.arange(t[0]+.1,t[-1],.1),t[-1],list(anchor_times)])
    unique=[]
    for x in candidates:
        if unique and x-unique[-1]<1e-8:unique[-1]=float(x)
        else:unique.append(float(x))
    knots=np.array(unique)
    values = np.array([np.interp(knots, t, smooth[:, j]) for j in range(len(NAMES))]).T
    zero=np.zeros(len(NAMES))
    spline=make_interp_spline(knots,values,k=5,axis=0,
                             bc_type=([(1,zero),(2,zero)],[(1,zero),(2,zero)]))
    scalar=[PPoly.from_spline((spline.t,spline.c[:,j],spline.k)) for j in range(len(NAMES))]
    keep=np.flatnonzero(np.diff(scalar[0].x)>1e-10)
    knots=np.r_[scalar[0].x[keep],scalar[0].x[keep[-1]+1]]
    curve=PPoly(np.stack([p.c[:,keep] for p in scalar],axis=-1),knots)
    scale = 1.
    for j, n in enumerate(NAMES):
        for order in range(4):
            derivative = curve.derivative(order)
            points = list(knots)
            if order < 4:
                # Exact interior extrema of each polynomial derivative.
                d = curve.derivative(order+1)
                for k in range(len(knots)-1):
                    for root in np.roots(np.trim_zeros(d.c[:, k, j], trim='f')):
                        if abs(root.imag) < 1e-9 and 0 < root.real < knots[k+1]-knots[k]:
                            points.append(knots[k]+float(root.real))
            evaluated = derivative(np.array(points))[:, j]
            if order == 0:
                lo, hi = JOINT_LIMITS[n]
                if np.min(evaluated) < lo-1e-7 or np.max(evaluated) > hi+1e-7:
                    raise ValueError('平滑轨迹接近关节限位：'+n+'；请微调或重新录制')
            else:
                scale = max(scale, (np.max(np.abs(evaluated)) / limits(n)[order-1])**(1./order))
    return {'times':knots, 'coeff':curve.c}, float(scale)*1.001


def plan(record, entry, directory):
    """Return controller-compatible blocks with dense curves kept in memory."""
    if record['trajectory'].get('timing','recorded')=='recorded':
        from .teaching_replay import plan as recorded_plan
        return recorded_plan(record,entry,directory)
    from .teaching_motion import checked_joints, duration
    from .teaching_trajectory import build, evaluate
    from .task_flow import full_record
    meta = record['trajectory']; t, original = load_raw(directory, meta['sha256'])
    indices = list(meta['indices']); frames = record['keyframes']
    if len(indices) != len(frames) or indices != sorted(set(indices)) or indices[0] != 0 or indices[-1] != len(t)-1:
        raise ValueError('轨迹编辑节点顺序或首尾节点不完整')
    q = original.copy(); anchors = t[indices]
    # A local, smooth offset field makes edits explicit without throwing away
    # the measured movement between editable anchors.
    for j, n in enumerate(NAMES):
        delta = np.array([f['joints_deg'][n]-original[i, j] for i, f in zip(indices, frames)])
        for k in range(len(indices)-1):
            a, b = indices[k:k+2]; u = (t[a:b+1]-t[a])/(t[b]-t[a]); s = u**3*(10-15*u+6*u*u)
            q[a:b+1, j] = original[a:b+1,j]+delta[k]+(delta[k+1]-delta[k])*s
    validate_arrays(t, q)
    replay_time,_=reset_timing(t,original,load_events(directory,meta['sha256'],len(t)))
    speeds = meta['speeds']; warped = replay_time.copy()
    for k in range(1, len(indices)):
        a, b = indices[k-1:k+1]
        desired=max((replay_time[b]-replay_time[a])*100./speeds[k],meta['minimums'][k])*100./record['speed_percent']
        warped[a:b+1] = warped[a] + (replay_time[a:b+1]-replay_time[a])*desired/(replay_time[b]-replay_time[a])
    events = jaw_events(t, q); event_at = {a:b for a,b in events}
    holds = {i:float(f['hold_sec']) for i,f in zip(indices, frames) if f['hold_sec'] > 0}
    stops = sorted({0, len(t)-1, *event_at, *holds})
    rows=[]; segments=[]; phases=[]; current=checked_joints(entry)
    def logical(i): return min(len(indices)-1, int(np.searchsorted(indices, i)))
    def append_rest(target, hold, label, i, minimum=.3):
        nonlocal current
        target=checked_joints(target)
        row=dict(joints_deg=target, transition_sec=duration(current,target,100.,minimum), hold_sec=hold, easing='smooth')
        one=dict(groups=list(GROUP_ORDER),loop_count=1,speed_percent=100,keyframes=[row])
        one, seg=build(one,current); rows.extend(one['keyframes']); segments.extend(seg)
        phases.append((logical(i), label, 0)); current=dict(target)
    first=dict(zip(NAMES,map(float,q[0])));first.update(meta.get('entry_gripper_hold',{}));body=dict(first)
    body.update({n:current[n] for n in GRIPPER_JOINTS})
    initial_jaw=any(abs(first[n]-current[n])>.001 for n in GRIPPER_JOINTS)
    append_rest(body,.25 if initial_jaw else holds.get(0,0.),'对齐录制起点',0,frames[0]['transition_sec'])
    if initial_jaw: append_rest(first,holds.get(0,0.),'起点夹爪开合',0)
    for a, b in zip(stops, stops[1:]):
        if a in event_at:
            end=event_at[a]; target=dict(current)
            target.update({n:float(q[end,NAMES.index(n)]) for n in GRIPPER_JOINTS})
            rows[-1]['hold_sec']=max(.25,rows[-1]['hold_sec'])
            append_rest(target,holds.get(a,0.),'关节到位 → 夹爪开合',a,max(.3,float(t[end]-t[a])))
            # A stationary body interval already spent opening/closing the jaws
            # should not be played a second time as an artificial pause.
            if end <= b and np.max(np.ptp(q[a:end+1,BODY],axis=0)) <= 1.: a=end
        if b <= a: continue
        times=warped[a:b+1]-warped[a]; values=q[a:b+1].copy()
        values[:,JAWS]=[current[n] for n in GRIPPER_JOINTS]
        values[0,BODY]=[current[n] for n in BODY_JOINTS]
        if times[-1] < .02: continue
        curve,scale=_curve(times,values,[warped[i]-warped[a] for i in indices if a<i<b]);scale=max(scale,.2/float(times[-1]))
        candidates=sorted({0.,float(times[-1]),*(float(warped[i]-warped[a]) for i in indices if a<i<b),
                     *(float(x) for x in np.arange(8./scale,times[-1],8./scale))})
        cuts=[0.]
        for x in candidates[1:-1]:
            if (x-cuts[-1])*scale>=.2:cuts.append(x)
        if len(cuts)>1 and (times[-1]-cuts[-1])*scale<.2:cuts.pop()
        cuts.append(float(times[-1]))
        for start,end in zip(cuts,cuts[1:]):
            seg=dict(curve=curve,curve_start=start,curve_end=end,scale=scale,seconds=(end-start)*scale)
            seg['start']=evaluate_curve(seg,0);seg['target']=evaluate_curve(seg,1)
            seg['v0']=evaluate_curve(seg,0,1);seg['v1']=evaluate_curve(seg,1,1)
            target=checked_joints(seg['target']); final=abs(end-times[-1])<1e-8
            rows.append(dict(joints_deg=target,transition_sec=seg['seconds'],hold_sec=holds.get(b,0.) if final else 0.,easing='smooth'))
            segments.append(seg); phases.append((logical(b),'连续轨迹',0)); current=target
    if len(t)-1 in event_at:
        raise ValueError('末尾夹爪交互不完整，请重新录制')
    if len(rows)>200: raise ValueError('优化后超过200个执行段，请分段录制')
    # Legacy action normalizers require >=0.2s transitions. Merge/split spacing
    # is validated rather than silently stretching only one through-point.
    if any(f['transition_sec'] < .2-1e-8 for f in rows):
        raise ValueError('相邻标记或夹爪交互间隔过短，请间隔至少0.2秒再记录')
    result=full_record(dict(groups=list(GROUP_ORDER),loop_count=1,speed_percent=100,keyframes=rows))
    return result,segments,phases
