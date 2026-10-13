"""Step posture settings and bounded navigation corridor; angles in degrees."""
import math


def normalize(value):
    if value is None:return None
    if not isinstance(value,dict):raise ValueError('步骤姿态格式无效')
    mode=value.get('body_mode','keep');height=value.get('lowering_cm',0);pitch=value.get('head_pitch_deg')
    if mode not in ('keep','forward','reverse'):raise ValueError('请选择保持、正向或反向下蹲')
    if isinstance(height,bool):raise ValueError('下降高度应填写厘米数')
    height=float(height)
    maximum=53. if mode=='reverse' else 56.
    if not math.isfinite(height) or not 0<=height<=maximum:raise ValueError(f'下降高度应在0～{maximum:g}cm内')
    if pitch is not None:
        if isinstance(pitch,bool):raise ValueError('头部角度应填写度数')
        pitch=float(pitch)
        if not math.isfinite(pitch) or not -30<=pitch<=45:raise ValueError('头部角度应在-30～45°内（正数俯视）')
    if mode=='keep' and pitch is None:return None
    return dict(body_mode=mode,lowering_cm=height if mode!='keep' else 0.,head_pitch_deg=pitch)


def targets(setting):
    from control_center_joint_control.body_height import body_height_targets
    from control_center_joint_control.joint_definitions import JOINT_LIMITS
    value=normalize(setting);result={}
    if value is None:return result
    if value['body_mode']!='keep':
        signed=value['lowering_cm']/100.*(-1 if value['body_mode']=='reverse' else 1)
        result.update(body_height_targets(signed,allow_reverse=True))
    if value['head_pitch_deg'] is not None:result['Joint_Neck_Pitch']=-value['head_pitch_deg']
    for name,v in result.items():
        lo,hi=JOINT_LIMITS[name]
        if not math.isfinite(v) or not lo<=v<=hi:raise ValueError('步骤姿态超出关节限位：'+name)
    return result


def permitted_navigation_joints(initial,current,setting,tolerance=3.):
    """Allow only requested motion along the original-to-target joint segment.

    Coupled ankle/knee/waist must share progress; no broad exemption for body.
    All non-returned joints retain the original navigation holding checks.
    """
    desired=targets(setting)
    if not desired:return set()
    if any(n not in initial or n not in current or not math.isfinite(current[n]) for n in desired):
        raise ValueError('调姿反馈不完整')
    body=[n for n in desired if n!='Joint_Neck_Pitch']
    if body:
        delta={n:desired[n]-initial[n] for n in body}
        norm=sum(d*d for d in delta.values())
        progress=max(0.,min(1.,sum((current[n]-initial[n])*delta[n] for n in body)/norm)) if norm>1e-9 else 0.
        if any(abs(current[n]-(initial[n]+progress*delta[n]))>tolerance for n in body):
            raise ValueError('身体偏离本步骤的联动调姿轨迹，停止导航')
    n='Joint_Neck_Pitch'
    if n in desired and not min(initial[n],desired[n])-tolerance<=current[n]<=max(initial[n],desired[n])+tolerance:
        raise ValueError('颈部偏离本步骤的角度范围，停止导航')
    return set(desired)
