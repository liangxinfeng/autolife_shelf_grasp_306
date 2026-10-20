"""Empirical torso-frame target trims, separate from calibrated marker-to-TCP."""
import copy,json,math

FRAME='Link_Waist_Yaw_to_Shoulder_Inner'

def checked_profiles(value):
    if not isinstance(value,dict) or set(value)!={'left','right'}:raise ValueError('需分别提供左右手三轴补偿')
    result={}
    for side in ('left','right'):
        raw=value[side]
        if not isinstance(raw,(list,tuple)) or len(raw)!=3:raise ValueError('补偿需包含前后、左右、上下三轴')
        if any(isinstance(v,bool) for v in raw):raise ValueError('补偿数值无效')
        v=[float(x) for x in raw]
        if not all(math.isfinite(x) for x in v) or math.sqrt(sum(x*x for x in v))>50.000001:
            raise ValueError('三轴合成补偿不能超过50毫米')
        result[side]=v
    return result

def defaults(cfg):
    source=cfg.get('arm','left')
    return checked_profiles({side:cfg.get(side+'_target_offset_torso_mm',
        cfg.get('target_offset_torso_mm',[0.,15.,0.]) if side==source else [0.,0.,0.]) for side in ('left','right')})

def from_display(front,right,up):
    # URDF torso uses +X forward, +Y robot-right, -Z upward.
    return [float(front),float(right),-float(up)]

def to_display(offset):return [offset[0],offset[1],-offset[2]]

def description(offset):
    names=[('前','后'),('右','左'),('上','下')]
    return '、'.join(f'向{pair[0] if v>0 else pair[1]}{abs(v):g}mm' for v,pair in zip(to_display(offset),names) if abs(v)>1e-9) or '无位置补偿'
