"""Check a planned start pose without conflating jaw settling with arm motion."""
import math


def start_pose_diagnostic(current, source, stage, kind):
    if not source:
        raise ValueError(stage+'：缺少规划起始姿态')
    changes=[]
    for name, reference in source.items():
        if name not in current:
            raise ValueError(stage+'：缺少关节反馈 '+name)
        actual,reference=float(current[name]),float(reference)
        if not math.isfinite(actual) or not math.isfinite(reference):
            raise ValueError(stage+'：关节反馈或起始姿态无效 '+name)
        # Holding a bottle is not a free-space jaw-positioning operation.
        # Use the executor's existing 2-degree closure arrival tolerance only
        # for the loaded LEFT jaw. Do not rebase any checked arm trajectory.
        loaded_jaw=kind in ('place','observe') and name=='Joint_Left_Gripper'
        limit=2.0 if loaded_jaw else .5
        changes.append(dict(joint=name,reference_deg=reference,actual_deg=actual,
                            delta_deg=actual-reference,tolerance_deg=limit))
    violations=[c for c in changes if abs(c['delta_deg'])>c['tolerance_deg']]
    violations.sort(key=lambda c:abs(c['delta_deg'])/c['tolerance_deg'],reverse=True)
    message=''
    if violations:
        v=violations[0]
        label='持瓶夹爪开度改变' if v['joint']=='Joint_Left_Gripper' and kind in ('place','observe') else '姿态改变'
        message=(f"{stage}{label}：{v['joint']} 偏差{abs(v['delta_deg']):.2f}°"
                 f"（允许{v['tolerance_deg']:.2f}°），请检查并重新规划")
    return dict(stage=stage,kind=kind,ok=not violations,message=message,
                changes=changes,violations=violations)
