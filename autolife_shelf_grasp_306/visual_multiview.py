"""补录外观与初始几何锚分离；移位不重建物体编号或动作参照。"""
from copy import deepcopy
import numpy as np
from .visual_reference import matrix
from .visual_registration import register_views,validate_surface


def prepare_view(original,views,meta,arrays):
    meta=deepcopy(meta);arrays={k:np.asarray(v).copy() for k,v in arrays.items()}
    # 原始拍摄帧坐标留档，不冒充初始参考坐标。
    meta.setdefault('captured_object_pose',meta['object_pose']);meta.setdefault('captured_camera_pose',meta['camera_pose'])
    meta.update(name=original['name'],object_pose=original['object_pose'],metric_registered=False)
    canonical=[v for v in views if v[0].get('metric_registered',True)]
    try:
        delta,stats,_,matched=register_views(canonical,arrays)
        validate_surface(matched['cloud'],arrays['cloud'],delta)
        if stats['angle_resolution_deg']>5:raise ValueError('共同表面不足以确定初始几何关系')
    except ValueError as exc:
        meta['registration_note']=str(exc)
        return meta,arrays
    inv=np.linalg.inv(delta)
    arrays['points']=arrays['points']@inv[:3,:3].T+inv[:3,3]
    arrays['cloud']=arrays['cloud']@inv[:3,:3].T+inv[:3,3]
    meta.update(metric_registered=True,camera_pose=(inv@matrix(meta['camera_pose'])).tolist(),
                reference_from_capture=inv.tolist(),registration_note='已关联初始几何参照')
    return meta,arrays


def seed_match(banks,current):
    """外观独立视角可定位像素；它的坐标绝不当作最初坐标输出纠偏。"""
    canonical=[v for v in banks if v[0].get('metric_registered',True)]
    hits=[];errors=[]
    try:hits.append(register_views(canonical,current))
    except ValueError as exc:errors.append(str(exc))
    from .visual_registration import register
    for meta,arrays in banks:
        if meta.get('metric_registered',True):continue
        try:
            delta,stats=register(arrays,current);hits.append((delta,stats,meta,arrays))
        except ValueError as exc:errors.append(str(exc))
    if not hits:raise ValueError('暂未认出已记录物体，请让已补录的一面进入画面，或框选补录当前外观。'+(errors[0] if errors else ''))
    # 补录独立参与识别，不能被初始帧几个勉强匹配的标签点抢占。
    hits.sort(key=lambda r:(-r[1]['inliers'],r[1]['rms_mm']))
    return hits[0]



def resolve_views(views):
    """补录过渡面后，在内存关联早先存下的独立视角，不改原始快照。"""
    canonical=[v for v in views if v[0].get('metric_registered',True)]
    pending=[v for v in views if not v[0].get('metric_registered',True)]
    original=views[0][0]
    for _ in range(len(pending)):
        rest=[]
        for meta,arrays in pending:
            m,a=prepare_view(original,canonical,meta,arrays)
            if m['metric_registered']:canonical.append((m,a))
            else:rest.append((meta,arrays))
        if len(rest)==len(pending):break
        pending=rest
        if not pending:break
    return canonical+pending
