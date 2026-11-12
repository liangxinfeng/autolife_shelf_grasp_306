"""完整URDF手指与名义平行开合包络；不把DM位置量当作URDF旋转角。"""
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


@lru_cache(maxsize=4)
def _parts(path, stamp, side):
    import pinocchio as pin
    model=pin.buildModelFromUrdf(path)
    geom=pin.buildGeomFromUrdf(model,path,pin.GeometryType.COLLISION)
    data=model.createData();gd=pin.GeometryData(geom);q=pin.neutral(model)
    pin.framesForwardKinematics(model,data,q)
    pin.updateGeometryPlacements(model,data,geom,gd,q)
    frame='Link_'+side+'_Wrist_Lower_to_Gripper'
    frame_id=model.getFrameId(frame)
    if frame_id>=model.nframes:raise ValueError('完整URDF缺少夹爪安装坐标系')
    wrist=data.oMf[frame_id];parts=[]
    for i,obj in enumerate(geom.geometryObjects):
        if not (obj.name.startswith(frame+'_') or obj.name.startswith('Link_'+side+'_Gripper')):continue
        if not hasattr(obj.geometry,'vertices'):raise ValueError('完整夹爪碰撞几何必须提供可校验的网格')
        T=wrist.inverse()*gd.oMg[i]
        vertices=np.asarray(obj.geometry.vertices(),float)@T.rotation.T+T.translation
        parts.append((obj.name,vertices))
    fingers=[v for name,v in parts if '_Finger_' in name]
    if len(fingers)!=2:raise ValueError('完整URDF必须包含左右两根手指，不能只校验简化手掌')
    return tuple(parts)


def gripper_geometry(planner):
    cfg=planner.cfg;side=cfg['arm'].title();path=Path(cfg['urdf_path']).resolve()
    tool=Rotation.from_quat(cfg['T_wrist_tool_quat_xyzw']).as_matrix()
    width=float(cfg['gripper_max_width_m'])
    key=(str(path),path.stat().st_mtime_ns,side,tuple(tool.ravel()),width)
    cached=getattr(planner,'_complete_gripper_geometry',None)
    if cached is not None and cached[0]==key:return cached[1]
    if not 0<width<.2:raise ValueError('夹爪毫米开口范围无效')
    parts=_parts(*key[:3]);closing=tool[:,1];forward=tool[:,0]
    fingers=[v for name,v in parts if '_Finger_' in name]
    # 实测的是平行夹爪0～最大开口。手指姿态采用完整模型零位；
    # 沿开合轴覆盖两端，形成整个开合扫掠包络，不猜测DM量对应的角度。
    center=float(np.mean([np.mean(v@closing) for v in fingers]));rows=[]
    for name,v in parts:
        rows.append(v)
        if '_Finger_' not in name:continue
        projection=v@closing-center;sign=1 if np.mean(projection)>0 else -1
        inner=float(np.min(sign*projection))
        for opening in (0.,width):
            rows.append(v+(sign*(opening/2-inner))*closing)
    points=np.concatenate(rows)
    # 平面距离的极值必然位于凸包顶点；只简化计算，不削小实体范围。
    vertices=points[ConvexHull(points).vertices]
    depths=np.concatenate([v@forward for v in fingers])
    result=dict(vertices=vertices,finger_start=float(depths.min()),finger_tip=float(depths.max()),
                method='full_urdf_parallel_jaw_sweep',links=[n for n,_ in parts],max_width_m=width)
    planner._complete_gripper_geometry=(key,result)
    return result


def contact_tcp(planner,physical_tcp):
    """低物体使用指腹前段，原来的手指转轴原点不能充当指腹接触点。"""
    g=gripper_geometry(planner)
    fraction=float(planner.cfg.get('low_object_finger_contact_fraction',.80))
    if not .5<=fraction<=.9:raise ValueError('低物体指腹位置需在指长的50%～90%之间')
    forward=Rotation.from_quat(planner.cfg['T_wrist_tool_quat_xyzw']).as_matrix()[:,0]
    depth=g['finger_start']+fraction*(g['finger_tip']-g['finger_start'])
    tcp=np.asarray(physical_tcp,float)
    return tcp+forward*(depth-float(tcp@forward))
