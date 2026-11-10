"""批量计算持物球到连杆包围盒的相同距离下界，不改变精确碰撞条件。"""
import numpy as np


def sphere_box_candidates(center,radius,rotations,translations,lows,highs):
    local=np.einsum('nji,nj->ni',rotations,np.asarray(center)-translations)
    distance=np.linalg.norm(local-np.clip(local,lows,highs),axis=1)
    # 数值舍入边界保守保留，随后仍使用原来的FCL精确距离检查。
    return np.flatnonzero(distance<=radius+1e-12)


def nearby_geometries(guard,center,radius):
    if not hasattr(guard,'_held_box_indices'):
        indices=[i for i,obj in enumerate(guard.kin.geom_model.geometryObjects)
                 if not any(obj.name.startswith(prefix) for prefix in guard.skip)]
        guard._held_box_indices=np.asarray(indices,int)
        guard._held_box_lows=np.asarray([guard.boxes[i][0] for i in indices])
        guard._held_box_highs=np.asarray([guard.boxes[i][1] for i in indices])
    indices=guard._held_box_indices
    if not len(indices):return indices
    placements=[guard.kin.geom_data.oMg[int(i)] for i in indices]
    selected=sphere_box_candidates(center,radius,
        np.asarray([t.rotation for t in placements]),np.asarray([t.translation for t in placements]),
        guard._held_box_lows,guard._held_box_highs)
    return indices[selected].tolist()
