"""保存姿态复位的模型兼容：不使用颈部小角度阈值，不拆分抬头恢复。

只适用于已确认几何误差的345模型和两对颈胸部件。头壳初始重叠必须
沿已检查路径退出，不得比起点更深、消失后再出现或残留到终点。
其他部件的碰撞和URDF关节限位仍由完整模型检查。
"""
import hashlib
import math
from contextlib import contextmanager
from pathlib import Path

MODEL_SHA256='118ba602bd1b0a0964d6d149d095f0825231613218faf7b38f08b35742e72e17'
BOOT_HEAD_PAIR=frozenset(('Link_Waist_Yaw_to_Shoulder_Inner_0','Link_Neck_Yaw_to_Head_0'))

@contextmanager
def reset_path_scope(guard,source,goal):
    from .grasp_collision import STATIC_NECK_PAIR,GROUPS
    original=guard.blocking_pairs;notes=[];requests=[];cleared=[False]
    kin=guard.kin;head_index=None;initial_depth=None
    enabled=hashlib.sha256(Path(kin.urdf_path).read_bytes()).hexdigest()==MODEL_SHA256
    def model_pairs(values):
        q=kin.q_from_feedback({g:[values[n] for n in names] for g,names in GROUPS.items()},clip=False)
        return {frozenset(p.split(' <-> ')) for p in kin.collision_pairs(q)}
    def depth():
        if head_index is None:return None
        result=kin.geom_data.collisionResults[head_index]
        if not result.isCollision():return 0.
        contacts=result.getContacts()
        if not contacts:return None
        values=[abs(float(c.penetration_depth)) for c in contacts]
        return max(values) if all(math.isfinite(v) for v in values) else None
    try:
        if enabled:
            for index,pair in enumerate(kin.geom_model.collisionPairs):
                names=frozenset((kin.geom_model.geometryObjects[pair.first].name,kin.geom_model.geometryObjects[pair.second].name))
                if names==BOOT_HEAD_PAIR:
                    head_index=index;request=kin.geom_data.collisionRequests[index]
                    requests.append((request,request.enable_contact,request.num_max_contacts))
                    request.enable_contact=True;request.num_max_contacts=64
            final=model_pairs(goal)
            enabled=not bool(final-{STATIC_NECK_PAIR})
            initial=model_pairs(source)
            enabled=enabled and not bool(initial-{STATIC_NECK_PAIR,BOOT_HEAD_PAIR})
            if BOOT_HEAD_PAIR in initial:initial_depth=depth()
            else:cleared[0]=True
        def blocking(pairs,values):
            remaining=original(pairs,values)
            if not enabled:return remaining
            inside=all(min(source[n],goal[n])-1e-6<=values[n]<=max(source[n],goal[n])+1e-6 for n in source)
            if not inside:return remaining
            keys={frozenset(p.split(' <-> ')) for p in remaining}
            if BOOT_HEAD_PAIR not in keys:cleared[0]=True
            current_depth=depth()
            at_goal=all(abs(values[n]-goal[n])<=1e-6 for n in goal)
            allow_head=(initial_depth is not None and not cleared[0] and not at_goal and
                        current_depth is not None and current_depth<=initial_depth+1e-6)
            result=[]
            for pair in remaining:
                key=frozenset(pair.split(' <-> '))
                if key==STATIC_NECK_PAIR or key==BOOT_HEAD_PAIR and allow_head:
                    if pair not in notes:notes.append(pair)
                else:result.append(pair)
            return result
        guard.cache.clear();guard.blocking_pairs=blocking
        yield notes
    finally:
        guard.blocking_pairs=original;guard.cache.clear()
        for request,contact,count in requests:
            request.enable_contact=contact;request.num_max_contacts=count
