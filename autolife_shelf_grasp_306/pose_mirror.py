"""从URDF关节轴推导双臂镜像符号；仅生成目标与检查路径，无硬件连接。"""
import math
import numpy as np
from autolife_handeye_calibration_306.core import ARMS,TORSO
from .pose_workspace_data import checked_targets

def mirror_signs(geometry):
    f=geometry.frames({});r=f[TORSO][:3,:3].T;S=np.diag([1.,-1.,1.])
    joints={j['name']:j for j in geometry.joints};signs=[]
    root_inverse=np.linalg.inv(f[TORSO])
    for left,right in zip(ARMS['left'],ARMS['right']):
        a,b=joints[left],joints[right]
        x=r@f[a['parent']][:3,:3]@a['origin'][:3,:3]@a['axis'];x/=np.linalg.norm(x)
        y=r@f[b['parent']][:3,:3]@b['origin'][:3,:3]@b['axis'];y/=np.linalg.norm(y)
        # 旋转轴是轴向矢量，镜像后需要乘det(S)=-1，不能直接按Y取反。
        alignment=float(np.dot(-S@x,y))
        offset=np.linalg.norm((root_inverse@f[b['child']])[:3,3]-S@(root_inverse@f[a['child']])[:3,3])
        if abs(alignment)<.99999 or offset>.002:
            raise ValueError('本机左右臂安装关系不满足关节镜像：'+left)
        signs.append(1. if alignment>0 else -1.)
    return signs

def mirror_targets(geometry,source,target_side):
    if target_side not in ARMS:raise ValueError('镜像目标侧无效')
    origin='left' if target_side=='right' else 'right'
    return checked_targets({dst:float(source[src])*sign for src,dst,sign in
        zip(ARMS[origin],ARMS[target_side],mirror_signs(geometry))})

def calculate(solver,source):
    targets=mirror_targets(solver.geometry,source,solver.cfg['arm'])
    count=max(1,math.ceil(max(abs(targets[n]-source[n]) for n in targets)/.5))
    solver.planner.guard.begin(source)
    for t in np.linspace(0,1,count+1):
        solver.planner.guard.check({**source,**{n:source[n]+t*(v-source[n]) for n,v in targets.items()}})
    end={**source,**targets}
    return dict(source=source,joints=targets,target=solver.pose(end).tolist(),frame=solver.geometry.root,
                checked_states=count+1,fixed_collision_notes=list(solver.planner.guard.fixed_pairs))
