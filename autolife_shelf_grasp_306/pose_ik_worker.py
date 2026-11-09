"""由300末端示教的固定非受控关节逻辑改编；复用345现有IK/碰撞模型，只计算。"""
import json,sys,math
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .planning import Planner,TORSO
from .pose_geometry import ModelGeometry
from .teaching_motion import checked_joints


class PoseSolver:
    def __init__(self,cfg):
        self.cfg=cfg;self.planner=Planner(cfg);self.geometry=ModelGeometry(cfg['urdf_path'])
        self.tcp=self.geometry.tool_transform(cfg['arm']);self.names=self.planner.names
        from .pose_collision_scope import ArmCollisionScope
        self.planner.guard=ArmCollisionScope(self.planner.guard,self.names,self.geometry)
    def pose(self,q):return self.geometry.frames(q)['Link_'+self.cfg['arm'].title()+'_Wrist_Lower_to_Gripper']@self.tcp
    def calculate(self,req):
        source=checked_joints(req['source']);actual=self.pose(source)
        self.planner.guard.begin(source)
        if req['operation']=='capture':return dict(source=source,target=actual.tolist(),frame=self.geometry.root)
        if req['operation']=='mirror':
            from .pose_mirror import calculate
            return calculate(self,source)
        self.planner.guard.check(source)
        target=np.array(req['target'],float)
        if target.shape!=(4,4) or not np.isfinite(target).all() or not np.allclose(target[3],[0,0,0,1]):raise ValueError('末端矩阵格式无效')
        if not np.allclose(target[:3,:3].T@target[:3,:3],np.eye(3),atol=1e-6) or abs(np.linalg.det(target[:3,:3])-1)>1e-6:raise ValueError('末端朝向无效')
        seed={**source,**{n:req.get('seed',source).get(n,source[n]) for n in self.names}}
        initial=self.pose(seed);inverse=np.linalg.inv(self.geometry.frames(source)[TORSO])
        count=max(1,math.ceil(np.linalg.norm(target[:3,3]-initial[:3,3])/.01),math.ceil(Rotation.from_matrix(initial[:3,:3].T@target[:3,:3]).magnitude()/math.radians(3)))
        if count>150:raise ValueError('目标过远，请减小调节距离')
        orientations=Slerp([0,1],Rotation.from_matrix([initial[:3,:3],target[:3,:3]]));q=seed
        for t in np.linspace(0,1,count+1)[1:]:
            position=inverse[:3,:3]@(initial[:3,3]*(1-t)+target[:3,3]*t)+inverse[:3,3]
            rotation=inverse[:3,:3]@orientations(t).as_matrix()@self.tcp[:3,:3].T
            q=self.planner.solve(q,self.tcp[:3,3],position,rotation,limit_deg=1.)
        if any(q[n]!=v for n,v in source.items() if n not in self.names):raise ValueError('IK改变了非所选手臂关节')
        end=self.pose(q);pos=float(np.linalg.norm(end[:3,3]-target[:3,3])*1000)
        angle=float(np.degrees(Rotation.from_matrix(end[:3,:3].T@target[:3,:3]).magnitude()))
        if pos>3 or angle>1:raise ValueError(f'目标不可达：位置误差{pos:.1f}mm，朝向误差{angle:.2f}°')
        count=max(1,math.ceil(max(abs(q[n]-source[n]) for n in self.names)/.5))
        for t in np.linspace(0,1,count+1):self.planner.guard.check({**source,**{n:source[n]+t*(q[n]-source[n]) for n in self.names}})
        return dict(source=source,joints={n:q[n] for n in self.names},target=target.tolist(),frame=self.geometry.root,position_error_mm=pos,angle_error_deg=angle,checked_states=count+1,fixed_collision_notes=list(self.planner.guard.fixed_pairs))


def main():
    solvers={}
    for line in sys.stdin:
        try:
            req=json.loads(line);key=json.dumps(req['config'],sort_keys=True)
            if key not in solvers:solvers[key]=PoseSolver(req['config'])
            reply=dict(result=solvers[key].calculate(req))
        except Exception as exc:reply=dict(error=str(exc))
        print('POSE_IK='+json.dumps(reply,allow_nan=False),flush=True)
if __name__=='__main__':main()
