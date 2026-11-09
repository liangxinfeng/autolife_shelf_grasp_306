"""末端示教只移动所选七轴：检查运动连杆与全身，不由固定部件互碰阻断。

原厂URDF与VR碰撞模型的身体安装变换须一致。夹爪采用已有简化几何，
不把DM电机位置量当作手指角，也不附加手眼标定板的虚拟球。
仅用于姿态页单臂IK；抓取、全身复位的碰撞检查保持不变。
"""
import numpy as np
import pinocchio as pin

def affected_joint(model,joint,controlled):
    while joint:
        if joint in controlled:return True
        joint=int(model.parents[joint])
    return False

class ArmCollisionScope:
    def __init__(self,guard,names,geometry):
        self.kin=guard.kin;self.names=tuple(names);self.geometry=geometry
        self.controlled={int(self.kin.model.getJointId(n)) for n in names}
        if 0 in self.controlled:raise ValueError('碰撞模型缺少所选手臂关节')
        self.moving=[affected_joint(self.kin.model,int(g.parentJoint),self.controlled)
                     for g in self.kin.geom_model.geometryObjects]
        if not any(self.moving):raise ValueError('碰撞模型未包含所选手臂几何')
        self.cache=set();self.source=None;self.fixed_pairs=[]
    def begin(self,source):
        self.source=dict(source);self.cache.clear();self.fixed_pairs=[]
        raw=self.configuration(source)
        # 核对模型的21轴连杆位置与朝向，防止跨型号/旧安装位姿产生假碰撞。
        frames=self.geometry.frames(source)
        pin.forwardKinematics(self.kin.model,self.kin.data,raw)
        pin.updateFramePlacements(self.kin.model,self.kin.data)
        for joint in self.geometry.joints:
            if joint['name'] not in self.kin.joint_q:continue
            name=joint['child']
            if not self.kin.model.existFrame(name):raise ValueError('碰撞模型缺少连杆：'+name)
            pose=self.kin.data.oMf[self.kin.model.getFrameId(name)].homogeneous
            if not np.allclose(pose,frames[name],atol=1e-6):
                raise ValueError('显示与碰撞模型的安装关系不一致：'+name)
    def configuration(self,values):
        from autolife_handeye_calibration_306.auto_plan_worker import GROUPS
        if self.source is None:raise ValueError('未绑定本次末端示教参考姿态')
        if any(n not in self.names and abs(values.get(n,float('inf'))-v)>1e-9 for n,v in self.source.items()):
            raise ValueError('末端示教改变了腰腿、头部、夹爪或另一侧手臂')
        q=self.kin.q_from_feedback({g:[values[n] for n in names] for g,names in GROUPS.items()},clip=False)
        if not np.all(np.isfinite(q)) or np.any(q<self.kin.lower-1e-6) or np.any(q>self.kin.upper+1e-6):
            raise ValueError('姿态超出URDF限位')
        return q
    def check(self,values):
        q=self.configuration(values);key=tuple(q)
        if key in self.cache:return
        pin.computeCollisions(self.kin.model,self.kin.data,self.kin.geom_model,self.kin.geom_data,q,False)
        blocking=[];fixed=[]
        for index,pair in enumerate(self.kin.geom_model.collisionPairs):
            if not self.kin.geom_data.collisionResults[index].isCollision():continue
            objects=self.kin.geom_model.geometryObjects
            label=objects[pair.first].name+' <-> '+objects[pair.second].name
            if self.moving[pair.first] or self.moving[pair.second]:blocking.append(label)
            else:fixed.append(label)
        self.fixed_pairs=sorted(set(self.fixed_pairs+fixed))
        if blocking:raise ValueError('所选手臂路径自碰撞：'+'；'.join(blocking[:3]))
        self.cache.add(key)
