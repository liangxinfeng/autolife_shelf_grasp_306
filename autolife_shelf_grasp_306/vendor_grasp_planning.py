"""原厂 KDL 逆解：固定膝踝，腰部和所选手臂共同求解；本文件没有运动输出。

保留现有物品几何与整条路径预检。抓点绑定规划瞬间的躯干坐标，
再转换到固定底座系，不能在弯腰后把目标重新绑定到移动的躯干。
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from .low_object_grasp import grasp_scope
import pinocchio as pin
from scipy.spatial.transform import Rotation
from autolife_handeye_calibration_306.core import ARMS, BODY, NECK, TORSO, Kinematics
from autolife_handeye_calibration_306.verification_core import wrist_frame
from .planning import Planner, IKFailure
from .carry import carry_goal
from .grasp_collision import CollisionGuard


BASE = 'Link_Zero_Point'


class WaistAwareCollisionGuard(CollisionGuard):
    """腰部共同转动不改变颈胸相对位置；只沿用已登记的同一静态重叠。"""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.relative_fk = Kinematics(cfg['urdf_path'])

    def begin_fixed_neck(self, source, goal=None):
        super().begin_fixed_neck(source, goal)
        self.reference_relative = self.relative_fk.fk(TORSO, 'Link_Neck_Pitch_to_Neck_Yaw', source)
        self._neck_key=tuple(source[n] for n in NECK);self._neck_relative=self.reference_relative

    def blocking_pairs(self, pairs, values):
        source = getattr(self, 'fixed_neck_source', None)
        if source is None or any(abs(values[n]-source[n]) > .05 for n in NECK):
            return pairs
        # 躯干到颈部链仅依赖颈部关节；颈部数值不变时复用完全相同的FK。
        key=tuple(values[n] for n in NECK)
        if key!=self._neck_key:
            self._neck_relative=self.relative_fk.fk(TORSO, 'Link_Neck_Pitch_to_Neck_Yaw', values);self._neck_key=key
        relative=self._neck_relative
        if not np.allclose(relative, self.reference_relative, atol=1e-8):
            return pairs
        return [p for p in pairs if p not in self.static_neck_pairs]


def vendor_imports(cfg):
    """使用本机已安装的原厂库；不拷贝二进制、不创建原厂硬件控制实例。"""
    for path in (cfg['vendor_python_path'], '/usr/lib/python3/dist-packages'):
        if path not in sys.path:
            sys.path.append(path)


class VendorGraspPlanner(Planner):
    guard_factory=WaistAwareCollisionGuard
    def __init__(self, cfg, context=None):
        from .planning_control import PlanningControl
        self.context=context or PlanningControl()
        self.context.set_stage("加载原厂IK",arm=cfg["arm"])
        super().__init__(cfg)
        # 父类直接创建本类碰撞检查器，避免同一请求重复加载两套网格。
        self.arm_names = list(self.names)
        self.names = BODY[2:] + self.arm_names
        self.arm_bias_index = self.names.index(self.arm_names[1])
        self.anchor = None
        self.solver = None
        self.full_pose_indices=[(str(name),self.pose_model.joints[i].idx_q)
                                for i,name in enumerate(self.pose_model.names)
                                if i and self.pose_model.joints[i].nq==1]

    def close(self):
        if self.solver is not None:self.solver.close();self.solver=None

    def candidates(self, yaws=None):
        # 原厂IK精确解完整位姿，不重复尝试同一求解的“软朝向”分支。
        # 先求预抓端点并预检关节空间过渡，再考虑直线接近。
        if yaws is None:yaws=[0.,-10.,10.,-20.,20.,-35.,35.,-50.,50.,-60.,60.]
        yaws=list(yaws);self.total_candidates=len(yaws)*5;self.current_candidate=0
        for yaw in yaws:
            for profile in ('joint_direct','joint_elbow_positive','joint_elbow_negative','direct','outward_8'):
                self.current_candidate+=1
                self.context.set_stage('候选路径',arm=self.cfg['arm'],
                    candidate=self.current_candidate,total_candidates=self.total_candidates)
                yield yaw,False,profile

    def carry_target(self, source):
        goal = carry_goal(self.cfg, source, self.arm_names)
        # 收回到准备时的腰角，避免后续放置仍按旧身体坐标执行。
        goal.update({n: self.source[n] for n in BODY})
        return goal

    def _carry_segment(self, source, tcp, product, grasp_rotation, started, clearance_waypoints=None):
        """先保持瓶身朝向抬离并退出物品原占地，再连续衔接收回姿态。

        不能从抓点直接做大范围关节插值：即使终点较高，中间瓶底仍可能下探。
        这里的采样只是离线几何预检，执行不在采样点停止或等待反馈。
        """
        from .bottle import horizontal_basis
        from .planning import CollisionFailure
        self.context.set_stage('持物收回',arm=self.cfg['arm'])
        try:
            return super()._carry_segment(source,tcp,product,grasp_rotation,started,clearance_waypoints)
        except (IKFailure,CollisionFailure):
            pass  # 只有已检查的直接路径不通过，才计算明确的抬离退离路径。
        up=np.asarray(self.cfg['up_torso'],float)
        start,rotation=self.pose(source,tcp)
        tool=Rotation.from_quat(self.cfg['T_wrist_tool_quat_xyzw']).as_matrix()
        forward=(rotation@tool)[:,0]
        forward=forward-up*np.dot(forward,up);forward/=np.linalg.norm(forward)
        last_error=None
        for lift,retreat in ((max(.07,float(self.cfg['lift_m'])),.12),(.10,.18)):
            previous=dict(source);points=[previous]
            try:
                for target in (start+up*lift,start+up*lift-forward*retreat):
                    begin,_=self.pose(previous,tcp)
                    count=max(2,int(np.ceil(np.linalg.norm(target-begin)/.008))+1)
                    for t in np.linspace(0,1,count)[1:]:
                        previous=self.solve(previous,tcp,begin+t*(target-begin),rotation)
                        self.check(previous,tcp,True,product)
                        points.append(previous)
                segment,count=self._checked_carry_segment(source,tcp,product,grasp_rotation,started,points)
                segment['name']='连续抬离并恢复已保存手臂姿态'
                segment['return_strategy']='clear_then_saved_arm'
                return segment,count
            except (IKFailure,CollisionFailure) as exc:
                last_error=exc
        raise last_error

    def pose(self, q, tcp):
        # 同一完整URDF的编译FK；每次更新全身反馈，不能遗漏腰腿或另一只手。
        for name,index in self.full_pose_indices:
            self.pose_q[index]=np.deg2rad(q.get(name,0.))
        pin.forwardKinematics(self.pose_model,self.pose_data,self.pose_q)
        pin.updateFramePlacements(self.pose_model,self.pose_data)
        wrist=self.pose_data.oMf[self.wrist_frame]
        if self.anchor is None:
            torso=self.pose_data.oMf[self.torso_frame]
            rotation=torso.rotation.T@wrist.rotation
            position=torso.rotation.T@(wrist.translation-torso.translation)
        else:
            rotation=self.anchor[:3,:3].T@wrist.rotation
            position=self.anchor[:3,:3].T@(wrist.translation-self.anchor[:3,3])
        return position+rotation@tcp,rotation.copy()

    def configure_source(self, source):
        self.source = dict(source)
        self.anchor = self.fk.fk(BASE, TORSO, source)
        tree = ET.parse(self.cfg['urdf_path']).getroot()
        vendor_config = Path(self.cfg['vendor_control_config']) if self.cfg.get('vendor_control_config') else (
            Path(self.cfg['vendor_python_path'])/'autolife_robot_arm/configs/robot_v3_1.json')
        dead = json.loads(vendor_config.read_text())['MOTOR_DEAD_ZONE']
        waist_bounds = []
        for i, name in enumerate(BODY[2:], 2):
            limit = tree.find(f"joint[@name='{name}']/limit")
            reach = float(self.cfg.get('vendor_grasp_waist_range_deg', [12., 8.])[i-2])
            if not 0 < reach <= 20:
                raise ValueError('抓取腰部参与范围必须在0～20度内')
            lo = max(np.rad2deg(float(limit.get('lower'))) +
                     dead['LEG_WAIST_LOWER_DEAD_ZONE'][i] + .5, source[name]-reach)
            hi = min(np.rad2deg(float(limit.get('upper'))) -
                     dead['LEG_WAIST_UPPER_DEAD_ZONE'][i] - .5, source[name]+reach)
            if not lo <= source[name] <= hi:
                raise ValueError('当前腰部位于原厂限位保护区，请先调整姿态')
            waist_bounds.append([lo, hi])
        self.bounds = np.deg2rad(np.array(waist_bounds + [
            self.joint_limit_policy['bounds_deg'][n] for n in self.arm_names]).T)
        from .vendor_ik_process import VendorIKProcess
        locked={n:source[n] for n in BODY[:2]}
        if self.solver is None:
            self.solver = VendorIKProcess(self.cfg,self.cfg['urdf_path'],BASE,wrist_frame(self.cfg),
                                         locked,checkpoint=self.context.checkpoint)
        else:
            self.solver.checkpoint=self.context.checkpoint
            try:self.solver.request('configure',locked=locked)
            except BaseException:self.close();raise
        try:
            # 先核对原厂链的顺序和坐标，不能把错误次序送给真机。
            q0 = np.deg2rad([source[n] for n in self.names])
            actual = np.asarray(self.solver.request('fk',q=q0.tolist()))
            expected = self.fk.fk(BASE, wrist_frame(self.cfg), source)
            if (np.linalg.norm(actual[:3]-expected[:3, 3]) > 1e-6 or
                    Rotation.from_matrix(expected[:3, :3].T @
                        Rotation.from_quat(actual[3:]).as_matrix()).magnitude() > 1e-6):
                raise ValueError('原厂IK链顺序或模型坐标不一致，未生成轨迹')
        except BaseException:
            self.close();raise

    @grasp_scope
    def plan(self, source, tcp, center, product):
        self.configure_source(source)
        try:
            plan = super().plan(source, tcp, center, product)
        except BaseException:
            self.close();raise
        # 与原有持物、界面协议兼容：这里只登记7轴复位臂形。
        if 'carry_pose_deg' in plan:
            plan['carry_pose_deg'] = [self.carry_target(source)[n] for n in self.arm_names]
        plan.update(execution_backend='vendor_trajectory_v1',
                    ik_backend='autolife_robot_sdk.KdlSolver.nr_jl',
                    controlled_joints=list(self.names),
                    source_torso_to_base=self.anchor.tolist(),
                    waist_bounds_deg={n: list(np.rad2deg(self.bounds[:, i]))
                                      for i, n in enumerate(BODY[2:])})
        from .vendor_grasp_trajectory import build_blocks
        self.context.set_stage('原厂轨迹定时与复核',arm=self.cfg['arm'])
        plan['vendor_blocks'] = build_blocks(plan, self.cfg)
        product=plan['product']
        # 真机执行的是原厂线性插值；必须检查它，而非只检查旧PCHIP曲线。
        checks = 0
        grasp_q=next(s for s in plan['segments'] if s['phase']=='grasp')['waypoints'][-1]
        grasp_position,grasp_rotation=self.pose(grasp_q,np.asarray(plan['tcp']))
        end_position,_=self.pose(plan['segments'][-1]['waypoints'][-1],np.asarray(plan['tcp']))
        up=np.asarray(self.cfg['up_torso'],float)
        def bottom(position,rotation):
            if 'bbox_axes_torso' in product:
                from .object_extent import bbox
                axes,extent=bbox(product)
                half=float(np.abs(up@(rotation@grasp_rotation.T@axes))@extent/2)
            else:
                axis=rotation@grasp_rotation.T@up;cosine=float(np.clip(abs(axis@up),0,1))
                half=float(product['height_m'])*cosine/2+float(product['diameter_m'])/2*np.sqrt(max(0.,1-cosine*cosine))
            return float(position@up)-half
        start_bottom=bottom(grasp_position,grasp_rotation)
        for block in plan['vendor_blocks']:
            for interval,(a,b) in enumerate(zip(block['waypoints'],block['waypoints'][1:])):
                count = max(2, int(np.ceil(max(abs(b[n]-a[n]) for n in self.names)/.5))+1)
                # 相邻区间共享端点只检查一次，间隔仍不超过原来的0.5度。
                for t in np.linspace(0,1,count)[1 if interval else 0:]:
                    q={**source,**{n:(b[n] if t==1 else a[n] if t==0 else a[n]+t*(b[n]-a[n])) for n in self.names}}
                    self.check(q, np.asarray(plan['tcp']), block['carrying'], product)
                    if block['carrying']:
                        position,rotation=self.pose(q,np.asarray(plan['tcp']))
                        delta=position-grasp_position
                        horizontal=delta-up*float(delta@up)
                        if float(position@up)<min(float(grasp_position@up),float(end_position@up))-.005:
                            raise IKFailure('原厂插值持物路径向下绕行')
                        if np.linalg.norm(horizontal)<.10 and (
                                float(delta@up)<-.002 or bottom(position,rotation)<start_bottom-.002):
                            raise IKFailure('原厂插值在起瓶范围内使物品底部下降')
                        tilt=np.rad2deg(np.arccos(np.clip(np.dot(rotation.T@up,grasp_rotation.T@up),-1.,1.)))
                        if tilt>float(self.cfg.get('carry_max_tilt_deg',45.)):
                            raise IKFailure('原厂插值持物倾角超出配置')
                    checks += 1
        plan['native_checked_states'] = checks
        return plan

    def check(self,q,tcp,carrying,product):
        self.context.checkpoint()
        return super().check(q,tcp,carrying,product)

    def solve(self, source, tcp, point, rotation, flexible=False, limit_deg=None,
              posture_target=None, global_goal=False, retreat_freedom=0., free_heading=False):
        """所有关节逆解交给原厂KDL；独立FK复核，不凭求解返回值判定到点。"""
        phase={0:'预抓取',1:'进入抓点',2:'抬离',3:'退离',4:'持物收回'}.get(getattr(self,'_candidate_stage',0),'逆解')
        self.context.set_stage(phase,arm=self.cfg['arm'])
        target = np.eye(4)
        target[:3, :3] = rotation
        target[:3, 3] = np.asarray(point)-rotation@tcp
        target = self.anchor@target
        pose = np.r_[target[:3, 3], Rotation.from_matrix(target[:3, :3]).as_quat()]
        chain_names = self.names
        seed = np.deg2rad([source[n] for n in chain_names])
        lo, hi = self.bounds
        if np.any(seed < lo-1e-7) or np.any(seed > hi+1e-7):
            raise IKFailure('原厂IK初值超出本次腰臂范围')
        try:
            result = self.solver.request('ik',pose=pose.tolist(),seed=seed.tolist(),
                lower=lo.tolist(),upper=hi.tolist())
            values = np.asarray(result, float)
        except (ValueError,OSError) as exc:
            raise IKFailure('原厂IK未收敛：'+str(exc)) from exc
        if values.shape != seed.shape or not np.isfinite(values).all():
            raise IKFailure('原厂IK未返回完整腰臂关节')
        if np.any(values < lo-1e-6) or np.any(values > hi+1e-6):
            raise IKFailure('原厂IK结果超限')
        q = {**source, **dict(zip(chain_names, np.rad2deg(values).tolist()))}
        position, actual_rotation = self.pose(q, tcp)
        error = np.linalg.norm(position-point)
        angle = Rotation.from_matrix(rotation.T@actual_rotation).magnitude()
        if error > .003 or angle > np.deg2rad(2.):
            raise IKFailure(f'原厂IK位置残差{error*1000:.1f}mm，方向残差{np.rad2deg(angle):.1f}°')
        return q
