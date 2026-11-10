"""放置复用抓取的原厂腰臂逆解；所有计算固定在观察时的参考系，无运动输出。"""
import numpy as np
from scipy.spatial.transform import Rotation
from autolife_handeye_calibration_306.core import BODY
from autolife_handeye_calibration_306.core import TORSO
from autolife_handeye_calibration_306.verification_core import wrist_frame
from .vendor_grasp_planning import VendorGraspPlanner,BASE


class VendorPlacePlanner(VendorGraspPlanner):
    placement_whole_body_check=True

    def placement_grasp_rotation(self,grasp):
        # 抓取时腰部可能前倾，不能把旧躯干内的朝向直接用在当前躯干系。
        q=next(s for s in grasp['segments'] if s['phase']=='grasp')['waypoints'][-1]
        return self.anchor[:3,:3].T@self.fk.fk(BASE,wrist_frame(self.cfg),q)[:3,:3]

    def plan_place(self,source,tcp,grasp,points,table,settings):
        from .demo_planning import plan_place
        self.configure_source(source)
        self.cfg['up_torso']=(self.anchor[:3,:3].T@np.array([0.,0.,1.])).tolist()
        return plan_place(self,source,tcp,grasp,points,table,settings)

    def finalize_place(self,plan,source,tcp,grasp,table,settings):
        """候选返回前完成真正执行轨迹的密集检查；失败交回候选循环继续搜索。"""
        from .demo_planning import TableGuard,check_release_departure
        from .demo_config import carried_height
        from .vendor_place_trajectory import build_blocks
        plan.update(placement_speed_policy='grasp_transfer_v2',place_protocol_version=4,execution_backend='vendor_place_trajectory_v1',
            ik_backend='autolife_robot_sdk.KdlSolver.nr_jl',controlled_joints=list(self.names),
            source_torso_to_base=self.anchor.tolist(),
            waist_bounds_deg={n:list(np.rad2deg(self.bounds[:,i])) for i,n in enumerate(BODY[2:])})
        self.context.set_stage('放置整段定时与模型复核',arm=self.cfg['arm'])
        plan['vendor_blocks']=build_blocks(plan,self.cfg)
        product=plan['product'];up=np.asarray(self.cfg['up_torso']);tcp=np.asarray(tcp)
        reference=self.placement_grasp_rotation(grasp)
        guard=TableGuard(self,table,product,tcp,reference)
        landing=plan['segments'][1]['waypoints'][-1]
        release_start,release_rotation=self.pose(landing,tcp)
        tool=release_rotation@Rotation.from_quat(self.cfg['T_wrist_tool_quat_xyzw']).as_matrix()
        checks=0
        checked_states=set()
        # 定时重采样后的原厂线性插值另行检查，不能用旧曲线检查替代。
        for block in plan['vendor_blocks']:
            for a,b in zip(block['waypoints'],block['waypoints'][1:]):
                count=max(2,int(np.ceil(max(abs(b[n]-a[n]) for n in self.names)/.1))+1)
                for t in np.linspace(0,1,count):
                    q=(a if t==0 else b if t==1 else
                       {**source,**{n:a[n]+t*(b[n]-a[n]) for n in self.names}})
                    key=(block['carrying'],tuple(sorted(q.items())))
                    if key in checked_states:continue
                    self.check(q,tcp,block['carrying'],product);guard.check(q,block['carrying'])
                    pos,rotation=self.pose(q,tcp)
                    if block['carrying']:
                        tilt=np.rad2deg(np.arccos(np.clip(np.dot(rotation.T@up,reference.T@up),-1.,1.)))
                        if tilt>45:raise ValueError('放置原厂插值导致持物倾斜过大')
                    else:
                        check_release_departure(pos-release_start,rotation,release_rotation,tool,up,
                            settings['release_retreat_m'],carried_height(product))
                    checks+=1
                    checked_states.add(key)
            if block['release_after']:guard.check_support(block['waypoints'][-1])
        plan['native_checked_states']=checks
        return plan
