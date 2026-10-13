"""Arm ownership and provisional tool geometry; no hardware operations."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation


def arm_name(value='left'):
    if value not in ('left','right'):raise ValueError('无效抓取手臂：'+str(value))
    return value


def gripper_name(arm='left'):
    return 'Joint_'+arm_name(arm).title()+'_Gripper'


def plan_arm(plan):
    arm=arm_name(plan.get('config',{}).get('arm','left'))
    if plan.get('arm',arm)!=arm:raise ValueError('路径与手臂归属不一致')
    return arm


def require_same_arm(plan,record):
    arm=plan_arm(plan)
    if arm_name(record.get('arm','left'))!=arm:raise ValueError('持物记录与路径手臂不一致')
    return arm


def gripper_targets(arm,angle,current,both=False):
    """Gripper driver accepts both fields. Inactive jaw holds measured position."""
    selected=gripper_name(arm)
    return {gripper_name(s):float(angle if both or gripper_name(s)==selected else current[gripper_name(s)])
            for s in ('left','right')}


def tool_geometry(cfg,calibration,fk,arm):
    """Share camera extrinsics, NOT a torso-space mirror of the left arm.

    For identical factory grippers only, transfer the measured local TCP through
    their URDF mounting frames. This is explicitly provisional, not a measured
    right-hand calibration. Robot and URDF identity are still validated.
    """
    from autolife_handeye_calibration_306.verification_core import load_geometry,wrist_frame
    arm=arm_name(arm);source=arm_name(calibration.get('config',{}).get('arm','left'))
    tcp=np.asarray(load_geometry(calibration,fk,cfg['tcp_offset_mm'],robot_id=cfg.get('robot_id')),float)
    result=copy.deepcopy(cfg);result['arm']=arm
    provisional=arm!=source
    if provisional:
        zero={j[0]:0. for j in fk.joints.values()}
        def mount(side):return fk.fk(wrist_frame({'arm':side}),'Link_'+side.title()+'_Gripper',zero)
        transfer=mount(arm)@np.linalg.inv(mount(source))
        # This fallback is only for homologous factory grippers, not arbitrary tools.
        if (np.linalg.norm(transfer[:3,3])>.005 or
                Rotation.from_matrix(transfer[:3,:3]).magnitude()>np.deg2rad(3)):
            raise ValueError('两侧夹爪安装几何不一致，无法复用夹取中心')
        tcp=(transfer@np.r_[tcp,1.])[:3]
        tool=transfer[:3,:3]@Rotation.from_quat(cfg['T_wrist_tool_quat_xyzw']).as_matrix()
        result['T_wrist_tool_quat_xyzw']=Rotation.from_matrix(tool).as_quat().tolist()
        # The historical 15mm torso-Y trim was measured on the left hand only.
        # It is not a camera extrinsic and must not silently bias the right hand.
        result['target_offset_torso_mm']=list(cfg.get(arm+'_target_offset_torso_mm',[0.,0.,0.]))
    result['tool_geometry']={'arm':arm,'source_arm':source,'provisional':provisional,
        'method':'urdf_gripper_mount_transfer' if provisional else 'handeye_measured_tcp',
        'camera_source':'shared_head_calibration','tcp_m':tcp.tolist(),
        'independent_tool_calibration':not provisional}
    return result,tcp


def rank_arms(fk,point,preferred='left'):
    from autolife_handeye_calibration_306.core import TORSO
    point=np.asarray(point,float)
    if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError('物体三维位置无效')
    distances={}
    for side in ('left','right'):
        child='Link_'+side.title()+'_Wrist_Lower_to_Gripper'
        while fk.joints[child][2]!=TORSO:child=fk.joints[child][2]
        shoulder=fk.joints[child][3][:3,3]
        distances[side]=float(np.linalg.norm(point-shoulder))
    return sorted(distances,key=lambda s:(round(distances[s],4),s!=preferred))


class ArmPlannerPool:
    def __init__(self,cfg,calibration,fk,factory,calibration_profiles=None):
        self.base=copy.deepcopy(cfg);self.calibration=calibration;self.fk=fk;self.factory=factory
        self.planners={};self.attempts=[];self.diagnostics=[]
        self.calibration_profiles=calibration_profiles

    def for_config(self,cfg):
        side=arm_name(cfg['arm'])
        if side not in self.planners:self.planners[side]=self.factory(copy.deepcopy(cfg))
        planner=self.planners[side];planner.cfg=copy.deepcopy(cfg)
        return planner

    def plan(self,source,point,product,current_cfg,stopped=lambda:False,on_status=lambda text:None,observation_for_arm=None):
        from .core import corrected_target
        from .target_offsets import checked_profiles
        profiles=(checked_profiles(current_cfg['target_offsets_by_arm_torso_mm'])
                  if 'target_offsets_by_arm_torso_mm' in current_cfg else None)
        self.attempts=[];self.diagnostics=[]
        order=rank_arms(self.fk,point,self.base.get('arm','left')) if self.base.get('auto_arm_selection',True) else [self.base['arm']]
        for side in order:
            if stopped():raise ValueError('用户停止')
            label='左手' if side=='left' else '右手'
            on_status('自动选手：检查'+label+'抓取与收回路径…')
            profile_cfg,profile_data=(self.base,self.calibration)
            if self.calibration_profiles is not None:
                profile_cfg,profile_data,_=self.calibration_profiles.for_arm(side)
            config,tcp=tool_geometry(profile_cfg,profile_data,self.fk,side)
            if self.calibration_profiles is not None:
                config['tool_geometry'].update(camera_source='arm_calibration_profile',
                    calibration_profile=config['calibration_profile'])
            if profiles is not None:
                config['target_offset_torso_mm']=list(profiles[side])
                config['target_offsets_by_arm_torso_mm']=copy.deepcopy(profiles)
            config['up_torso']=list(current_cfg['up_torso'])
            if 'semantic_prompt_overrides' in current_cfg:
                config['semantic_prompt_overrides']=copy.deepcopy(current_cfg['semantic_prompt_overrides'])
            planner=self.for_config(config)
            try:
                arm_point,arm_product=(observation_for_arm(side,config) if observation_for_arm else (point,product))
                center,correction=corrected_target(arm_point,config.get('target_offset_torso_mm',[0.,0.,0.]))
                plan=planner.plan(source,tcp,center,arm_product)
            except ValueError as exc:
                self.attempts.append({'arm':side,'status':'unavailable','reason':str(exc)})
                self.diagnostics.append({'arm':side,'failures':getattr(planner,'candidate_failures',[])})
                continue
            if stopped():raise ValueError('用户停止')
            self.attempts.append({'arm':side,'status':'planned'})
            plan.update(arm=side,arm_selection=copy.deepcopy(self.attempts),tool_geometry=config['tool_geometry'])
            return plan,planner,config,tcp,center,correction
        raise ValueError('双臂路径均未通过：'+'；'.join(('左手' if a['arm']=='left' else '右手')+'：'+a['reason'] for a in self.attempts))
