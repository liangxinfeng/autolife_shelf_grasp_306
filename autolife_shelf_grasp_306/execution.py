"""Experimental, explicitly gated executor; never launched by perception.

Reuses hardware ownership/hold protection. Stable closure automatically starts
the checked carry path, but is NOT proof that an object was securely grasped.
"""
import argparse
import json
import time
import uuid
from pathlib import Path
import numpy as np
import rclpy
from autolife_handeye_calibration_306.bridge import competing_controller
from .bridge import GraspBridge as CalibrationBridge
from autolife_handeye_calibration_306.core import ARMS,BODY,NECK
from autolife_handeye_calibration_306.verification_trajectory import JointPath
from control_center_joint_control.joint_definitions import GROUPS
from .core import execution_blockers,reset_blockers,atomic_json,open_tolerance,check_plan_identity
from .motion import Follower,Arrival,GoalSender,configure_grasp_path,configure_reset_path,transit_pose_ok,feedback_lead,transfer_lead
from .motion import intermediate_tolerance
from .sync_session import PositionSyncSession
from .joint_limits import validate_grasp_plan,check_pose
from .carry import validate_carry_plan
from .pose_handoff import start_pose_diagnostic,await_loaded_pose,await_start_pose
from .arm_selection import plan_arm,gripper_name,gripper_targets


def hold_groups(node,groups):
    """Cancel this executor's pending goals and hold fresh measured positions."""
    if not node._hardware_ready:return
    q=node.snapshot();payload={}
    for group in groups:
        node._active_goals.pop(group,None);node._goal_sent_groups.discard(group)
        node.progress_watch.pop(group,None)
        if group!='grippers':
            payload[GROUPS[group]['command_key']]=[q[n] for n,*_ in GROUPS[group]['joints']]
    if payload:node._publish_vendor(node._whole_body_pub,payload)
    if 'grippers' in groups and not getattr(node,'preserve_loaded_grippers',False):
        node._publish_vendor(node._gripper_pub,{'left_gripper_target_joints_position':[q['Joint_Left_Gripper']],
                                              'right_gripper_target_joints_position':[q['Joint_Right_Gripper']]})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--run',required=True)
    parser.add_argument('--expected-plan-id');parser.add_argument('--expected-product-id')
    args=parser.parse_args();run=Path(args.run);node=None;session=None;base_guard=None;record={'status':'not_started'};preparing=False;placing=False;observing=False;workspace_reset=False
    arm='left';arm_group='left_arm';vendor_grasp=False;vendor_place=False;native_motion=False;native_trajectory_function=None
    preloader=None
    timing_started=time.monotonic();timings={}
    def mark(name):
        timings[name]=round(time.monotonic()-timing_started,4)
        atomic_json(run/'execution_timing.json',dict(started_wall_time=time.time()-(time.monotonic()-timing_started),stages=timings))
    mark('executor_started')
    try:
        plan=json.loads(Path(args.plan).read_text('utf-8'));cfg=plan['config']
        vendor_grasp=plan.get('execution_backend')=='vendor_trajectory_v1'
        vendor_place=plan.get('execution_backend')=='vendor_place_trajectory_v1'
        native_motion=vendor_grasp or vendor_place
        if vendor_grasp:
            if plan.get('kind') in ('prepare','place','observe'):
                raise ValueError('原厂抓取整段执行器不能用于复位或放置')
            from .vendor_grasp_trajectory import validate_blocks,execute_blocks,load_trajectory_function
            validate_blocks(plan)
            # 在使能/进入SYNC之前加载原厂模块，避免首次导入阻塞控制心跳。
            from .trajectory_preload import TrajectoryPreload
            preloader=TrajectoryPreload(load_trajectory_function,cfg)
        if vendor_place:
            from .vendor_place_trajectory import validate_blocks as validate_place_blocks
            from .vendor_grasp_trajectory import load_trajectory_function
            validate_place_blocks(plan)
            from .trajectory_preload import TrajectoryPreload
            preloader=TrajectoryPreload(load_trajectory_function,cfg)
        arm=plan_arm(plan);arm_group=arm+'_arm';other='right' if arm=='left' else 'left';jaw=gripper_name(arm)
        preparing=plan.get('kind')=='prepare'
        workspace_reset=bool(preparing and plan.get('workspace_reset'))
        if plan.get('reset_protocol_version')==4 and not workspace_reset:raise ValueError('全身复位请求缺少目标标记')
        if workspace_reset:
            from .workspace_reset import validate_plan
            validate_plan(plan)
        placing=plan.get('kind')=='place'
        observing=plan.get('kind')=='observe'
        from .grip_width import command_from_plan
        close_angle=command_from_plan(plan) if not (preparing or placing or observing) else None
        if placing or observing:
            from .demo_protocol import validate_place
            validate_place(plan,run)
        if preparing and plan.get('prepare_mode') is not None:
            if plan['prepare_mode']!='hold_then_vr' or plan.get('reset_protocol_version') not in (2,3,4):
                raise ValueError('旧版复位请求不可执行，请重新点击复位并准备')
        if not preparing:check_plan_identity(plan,run.name,args.expected_plan_id,args.expected_product_id)
        if not (placing or observing):
            (run/'held_object.json').unlink(missing_ok=True)
        receipt_path=run/'prepare_result.json'
        if preparing:atomic_json(receipt_path,{'status':'preparing'})
        elif not (placing or observing) and cfg.get('require_reset_before_grasp',True):
            receipt=json.loads(receipt_path.read_text('utf-8')) if receipt_path.exists() else {}
            if receipt.get('status')!='prepared' or not plan.get('prepare_token') or receipt.get('token')!=plan['prepare_token']:
                raise ValueError('每轮抓取前请先复位并准备，再重新识别规划')
        blockers=reset_blockers(cfg) if preparing else execution_blockers(cfg)
        if blockers:raise ValueError('；'.join(blockers))
        if time.monotonic()-plan['created_monotonic']>cfg['plan_ttl_sec']:raise ValueError('规划已过期，请重新规划')
        if competing_controller():raise ValueError(competing_controller())
        if not plan.get('self_collision_checked'):raise ValueError('缺少路径检查')
        neck_contact_segments=[s for s in plan['segments'] if s.get('supported_neck_recovery')]
        if neck_contact_segments:
            from .supported_neck_recovery import validate_segment,verify_installed_model
            if (not preparing or plan.get('reset_protocol_version') not in (3,4) or
                    len(neck_contact_segments)!=1 or plan['segments'][0] is not neck_contact_segments[0]):
                raise ValueError('颈部支撑恢复只能作为手动复位的第一个阶段')
            if cfg.get('supported_neck_recovery_enabled') is not True:
                raise ValueError('本机未开放已验证的颈部支撑恢复')
            validate_segment(neck_contact_segments[0],plan['source_q'])
            verify_installed_model()
        elif preparing and plan.get('reset_protocol_version')==3:
            raise ValueError('颈部恢复计划缺少恢复阶段，请重新准备')
        if not preparing:
            up=np.asarray(plan.get('up_torso',[]),float)
            if plan.get('geometry_version')!=2 or up.shape!=(3,) or not np.isfinite(up).all() or abs(np.linalg.norm(up)-1)>.001:
                raise ValueError('旧版或无效抓取路径，请重新规划')
        from autolife_handeye_calibration_306.core import digest
        if digest(cfg['urdf_path'])!=plan['urdf_sha256']:raise ValueError('机器人模型已改变')
        if not preparing:
            from .grasp_geometry import validate_plan_geometry
            validate_plan_geometry(plan,cfg)
        # Reject stale plans BEFORE creating a bridge, unlocking or entering SYNC.
        limit_policy=None
        if not preparing:
            limits={n:(lo,hi) for n,_,lo,hi in GROUPS[cfg['arm']+'_arm']['joints']}
            limit_policy=validate_grasp_plan(plan,ARMS[cfg['arm']],limits)
            if not (placing or observing):validate_carry_plan(plan,ARMS[cfg['arm']])
        tolerance=float(cfg.get('reset_tolerance_deg',1.)) if preparing else .3
        command_tolerance=1. if workspace_reset else intermediate_tolerance(tolerance,preparing)
        bridge_lead=max(transfer_lead(cfg,s.get('phase',''),plan.get('kind')) for s in plan['segments'])
        ros_args=['--ros-args','-p','robot_id:=345','-p',f'goal_tolerance_deg:={command_tolerance}',
                  '-p','command_update_rate_hz:=60.0',
                  '-p',f'max_single_step_deg:={bridge_lead}',
                  '-p','max_gripper_command_lead_deg:=30.0']
        if native_motion:
            # 此阶段确认通讯与保持，后续SYNC独立核对切换后的连续新反馈。
            ros_args+=['-p','hardware_reset_stable_sec:=0.2']
        if vendor_grasp:
            # 使能的旧3°到位门限不能用于后续0.5°起点交接。仅本抓取进程生效。
            ros_args+=['-p','hardware_reset_position_tolerance_deg:=0.3']
        for name in ['joint_states','target_joint_states','status','preview_command','apply_command','action_command',
                     'set_hardware_enabled','set_hardware_enabled_without_reset','set_leg_waist_enabled']:
            ros_args+=['-r',f'/control_center_joint_control/{name}:=/shelf_grasp306/{name}']
        mark('preflight_complete')
        rclpy.init(args=ros_args)
        if workspace_reset:
            from .workspace_reset_bridge import make_reset_bridge
            from types import SimpleNamespace
            context=SimpleNamespace(owner=SimpleNamespace(root=Path(cfg['workspace_reset_path']).parents[2],run=run))
            node=make_reset_bridge(CalibrationBridge,context)()
        else:node=CalibrationBridge()
        node.preserve_loaded_grippers=bool(placing or observing)
        from .task_pause import BridgePause
        node.task_pause=BridgePause(node,run)
        motion_now=node.task_pause.now
        source=plan['source_q']
        handoff_diagnostics={}
        def check_start_pose(stage):
            def save(diagnostic):handoff_diagnostics[stage]=diagnostic
            try:
                if (placing or observing) and stage!='执行前':
                    await_loaded_pose(node.snapshot,spin,lambda:node._feedback_monotonic,
                        source,stage,plan.get('kind'),time.monotonic,save,arm=arm)
                elif vendor_grasp and stage in ('使能期间','SYNC切换期间'):
                    await_start_pose(node.snapshot,spin,lambda:node._feedback_monotonic,
                        source,stage,time.monotonic,save,arm=arm)
                else:
                    from .feedback_state import await_ready_snapshot
                    current=await_ready_snapshot(node.snapshot,spin,stage,on_report=save)
                    diagnostic=start_pose_diagnostic(current,source,stage,plan.get('kind'),arm=arm);save(diagnostic)
                    if not diagnostic['ok']:raise ValueError(diagnostic['message'])
            finally:
                atomic_json(run/'pose_handoff.json',dict(plan_id=plan.get('plan_id'),phases=handoff_diagnostics))
        names=(BODY if workspace_reset else [])+ARMS['left']+ARMS['right']+NECK if preparing else ARMS[arm]
        def spin():
            for _ in range(10):rclpy.spin_once(node,timeout_sec=0)
            if (run/'stop').exists():raise ValueError('用户停止')
            heartbeat=run/'heartbeat'
            if not heartbeat.exists() or time.time()-heartbeat.stat().st_mtime>1.:raise ValueError('界面心跳中断')
            if session is not None:session.tick()
            if base_guard is not None and base_guard.last is not None:base_guard.check()
            time.sleep(.005)
        deadline=time.monotonic()+3
        while not node._is_feedback_fresh():
            spin()
            if time.monotonic()>deadline:raise ValueError('关节反馈未就绪')
        from .feedback_refresh import refresh, await_snapshot
        print('核对原厂实时关节位置（不使能、不运动）…',flush=True)
        refreshed=refresh(node,spin)
        await_snapshot(node.snapshot,spin,refreshed)
        mark('feedback_refreshed')
        check_start_pose('执行前')
        if (placing or observing) and plan.get('placement_mode')=='standalone':
            from .placement_request import standalone_source,check_held_gripper
            _,_,held_record=standalone_source(run)
            check_held_gripper(node.snapshot(),held_record)
        released=False
        if placing or observing or vendor_grasp:
            from .demo_protocol import StationaryBaseGuard
            base_guard=StationaryBaseGuard(node)
            deadline=time.monotonic()+2
            while base_guard.last is None:
                spin()
                if time.monotonic()>deadline:raise ValueError('等待底盘停稳反馈超时')
            base_guard.check()
            if plan.get('base_anchor') is not None:
                from .demo_protocol import check_base_anchor
                check_base_anchor(list(base_guard.last[1:4]),plan['base_anchor'])
        if preparing:print('路径检查完成；保持当前位置使能，按预检顺序恢复准备姿态。',flush=True)
        if preloader is not None:
            native_trajectory_function=preloader.result(spin)
            preloader.close();preloader=None
            mark('trajectory_function_ready')
        mark('enable_requested')
        if native_motion:
            # 抓取/放置共用已验证的单次保持式接管；原厂入口同时完成使能和SYNC。
            from .pose_manual_session import ManualHoldSession
            session=ManualHoldSession(node,preserve_gripper_targets=bool(placing or observing))
            session.requires_grasp_finish=vendor_grasp
            session.enter_prepared(spin,refreshed)
            mark('enable_ready');mark('sync_ready')
            check_start_pose('SYNC切换期间')
            print('单次保持式接管已就绪；未重复逐组使能。',flush=True)
        else:
            response=node.unlock()
            if not response.success:raise ValueError(response.message)
            deadline=time.monotonic()+(8. if vendor_grasp else 47.)
            while not node._hardware_ready:
                spin()
                if not getattr(node,'_hardware_enabled',True):
                    raise ValueError('使能复位已停止：'+getattr(node,'last_lock_reason','请检查硬件状态'))
                if time.monotonic()>deadline:
                    diagnostic=start_pose_diagnostic(node.snapshot(),source,'使能稳定等待',plan.get('kind'),arm=arm)
                    handoff_diagnostics['使能稳定等待']=diagnostic
                    atomic_json(run/'pose_handoff.json',dict(plan_id=plan.get('plan_id'),phases=handoff_diagnostics))
                    raise ValueError('保持式使能未稳定，尚未开始运动；'+(diagnostic['message'] or '原厂准备未完成'))
            mark('enable_ready')
            check_start_pose('使能期间')
            if cfg.get('use_position_sync_session',not preparing) or neck_contact_segments or placing or observing:
                session=PositionSyncSession(node,preserve_gripper_targets=bool(placing or observing))
                session.requires_grasp_finish=vendor_grasp;session.enter(spin)
                mark('sync_ready')
                check_start_pose('SYNC切换期间')
                print('保持式位置SYNC已就绪；按预检轨迹分阶段限速。',flush=True)
        if native_motion:
            response=node.enable_planned_waist(plan)
            if not response.success:raise ValueError(response.message)
        if workspace_reset:
            from std_srvs.srv import SetBool
            response=node.enable_workspace_body()
            if not response.success:raise ValueError(response.message)
        # A reset has no visual target. Source-pose and URDF checks above remain
        # mandatory; a slow hold-only enable must not expire its checked path.
        if not preparing and time.monotonic()-plan['created_monotonic']>cfg['plan_ttl_sec']:raise ValueError('使能后规划过期，请重新规划')
        def checked():
            spin()
            while (run/'task_pause').exists():
                if not node._hardware_ready:raise ValueError('暂停期间控制权失效')
                spin()
            q=node.snapshot()
            if base_guard is not None:base_guard.check()
            if not node._hardware_ready:raise ValueError('控制权/硬件状态失效：'+getattr(node,'last_lock_reason','请检查驱动状态'))
            fixed=([] if workspace_reset else BODY) if preparing else (BODY[:2] if native_motion else BODY)+NECK+ARMS[other]+[gripper_name(other)]
            if fixed and max(abs(q[n]-source[n]) for n in fixed)>.7:raise ValueError('固定身体/另一侧关节已移动')
            if observing and abs(q[jaw]-plan['held_gripper_deg'])>8:
                raise ValueError('观察期间夹爪状态改变，已停止抬手')
            if (placing or observing) and plan.get('placement_mode')=='standalone' and not released:
                check_held_gripper(q,held_record)
            return q
        def start_gripper(angle,both=False,closing=False,explicit_targets=None):
            if observing:raise ValueError('观察姿态阶段禁止开合夹爪')
            begin=checked();start=motion_now()
            targets=explicit_targets if explicit_targets is not None else gripper_targets(arm,angle,begin,both=both)
            if not native_motion:node.apply('grippers',targets,speed_scale=cfg.get('gripper_speed_scale',.5))
            tolerance=2. if closing else open_tolerance(cfg)
            return dict(targets=targets,begin=begin,start=start,closing=closing,history=[],last_log=0.,
                        tolerance=tolerance,arrival=Arrival(targets,tolerance),result=None,
                        release_width_mm=plan['release_opening_m']*1000 if vendor_place and not closing else None,
                        required_opening_mm=(min(plan['grasp_width_m']*1000+float(cfg.get('grasp_opening_margin_mm',10.)),
                            float(cfg['gripper_max_width_m'])*1000) if vendor_grasp and not closing else None))
        def poll_gripper(state,q,now):
            if state['result'] is not None:return state['result']
            targets=state['targets'];tolerance=state['tolerance'];closing=state['closing']
            if now-state['last_log']>.5:
                atomic_json(run/'gripper_progress.json',{'targets_deg':targets,'measured_deg':{n:q[n] for n in targets},
                    'tolerance_deg':tolerance,'closing':closing,'time':now});state['last_log']=now
            if state['arrival'].update(q,now):state['result']='at_target'
            elif closing:
                value=q[jaw];state['history'].append((now,value))
                history=state['history']=[v for v in state['history'] if now-v[0]<=.7]
                if (now-state['start']>1. and len(history)>5 and now-history[0][0]>.6 and
                        abs(value-state['begin'][jaw])>8. and np.ptp([v[1] for v in history])<.8):
                    state['result']='contact_unconfirmed'
            if state['result'] is not None:hold_groups(node,['grippers']);return state['result']
            if now-state['start']>=15:
                details='，'.join(f'{n}: 实际{q[n]:.2f}°/目标{v:.2f}°' for n,v in targets.items())
                raise ValueError(f'夹爪未稳定到位（容差{tolerance:.1f}°）：'+details)
            return None
        def finish_gripper(state):
            if native_motion:
                from .grasp_gripper import execute as execute_gripper
                return execute_gripper(node,cfg,arm,state,checked,motion_now,hold_groups,
                    lambda value:atomic_json(run/'gripper_progress.json',value),plan.get('grasp_width_m',0.))
            while state['result'] is None:poll_gripper(state,checked(),motion_now())
            return state['result']
        opening=None
        if preparing:
            if neck_contact_segments:
                from .supported_neck_recovery import execute_segment,verify_clear_in_worker
                segment=neck_contact_segments[0]
                print(segment['name']+'…',flush=True)
                actual=execute_segment(node,segment,source,checked,hold_groups,motion_now,
                    lambda progress:atomic_json(run/'motion_progress.json',progress))
                verify_clear_in_worker(cfg,actual,checked,motion_now)
                print('头颈已抬离，完整自碰撞检查通过；现在继续双臂复位。',flush=True)
            from .reset_recovery import execute_recovery
            for segment in plan['segments']:
                if segment.get('neck_recovery'):
                    print(segment['name']+'…',flush=True)
                    execute_recovery(node,segment,checked,hold_groups,motion_now)
            print('恢复全身姿态，关节到位后恢复已保存夹爪开度…' if workspace_reset else '双臂、头部复位并同步张开双夹爪…',flush=True)
            if not workspace_reset:opening=start_gripper(cfg['gripper_open_deg'],both=True)
        elif not (placing or observing):
            atomic_json(receipt_path,{'status':'consumed','token':plan.get('prepare_token')})
            finish_gripper(start_gripper(cfg['gripper_open_deg']))
            mark('opening_complete')
        contact=None
        fk=None
        if not preparing:
            from autolife_handeye_calibration_306.core import Kinematics,TORSO
            from autolife_handeye_calibration_306.verification_core import wrist_frame,q_for_fk
            fk=Kinematics(cfg['urdf_path'])
        if workspace_reset:
            from .pose_reset_execution import execute_pose
            execute_pose(node,context,plan,checked,motion_now,run)
        if vendor_grasp:
            def native_close():
                value=finish_gripper(start_gripper(close_angle,closing=True))
                atomic_json(run/'grip_result.json',dict(state=value,commanded_angle_deg=close_angle,
                    grip_command=plan.get('grip_command'),arm=arm,measured_angle_deg=node.snapshot()[jaw]))
                return value
            contact=execute_blocks(node,plan,checked,motion_now,native_close,hold_groups,
                lambda progress:atomic_json(run/'motion_progress.json',progress),trajectory_function=native_trajectory_function)
        if vendor_place:
            from .vendor_place_trajectory import execute_blocks as execute_place_blocks
            def native_release():
                # release 状态在 checked() 中使用，调用前先转入显式放置阶段。
                node.preserve_loaded_grippers=False
                (run/'held_object.json').unlink(missing_ok=True)
                finish_gripper(start_gripper(cfg['gripper_open_deg']))
            def release_at_landing():
                nonlocal released
                released=True
                native_release()
            def endpoint_ok(q,endpoint,landing):
                return transit_pose_ok(fk.fk('Link_Zero_Point',wrist_frame(cfg),q_for_fk(q,cfg)),
                    fk.fk('Link_Zero_Point',wrist_frame(cfg),q_for_fk(endpoint,cfg)),plan['tcp'],
                    position_limit_m=.004 if landing else .008)
            execute_place_blocks(node,plan,checked,motion_now,release_at_landing,hold_groups,
                lambda progress:atomic_json(run/'motion_progress.json',progress),endpoint_ok,native_trajectory_function)
        for index,segment in enumerate([] if workspace_reset or native_motion else plan['segments']):
            if preparing and (segment.get('neck_recovery') or segment.get('supported_neck_recovery')):continue
            print(segment['name']+'…',flush=True)
            path=JointPath(names,segment['waypoints'])
            if preparing:configure_reset_path(path,cfg)
            else:configure_grasp_path(path,cfg)
            if placing or observing:
                cap=float(segment.get('path_speed_cap',4.))
                if not np.isfinite(cap) or cap<=0:raise ValueError('放置速度限制无效')
                path.max_speed=min(path.max_speed,cap)
            lead=transfer_lead(cfg,segment.get('phase',''),plan.get('kind'))
            # Older plans have no checked exit boundary: keep their return
            # lead at 3 degrees throughout instead of guessing clearance.
            transfer_from=segment.get('transfer_from_s',path.length) if placing and segment.get('phase')=='place_return' else None
            follower=Follower(path,motion_now(),lead,
                segment['path_speed_profile'] if placing and plan.get('place_protocol_version') in (2,3) else None,
                transfer_from_s=transfer_from)
            pause_generation=node.task_pause.generation
            sender=GoalSender();last_command=-float('inf')
            deadline=motion_now()+90;last_progress=0.
            stage_started=motion_now();previous_sample=None
            transit=not preparing and segment.get('phase') in ('pregrasp','withdraw','carry','place_approach','place_return','observe_lift')
            endpoint=segment['waypoints'][-1]
            goal_pose=fk.fk(TORSO,wrist_frame(cfg),q_for_fk(endpoint,cfg)) if fk is not None else None
            arrival=Arrival({n:endpoint[n] for n in names},tolerance if preparing else 1. if transit else .35,
                            duration=.15 if preparing else .2 if transit else .35)
            while True:
                q=checked();now=motion_now()
                if node.task_pause.generation!=pause_generation:
                    follower.velocity=0.;follower.measured_rate=0.;follower.last=now
                    sender=GoalSender();pause_generation=node.task_pause.generation
                if opening is not None:poll_gripper(opening,q,now)
                if now>deadline:
                    worst=max(names,key=lambda n:abs(q[n]-endpoint[n]))
                    raise ValueError(f'阶段运动超时：{segment["name"]}；{worst} '
                        f'实测{q[worst]:.2f}°/目标{endpoint[worst]:.2f}°；'
                        f'轨迹进度{100*follower.progress/path.length:.1f}%')
                command=None
                if now-last_command>=1./60.:
                    try:command=follower.step(q,now)
                    except ValueError as exc:
                        atomic_json(run/'trajectory_failure.json',dict(plan_id=plan.get('plan_id'),
                            stage=segment['name'],phase=segment.get('phase'),time=now,reason=str(exc),
                            tracking=follower.tracking,bridge_commands=node._active_goals.get(arm_group,{}).get('commands',{})))
                        raise
                    last_command=now
                if command is not None:
                    if limit_policy is not None:check_pose(limit_policy,command)
                    groups=[('left_arm',ARMS['left']),('right_arm',ARMS['right']),('neck',NECK)] if preparing else [(arm_group,names)]
                    for group,ns in groups:
                        targets={n:command[n] for n in ns}
                        if sender.needed(group,targets,q,command_tolerance):node.apply(group,targets,speed_scale=1.)
                # Reset uses the same 1deg physical tolerance as handeye. Do not
                # wait forever for unrelated/last sub-tolerance bridge goals.
                if (preparing or follower.final) and arrival.update(q,now):
                    accepted=not transit or transit_pose_ok(fk.fk(TORSO,wrist_frame(cfg),q_for_fk(q,cfg)),goal_pose,plan['tcp'])
                    if placing and segment.get('phase')=='place_lower':
                        accepted=transit_pose_ok(fk.fk(TORSO,wrist_frame(cfg),q_for_fk(q,cfg)),goal_pose,plan['tcp'],position_limit_m=.004)
                    if accepted:
                        print('到位：'+segment['name']+('（关节≤1°且末端≤8mm、方向≤2°）' if transit else ''),flush=True)
                        hold_groups(node,(['leg_waist'] if workspace_reset else [])+['left_arm','right_arm','neck'] if preparing else [arm_group]);break
                if now-last_progress>.5:
                    measured=np.array([q[n] for n in names])
                    actual_speed=(float(np.max(np.abs(measured-previous_sample[1]))/(now-previous_sample[0]))
                                  if previous_sample is not None else 0.)
                    progress={'stage':segment['name'],'progress':follower.progress/path.length,
                        'max_error_deg':max(abs(q[n]-segment['waypoints'][-1][n]) for n in names),
                        'active_groups':list(node._active_goals),'time':now,
                        'elapsed_sec':now-stage_started,'path_speed_limit_deg_s':path.max_speed,
                        'path_speed_deg_s':follower.velocity,'measured_peak_deg_s':actual_speed,
                        'command_lead_deg':float(np.max(np.abs(follower.command-measured))),
                        'feedback_limited_ticks':follower.limited_ticks,'ticks':follower.total_ticks}
                    progress['position_sync_active']=bool(session and session.active)
                    progress['tracking']=follower.tracking
                    progress['measured_deg']={n:q[n] for n in names}
                    progress['endpoint_deg']={n:endpoint[n] for n in names}
                    progress['bridge_commands_deg']=dict(node._active_goals.get(arm_group,{}).get('commands',{}))
                    if goal_pose is not None:
                        pose=fk.fk(TORSO,wrist_frame(cfg),q_for_fk(q,cfg));tcp=np.asarray(plan['tcp'])
                        progress['model_tcp_error_mm']=float(np.linalg.norm(
                            pose[:3,3]+pose[:3,:3]@tcp-goal_pose[:3,3]-goal_pose[:3,:3]@tcp)*1000)
                    atomic_json(run/'motion_progress.json',progress)
                    with (run/'motion_trace.jsonl').open('a') as trace:trace.write(json.dumps(progress)+'\n')
                    previous_sample=(now,measured.copy())
                    last_progress=now
            if placing and segment.get('phase')=='place_lower':
                # Consume before opening, including a later retreat failure.
                released=True
                node.preserve_loaded_grippers=False  # 仅到达放置点后允许显式松爪。
                (run/'held_object.json').unlink(missing_ok=True)
                finish_gripper(start_gripper(cfg['gripper_open_deg']))
                print('已到放置点并松爪，退离桌面…',flush=True)
            if segment.get('phase','grasp' if index==1 else '')=='grasp' and not preparing and not placing and not observing:
                contact=finish_gripper(start_gripper(close_angle,closing=True))
                atomic_json(run/'grip_result.json',{'state':contact,'commanded_angle_deg':close_angle,
                    'grip_command':plan.get('grip_command'),
                                                   'arm':arm,'measured_angle_deg':node.snapshot()[jaw]})
                # User-authorized automatic carry. A settled jaw can also mean
                # friction or an empty grasp: do not label this grip success.
                print('闭爪稳定，自动抬起并收回。',flush=True)
        if preparing:
            if workspace_reset:
                jaws={n:cfg['workspace_reset_joints'][n] for n in ('Joint_Left_Gripper','Joint_Right_Gripper')}
            else:finish_gripper(opening)
            q=checked()
            if workspace_reset:
                from .prepare import ready_after_place
                if not ready_after_place(cfg,q):raise ValueError('全身复位最终反馈未到保存目标，未标记准备完成')
            targets=jaws if workspace_reset else {n:cfg['gripper_open_deg'] for n in ['Joint_Left_Gripper','Joint_Right_Gripper']}
            if max(abs(q[n]-v) for n,v in targets.items())>open_tolerance(cfg):
                raise ValueError('复位末尾夹爪开度发生变化，请检查')
            record={'status':'prepared','q':node.snapshot(),'token':uuid.uuid4().hex,'time':motion_now()}
            print('全身及双夹爪已恢复到保存的复位姿态；请重新识别。' if workspace_reset else '复位完成，双夹爪全开；重新识别饮料。',flush=True)
        elif observing:
            record={'status':'observed','plan_id':plan['plan_id'],'q':node.snapshot()}
            print('观察姿态已停稳，保持夹爪，重新采集桌面。',flush=True)
        elif placing:
            record={'status':'placed','plan_id':plan['plan_id'],'q':node.snapshot(),'physical_placement_verified':False}
            print('演示流程完成：已执行放置并空手收回。',flush=True)
        else:
            record={'status':'completed','q':node.snapshot(),'grip_success_verified':False,
                    'grip_state':contact,'carry_trigger':'stable_closure',
                    'plan_id':plan['plan_id'],'product':plan['product']['name'],
                    'physical_accuracy_measured':False}
            mark('grasp_motion_complete')
            print('抓取流程结束，保持饮料；不会自动松爪。',flush=True)
    except Exception as exc:
        record={'status':'stopped','reason':str(exc)}
        if node and getattr(node,'enable_events',None):record['enable_events']=node.enable_events
        if node and getattr(node,'last_stop_details',None):record['diagnostic']=node.last_stop_details
        print('已停止：'+str(exc),flush=True)
    finally:
        if preloader is not None:preloader.close()
        if node:
            handoff_ok=True
            if workspace_reset and getattr(node,'_action_state',None):
                node._cancel_action('复位结束或停止，清除未完成动作')
            retain_grip=bool(vendor_grasp and record.get('status')=='completed' and contact in ('at_target','contact_unconfirmed'))
            try:hold_groups(node,(['leg_waist'] if workspace_reset else [])+['left_arm','right_arm','neck','grippers'] if preparing else (['leg_waist'] if native_motion else [])+[arm_group]+([] if retain_grip else ['grippers']))
            except Exception as exc:
                handoff_ok=False;print('保持反馈不可用：'+str(exc),flush=True)
            if session is not None:
                try:
                    handoff_ok=(session.close(preserve_grippers=True) if retain_grip else session.close()) and handoff_ok
                    if not handoff_ok:print('SYNC退出未收到确认，厂商心跳保护将保持并退出。',flush=True)
                except Exception as exc:
                    handoff_ok=False;print('SYNC退出异常，厂商心跳保护将接管：'+str(exc),flush=True)
            node._lock('饮料抓取停止/完成，保持当前位置')
            if record.get('status')=='prepared' and not handoff_ok:
                record={'status':'stopped','reason':'复位已停住，但保持交接未确认'}
            if record.get('status')=='completed':
                before=dict(record['q']);diagnostics={}
                try:
                    if not handoff_ok:raise ValueError('SYNC退出未确认，不能确认持物状态')
                    def feedback_only():
                        if (run/'stop').exists():raise ValueError('用户停止')
                        if not (run/'heartbeat').exists() or time.time()-(run/'heartbeat').stat().st_mtime>1:
                            raise ValueError('界面心跳中断')
                        rclpy.spin_once(node,timeout_sec=.02)
                    record['q']=await_loaded_pose(node.snapshot,feedback_only,lambda:node._feedback_monotonic,
                        before,'抓取结束保持确认','place',time.monotonic,lambda d:diagnostics.update(d),arm=arm)
                    record['pre_hold_q']=before;record['hold_feedback_confirmed']=True
                except Exception as exc:
                    record={'status':'stopped','reason':'抓取后持物未确认：'+str(exc),'plan_id':plan.get('plan_id')}
                    print(record['reason'],flush=True)
                atomic_json(run/'held_handoff.json',diagnostics)
            if record.get('status')=='placed':
                from .prepare import ready_after_place
                record['auto_prepare_allowed']=bool(handoff_ok and not (run/'stop').exists())
                # Validate AFTER the SYNC-to-hold handoff, not just at the last
                # commanded endpoint. Failed/cancelled runs never become ready.
                try:
                    if not handoff_ok or (run/'stop').exists():raise ValueError('保持交接未确认或用户停止')
                    fresh=await_loaded_pose(node.snapshot,spin,lambda:node._feedback_monotonic,
                        record['q'],'放置后准备确认','place',time.monotonic,lambda d:None,arm=arm)
                    if ready_after_place(cfg,fresh):
                        atomic_json(run/'prepare_result.json',dict(status='prepared',q=fresh,
                            token=uuid.uuid4().hex,time=time.monotonic(),source='placed_return'))
                        record['prepared_after_place']=True
                except Exception as exc:
                    record['preparation_note']=str(exc)
                    record['auto_prepare_allowed']=False
            node.destroy_node()
        if base_guard is not None:base_guard.close()
        if rclpy.ok():rclpy.shutdown()
        mark('executor_finished')
        record['execution_timing_sec']=dict(timings)
        record['arm']=arm
        atomic_json(run/'execution_result.json',record)
        if record.get('status')=='completed':atomic_json(run/'held_object.json',record)
        if preparing:atomic_json(run/'prepare_result.json',record)


if __name__=='__main__':main()
