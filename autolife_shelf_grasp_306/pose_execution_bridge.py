"""姿态执行与全身复位共用的原厂动作输出适配；不创建Qt窗口。"""
import json,time

def make_bridge(base,context):
    class ReceiptBridge(base):
        outcome = None
        teaching_segments = None
        def __init__(inner):
            super().__init__()
            from rclpy.parameter import Parameter
            inner.set_parameters([Parameter('command_update_rate_hz',value=50.)])
            for timer in inner.timers:
                if timer.callback==inner._advance_motion:
                    timer.timer_period_ns=20_000_000;timer.reset()

        def _advance_action(inner):
            from .teaching_replay import advance
            if advance(inner):return
            from .pose_synchronized_output import before_tick
            try:
                before_tick(inner)
                super()._advance_action()
            except (ValueError,KeyError,TypeError) as exc:
                inner._lock('姿态同步计算失败，已停止目标输出：'+str(exc))
                return
            state=inner._action_state
            if state and state['stage']=='hold':
                from control_center_joint_control.joint_definitions import BODY_JOINTS
                quiet=inner._is_feedback_fresh() and all(abs(float(inner._details.get(n,{}).get('speed',float('inf'))))<=2. for n in BODY_JOINTS)
                if quiet and state.get('jaw_steady_since') is None:state['jaw_steady_since']=time.monotonic()
                elif not quiet:state['jaw_steady_since']=None
            if state and state['stage']=='hold' and state['record']['keyframes'][state['frame_index']]['hold_sec']==0:
                inner._advance_action_frame(time.monotonic())

        def _publish_action_targets(inner,state,desired,frame):
            if inner.teaching_segments and state['stage']=='transition':
                from .teaching_trajectory import evaluate
                segment=inner.teaching_segments[state['frame_index']]
                desired=evaluate(segment,state['progress'])
                inner._preview.update(desired)
                if segment.get('recorded_timing'):
                    from .teaching_output import publish
                    publish(inner,state,desired)
                    return
                if 'curve' in segment:
                    # Recorded paths can reverse inside a block. Keep
                    # the vendor feedback-lead guard, using each new
                    # curve sample as the local target rather than
                    # clamping the whole curve to its final endpoint.
                    proxy=dict(state);proxy['start_values']=dict(state['commanded_values'])
                    local_frame=dict(frame,joints_deg=desired)
                    from .pose_gripper_intent import publish_pose
                    publish_pose(inner,super()._publish_action_targets,proxy,desired,local_frame)
                    return
                from .pose_synchronized_output import progress,publish_body
                desired=progress(inner,state,segment,state['progress'])
                inner._preview.update(desired)
                from .pose_gripper_intent import publish_pose
                publish_pose(inner,lambda st,q,f:publish_body(inner,st,q,f),state,desired,frame)
                return
            from .pose_gripper_intent import publish_pose
            publish_pose(inner,super()._publish_action_targets,state,desired,frame)

        def _through_point(inner):
            state=inner._action_state
            if not state or not inner.teaching_segments:return False
            i=state['frame_index'];frames=state['record']['keyframes']
            return i<len(frames)-1 and frames[i]['hold_sec']==0 and any(
                abs(v)>1e-8 for v in inner.teaching_segments[i]['v1'].values())

        def _advance_hardware_prepare(inner):
            from .teaching_enable import wait_for_enable_feedback
            if wait_for_enable_feedback(inner):return
            super()._advance_hardware_prepare()
        def _advance_motion(inner):
            if getattr(inner,'task_pause',None) and inner.task_pause.tick():return
            super()._advance_motion()
        def _record_action_event(inner,kind,reason=''):
            # Editor playback and task-flow playback share this bridge.
            # Persist evidence before cancellation clears action state.
            state=inner._action_state
            if not state:
                prepare=getattr(inner,'_hardware_prepare',None)
                if not prepare:return
                try:
                    from control_center_joint_control.joint_definitions import BODY_JOINTS
                    from .core import atomic_json
                    entry=dict(time=time.time(),event=kind,reason=str(reason),name='使能准备',
                        frame_index=0,loop_index=0,stage=prepare.get('stage'),
                        targets=prepare.get('hold_targets',{}),measured=dict(inner._current),
                        commanded={},details=dict(inner._details),
                        lost_joints=inner._communication_lost_joints(BODY_JOINTS),
                        feedback_age_sec=time.monotonic()-inner._feedback_monotonic)
                    with (context.owner.run/'action_events.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(entry,ensure_ascii=False)+'\n')
                    if kind in ('stopped','locked'):atomic_json(context.owner.run/'last_action_error.json',entry)
                except (OSError,ValueError,KeyError,TypeError,AttributeError):pass
                return
            try:
                frame=state['record']['keyframes'][state['frame_index']]
                entry=dict(time=time.time(),event=kind,reason=str(reason),name=state['name'],
                    frame_index=state['frame_index']+1,loop_index=state['loop_index'],
                    targets=state.get('replay_desired',frame['joints_deg']),measured=dict(inner._current),
                    segment_end_targets=frame['joints_deg'],recorded_elapsed=state.get('recorded_elapsed'),
                    commanded=dict(state['commanded_values']),details=dict(inner._details),
                    feedback_age_sec=time.monotonic()-inner._feedback_monotonic)
                from .core import atomic_json
                with (context.owner.run/'action_events.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(entry,ensure_ascii=False)+'\n')
                if kind in ('stopped','locked'):atomic_json(context.owner.run/'last_action_error.json',entry)
            except (OSError,ValueError,KeyError,TypeError,AttributeError):pass
        def _action_target_reached(inner,targets):
            state=inner._action_state
            if state and inner.teaching_segments and not inner.teaching_segments[state['frame_index']].get('recorded_timing'):
                from .pose_gripper_intent import stable_closing
                if stable_closing(inner,state,targets,state['commanded_values'],state['start_values']):
                    from control_center_joint_control.joint_definitions import GRIPPER_JOINTS
                    context.gripper_targets={n:state['commanded_values'][n] for n in GRIPPER_JOINTS}
                    inner._event('闭合目标已发送并稳定保持；物体限制开口不继续等待，不代表已验证抓牢')
                    return True
            if inner._through_point():
                # No settle or dwell at a through point. Retain bounded
                # tracking: a lagging actuator cannot skip a waypoint.
                from control_center_joint_control.joint_definitions import GRIPPER_JOINTS
                state=inner._action_state
                lead=max(.1,float(inner.get_parameter('max_single_step_deg').value))
                return all(abs(state['commanded_values'][n]-v)<=.25 and
                    abs(inner._current[n]-v)<=(5. if n in GRIPPER_JOINTS else lead+.25)
                    for n,v in targets.items())
            if super()._action_target_reached(targets):
                from .action_arrival import remember_opening_arrival
                remember_opening_arrival(inner,targets)
                if inner._action_state:
                    inner._action_state.pop('opening_arrival',None)
                    inner._action_state.pop('arm_arrival',None)
                return True
            from .action_arrival import settled_opening_arrival,settled_full_open_arrival
            if settled_full_open_arrival(inner,targets):
                inner._record_action_event('settled_full_open','开口端目标已下发；目标与反馈均在全开端区域内，残差≤5编码单位并稳定0.5秒；其他关节容差不变')
                inner._event('夹爪全开端已稳定，继续下一阶段')
                return True
            if settled_opening_arrival(inner,targets):
                inner._record_action_event('settled_opening','松开或保持已确认松开状态，夹爪残差≤5°并稳定0.5秒')
                inner._event('松爪/保持松开已稳定，小于5°残差继续下一帧；身体与手臂容差不变')
                return True
            from .action_pose_arrival import settled_arm_arrival,PoseEnvelope
            def geometry(actual,target):
                if not hasattr(inner,'_arrival_envelope'):
                    import yaml
                    cfg=yaml.safe_load((context.owner.root/'config/grasp.yaml').read_text('utf-8'))
                    inner._arrival_envelope=PoseEnvelope(cfg['urdf_path'])
                return inner._arrival_envelope.check(actual,target)
            receipt=settled_arm_arrival(inner,targets,geometry)
            if receipt is None:return False
            # Only already stationary arms are held at their measured
            # position. Do not change jaws, waist/legs, neck or records.
            from control_center_joint_control.joint_definitions import GROUPS,group_joint_names
            affected={side+'_arm' for side in ('left','right') if any(
                n.startswith('Joint_'+side.title()+'_') for n in receipt['errors_deg'])}
            state=inner._action_state
            payload={}
            for group in ('neck','left_arm','right_arm','leg_waist'):
                names=group_joint_names(group)
                values=[float(inner._current[n]) if group in affected else
                        float(state['commanded_values'][n] if n in state['commanded_values'] else
                              inner._controller_target[n]) for n in names]
                payload[GROUPS[group]['command_key']]=values
                if group in affected:state['commanded_values'].update(zip(names,values))
            inner._publish_vendor(inner._whole_body_pub,payload)
            state.setdefault('arrival_deviations',[]).append(receipt)
            inner._record_action_event('settled_arm_tolerance',json.dumps(receipt,ensure_ascii=False))
            inner._event('手臂小偏差已停稳并通过5mm运动学包络检查，从实测姿态衔接；未跳过关键帧')
            return True
        def _action_error_summary(inner,frame):
            errors=sorted(((abs(float(inner._current[n])-float(v)),n,float(v))
                for n,v in frame['joints_deg'].items()),reverse=True)[:3]
            from control_center_joint_control.joint_definitions import GRIPPER_JOINTS
            return '；'.join(f'{n} 实测{inner._current[n]:.3f}/目标{v:.3f}/偏差{e:.3f}'
                            + ('编码单位' if n in GRIPPER_JOINTS else '°') for e,n,v in errors)
        def _advance_action_frame(inner, now):
            state=inner._action_state;index=state['frame_index'];frames=state['record']['keyframes']
            from .teaching_motion import jaws_after_body
            next_timed=inner.teaching_segments and index+1<len(inner.teaching_segments) and inner.teaching_segments[index+1].get('recorded_timing')
            if not next_timed and jaws_after_body(frames[index],frames[index+1] if index+1<len(frames) else None):
                from control_center_joint_control.joint_definitions import BODY_JOINTS
                gate=state.setdefault('jaw_arrival_gate',dict(start=now))
                if now-gate['start']>8.:
                    inner._cancel_action('关节到位后仍未停稳，未执行夹爪开合');return
                if not inner._is_feedback_fresh() or any(abs(float(inner._details.get(n,{}).get('speed',float('inf'))))>2. for n in BODY_JOINTS):
                    state['jaw_steady_since']=None;return
                if state.get('jaw_steady_since') is None:return
                if now-state['jaw_steady_since']<.25:return
                state.pop('jaw_arrival_gate',None)
            through=inner._through_point()
            old_end=state['frame_started']+frames[index]['transition_sec']
            old_commands=dict(state['commanded_values'])
            from control_center_joint_control.joint_definitions import GRIPPER_JOINTS
            context.gripper_targets={n:state['commanded_values'][n] for n in GRIPPER_JOINTS}
            inner._record_action_event('frame_completed')
            name = inner._action_state['name']
            deviations=len(inner._action_state.get('arrival_deviations',[]))
            super()._advance_action_frame(now)
            if inner._action_state:inner._action_state.pop('jaw_steady_since',None)
            if through and inner._action_state:
                next_state=inner._action_state
                # Preserve the planned tangent and time across a pass
                # point, rather than restarting a rest-to-rest move.
                next_state['start_values']=dict(frames[index]['joints_deg'])
                next_state['commanded_values']=old_commands
                if now-old_end<.08:next_state['frame_started']=old_end
                else:
                    inner._cancel_action('通过点跟随滞后，已停止；请降低动作速度后重试')
                    return
            if inner._action_state is None:
                inner.outcome = (name, True, f'动作完成（{deviations}帧以验证后的小偏差衔接）' if deviations else '动作完成')
        def _cancel_action(inner, reason):
            state=inner._action_state
            if state and state.get('stage')=='transition' and state.get('progress',0.)>=1.:
                phases=getattr(context,'teaching_phases',[]);index=state['frame_index']
                phase=phases[index][1] if index<len(phases) else '姿态到位检查'
                frame=state['record']['keyframes'][index]
                reason=f'{phase}：{reason}；'+inner._action_error_summary(frame)
            inner._record_action_event('stopped',reason)
            if inner._action_state: inner.outcome = (inner._action_state['name'], False, reason)
            super()._cancel_action(reason)
        def _lock(inner, reason):
            inner._record_action_event('locked',reason)
            if inner._action_state: inner.outcome = (inner._action_state['name'], False, reason)
            super()._lock(reason)
    return ReceiptBridge
