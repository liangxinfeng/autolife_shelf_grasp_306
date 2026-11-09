"""借用唯一ActionExecutor进行手动调节；无独立ROS控制器、无自动复位。"""
import json,time
from .pose_workspace_data import checked_targets,operator_commands

def source_changed(current,source):
    # DM夹爪反馈量不是关节角精度，沿用现有交接的夹爪8单位容差。
    return any(abs(current[n]-v)>(8. if 'Gripper' in n else 1.) for n,v in source.items())

def begin(engine,token,values,source,speed=.4,profile=''):
    values=checked_targets(values)
    if engine.moving():raise ValueError('当前动作尚未结束')
    engine.ensure();engine.panel.setEnabled(False)
    engine.bridge.outcome=None
    engine.job=dict(token=token,manual=True,targets=values,source=dict(source),speed=speed,
                    profile=profile,phase='feedback',started=time.monotonic(),stable=None)


def advance(engine):
    j=engine.job;b=engine.bridge
    from std_srvs.srv import SetBool
    from std_msgs.msg import String
    def fail(text):raise ValueError(text)
    try:
        if time.monotonic()-j['started']>90:fail('手动调节超时，已停止并保持')
        if engine.sync_session:engine.sync_session.tick()
        if j['phase']!='feedback' and not b._hardware_enabled:fail(b._last_event)
        if j['phase']=='feedback':
            if not b._is_feedback_fresh():return
            if source_changed(b._current,j['source']):
                fail('机器人位置已改变，请重新取当前姿态后执行')
            # 先经过真实底层解析器，再请求使能：格式错误不会导致电机切换。
            commands=operator_commands(j['targets'],b._current,j['speed'],j['profile'])
            for cmd in commands:b._parse_operator_command(json.dumps(cmd))
            if getattr(engine,'manual_session',False) and engine.sync_session and engine.sync_session.active:
                engine.sync_session.tick();j['phase']='submit'
            else:
                from .pose_manual_session import ManualHoldSession
                from rclpy.parameter import Parameter
                results=b.set_parameters([Parameter('goal_tolerance_deg',value=.2)])
                if not all(r.successful for r in results):fail('手动关节精度参数未加载')
                engine.sync_exit_ok=True;engine.sync_session=ManualHoldSession(b);engine.sync_session.begin();j['phase']='sync'
        elif j['phase'] in ('sync','submit'):
            if j['phase']=='sync' and not engine.sync_session.poll():return
            engine.manual_session=True
            if not b._is_feedback_fresh():fail('反馈中断，未发送调节目标')
            if source_changed(b._current,j['source']):fail('使能后姿态变化，请重新取当前姿态')
            commands=operator_commands(j['targets'],b._current,j['speed'],j['profile'])
            if any(c['group']=='leg_waist' for c in commands):
                r=b._set_leg_waist_enabled(SetBool.Request(data=True),SetBool.Response())
                if not r.success:fail(r.message)
                # 原厂保持入口使能双臂、腰和颈；踝膝按本次实际选择补充使能。
                enabled=getattr(engine.sync_session,'lower_enabled',set())
                lower={n:True for n in ('Joint_Ankle','Joint_Knee') if n in j['targets'] and n not in enabled}
                if lower:
                    b._publish_vendor(b._joint_enable_pub,lower);engine.sync_session.lower_enabled=enabled|set(lower)
            for cmd in commands:
                b._on_apply(String(data=json.dumps(cmd)))
                if cmd['group'] not in b._active_goals:fail('调节目标未被接收：'+b._last_event)
            j['phase']='moving'
        elif j['phase']=='moving':
            if not b._hardware_ready or not b._is_feedback_fresh():fail('控制权或关节反馈失效')
            if b._active_goals or not engine.settled():j['stable']=None;return
            # 原桥已有组目标到位检查；夹爪使用本机反馈容差，不能硬套角度精度。
            if any(abs(b._current[n]-v)>(8. if 'Gripper' in n else .3) for n,v in j['targets'].items()):
                fail('目标未到位，请查看当前反馈或按停止')
            if j['stable'] is None:j['stable']=time.monotonic()
            elif time.monotonic()-j['stable']>.3:
                token=j['token'];engine.job=None;engine.panel.setEnabled(True)
                engine.completed.emit(token,True,'调节完成；保持会话已就绪，可继续选择关节直接执行')
    except Exception as exc:
        token=j['token'];engine.stop()
        # 不让释放异常掩盖停止结果；底层停止已先执行。
        try:engine.release()
        except Exception:pass
        engine.completed.emit(token,False,str(exc))
