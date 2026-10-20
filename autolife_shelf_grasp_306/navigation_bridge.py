"""ROS navigation client in its own process. Never publishes chassis velocity."""
import argparse,json,math,queue,sys,threading,time
from collections import deque
from pathlib import Path
import numpy as np
from .navigation_logic import (active_map,map_catalog,write_active,waypoint_poses,
                               GoalProgress,LocalizationGate,NavigationPause,IDLE)


_emit_lock=threading.Lock()
_event_file=None
def emit(event,**data):
    with _emit_lock:
        record=dict(event=event,**data)
        if _event_file is not None and event!='nav_snapshot':
            try:
                with _event_file.open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(time=time.time(),**record),ensure_ascii=False)+'\n')
            except OSError:pass
        print(json.dumps(record,ensure_ascii=False),flush=True)


def main():
    global _event_file
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    run=Path(args.run)
    _event_file=run/'navigation_events.jsonl'
    import cv2,rclpy,tf2_ros,redis
    from rclpy.qos import QoSProfile,DurabilityPolicy,qos_profile_sensor_data
    from nav_msgs.msg import OccupancyGrid,Odometry
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from std_msgs.msg import String,Bool
    from std_srvs.srv import Trigger,Empty
    from action_msgs.msg import GoalStatusArray
    from nav2_msgs.srv import LoadMap
    from nav2_msgs.msg import Costmap
    from control_center_navigation.srv import ListWaypoints,NavigateToWaypoint,SwitchWaypointMap
    rclpy.init();node=rclpy.create_node('grasp306_navigation_client')
    from .navigation_runtime import NavigationRuntime
    runtime=NavigationRuntime(run,emit,lambda:[i.node_name for i in node.get_publishers_info_by_topic('/map')])
    tf=tf2_ros.Buffer();listener=tf2_ros.TransformListener(tf,node)
    commands=queue.Queue();running=True
    def reader():
        for line in sys.stdin:
            try:commands.put(json.loads(line))
            except ValueError:pass
        commands.put({'action':'quit'})
    threading.Thread(target=reader,daemon=True).start()
    clients={key:node.create_client(kind,topic) for key,kind,topic in [
        ('list',ListWaypoints,'/list_waypoints'),('navigate',NavigateToWaypoint,'/navigate_to_waypoint'),
        ('pause',Trigger,'/pause_navigation'),('resume',Trigger,'/resume_navigation'),
        ('stop_ready',Trigger,'/navigation_stop_ready'),
        ('nomotion',Empty,'/request_nomotion_update'),
        ('cancel',Trigger,'/cancel_navigation'),('load',LoadMap,'/map_server/load_map'),
        ('switch',SwitchWaypointMap,'/switch_waypoint_map'),
        ('relocalize',Trigger,'/cartographer_global_relocalization/relocalize'),
        ('cancel_localization',Trigger,'/cartographer_global_relocalization/cancel')]}
    latest={};odom=None;odom_stamp=None;nav_bool=False;actions={};names=[];map_meta=None;arm=None
    localization_status={};obstacle_meta=None;obstacle_time=0.
    arm_history=deque(maxlen=160)
    goal=None;ack=False;arrived=False;stable=None;cancel_reason=None;cancel_sent=False
    pause=NavigationPause();last_pause_tick=time.monotonic()
    sent_at=0.;last_publish=0.;last_list=0.;list_future=None;map_op=None;map_fault=False
    pending=None;pending_kind=None;pending_time=0.;stopping=False;cancel_time=0.
    manual_pending=None;manual_confirmation=None;nomotion_future=None;last_nomotion=0.
    waiting_navigation=None
    from .task_flow import NavigationSettling
    localization_gate=LocalizationGate();pose_reason='等待定位坐标'
    redis_client=redis.Redis(socket_timeout=.15,socket_connect_timeout=.15)
    def status(m):
        nonlocal latest,arrived,cancel_reason
        try:
            data=json.loads(m.data);latest=data
            localization_gate.navigation(data)
            if goal and not cancel_reason:
                if (data.get('state')=='paused' and goal.get('workflow') and
                    pause.owns_paused_status((run/'task_pause').exists(),float(data.get('stamp',0)))):return
                result=goal['progress'].update(data)
                if result=='arrived':arrived=True
        except ValueError as exc:
            if goal:cancel_reason=str(exc)
    def odometry(m):
        nonlocal odom,odom_stamp
        stamp=(m.header.stamp.sec,m.header.stamp.nanosec)
        if stamp==odom_stamp:return
        odom_stamp=stamp;v=m.twist.twist
        odom=(time.monotonic(),math.hypot(v.linear.x,v.linear.y),abs(v.angular.z))
    def vendor(m):
        nonlocal nav_bool
        nav_bool=m.data
    def localization(m):
        nonlocal localization_status
        try:
            localization_status=json.loads(m.data)
            if localization_gate.pending and not localization_gate.initial_stamp and localization_status.get('state')=='failed':
                localization_gate.fail('自动重定位失败，请手动设置位置或重新搜索')
            # Attempt counts reset on every retry. Success is confirmed using
            # this request's initialpose + AMCL + fresh TF, never counter ordering.
        except (ValueError,TypeError):pass
    def initial_pose(m):
        if m.header.frame_id=='map':
            p=m.pose.pose.position;q=m.pose.pose.orientation
            localization_gate.initial(m.header.stamp.sec+m.header.stamp.nanosec/1e9,
                [p.x,p.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))])
    def amcl_pose(m):
        if m.header.frame_id!='map':return
        p=m.pose.pose.position;q=m.pose.pose.orientation;c=m.pose.covariance
        localization_gate.amcl(m.header.stamp.sec+m.header.stamp.nanosec/1e9,
            [p.x,p.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))],[c[0],c[7],c[35]])
    def joints(m):
        nonlocal arm
        from autolife_handeye_calibration_306.core import BODY,NECK,ARMS
        try:
            d=json.loads(m.data);q={}
            for key,ns in [('leg_waist_joint_state',BODY),('neck_joint_state',NECK),
                           ('left_arm_joint_state',ARMS['left']),('right_arm_joint_state',ARMS['right']),
                           ('left_gripper_state',['Joint_Left_Gripper']),('right_gripper_state',['Joint_Right_Gripper'])]:
                if any(d[key].get('communication_lost',[])) or len(d[key]['position'])!=len(ns):return
                q.update(zip(ns,d[key]['position']))
            if all(math.isfinite(v) for v in q.values()):
                arm=(time.monotonic(),q);arm_history.append(arm)
        except (ValueError,KeyError,TypeError):pass
    def check_arm(expected):
        if arm is None or time.monotonic()-arm[0]>.5:raise ValueError('关节反馈过期，停止导航')
        allowed=set()
        if goal and goal.get('step_posture'):
            from .step_posture import permitted_navigation_joints
            allowed=permitted_navigation_joints(expected,arm[1],goal['step_posture'])
        if any(n not in expected or (n not in allowed and abs(v-expected[n])>(8. if n in ('Joint_Left_Gripper','Joint_Right_Gripper') else 3.)) for n,v in arm[1].items()):
            raise ValueError('手臂或身体已偏离运输姿态，停止导航')
    def map_received(m):
        nonlocal map_meta
        try:
            w,h=m.info.width,m.info.height
            grid=np.asarray(m.data,dtype=np.int16).reshape(h,w)
            pixels=np.full((h,w),38,dtype=np.uint8);pixels[grid==0]=230;pixels[grid>0]=0
            pixels=np.flipud(pixels);scale=min(1.,1600/w,1600/h)
            pixels=cv2.resize(pixels,(max(1,int(w*scale)),max(1,int(h*scale))),interpolation=cv2.INTER_NEAREST)
            temp=run/'map_next.png';cv2.imwrite(str(temp),pixels);temp.replace(run/'navigation_map.png')
            p=m.info.origin.position;q=m.info.origin.orientation
            map_meta=dict(width=w,height=h,resolution=m.info.resolution,x=p.x,y=p.y,
                          yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)))
        except (ValueError,ZeroDivisionError):pass
    def obstacle_received(m):
        nonlocal obstacle_meta,obstacle_time
        if m.header.frame_id!='map' or time.monotonic()-obstacle_time<.5:return
        info=m.metadata;w,h=info.size_x,info.size_y
        if not w or not h:return
        a=np.asarray(m.data,dtype=np.uint8).reshape(h,w)
        pixels=np.zeros((h,w,4),np.uint8);pixels[a==254]=[65,65,245,200]
        pixels=np.flipud(pixels);scale=min(1.,1600/w,1600/h)
        pixels=cv2.resize(pixels,(max(1,int(w*scale)),max(1,int(h*scale))),interpolation=cv2.INTER_NEAREST)
        temp=run/'obstacles_next.png';cv2.imwrite(str(temp),pixels);temp.replace(run/'navigation_obstacles.png')
        p=info.origin.position;q=info.origin.orientation
        obstacle_meta=dict(width=w,height=h,resolution=info.resolution,x=p.x,y=p.y,
            yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)))
        obstacle_time=time.monotonic()
    qos=QoSProfile(depth=10,durability=DurabilityPolicy.TRANSIENT_LOCAL)
    # A short-lived voice-only preference; expiry restores control-center behavior.
    proximity_publisher=node.create_publisher(Bool,'/grasp306/proximity_voice_enabled',1)
    last_voice_publish=-1e9
    from .voice_settings import read_proximity_voice
    initial_publisher=node.create_publisher(PoseWithCovarianceStamped,'/initialpose',qos)
    node.create_subscription(String,'/waypoints/navigation_status',status,qos)
    node.create_subscription(String,'/global_relocalization/status',localization,qos)
    node.create_subscription(PoseWithCovarianceStamped,'/initialpose',initial_pose,qos)
    node.create_subscription(PoseWithCovarianceStamped,'/amcl_pose',amcl_pose,qos)
    node.create_subscription(OccupancyGrid,'/map',map_received,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    node.create_subscription(Costmap,'/global_costmap/costmap_raw',obstacle_received,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    node.create_subscription(Odometry,'/topic_gv_wheel_odom_0_300',odometry,qos_profile_sensor_data)
    node.create_subscription(Bool,'/robot_navigation_0_300/is_navigating',vendor,10)
    node.create_subscription(String,'/topic_arm_whole_body_and_gripper_current_joints_status_0_300',joints,qos_profile_sensor_data)
    for topic in ('/navigate_to_pose/_action/status','/navigate_through_poses/_action/status','/dock_robot/_action/status'):
        node.create_subscription(GoalStatusArray,topic,lambda m,k=topic:actions.update({k:any(s.status in (1,2,3) for s in m.status_list)}),qos)
    def call(kind,request):
        if not clients[kind].service_is_ready():raise ValueError('请点击“启动导航”，等待后台就绪（'+kind+'服务未就绪）')
        return clients[kind].call_async(request)
    def busy():return nav_bool or any(actions.values()) or latest.get('state','idle') not in IDLE or latest.get('state')=='paused'
    def still():return odom is not None and time.monotonic()-odom[0]<.4 and odom[1]<.015 and odom[2]<.025
    def pose():
        nonlocal pose_reason
        try:
            t=tf.lookup_transform('map','base_link',rclpy.time.Time())
            now_stamp=node.get_clock().now().nanoseconds/1e9
            if now_stamp-t.header.stamp.sec-t.header.stamp.nanosec/1e9>1.:
                pose_reason='定位坐标TF已过期';return None
            p=t.transform.translation;q=t.transform.rotation
            current=[p.x,p.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))]
            if not all(math.isfinite(v) for v in current):pose_reason='定位坐标无效';return None
            if not localization_gate.check(current,now_stamp):
                pose_reason=localization_gate.error or '等待本轮定位确认（支持控制中心手动定位）';return None
            pose_reason='';return current
        except Exception as exc:pose_reason='未收到map到base_link坐标变换：'+str(exc)[:100];return None
    def publish_manual(values,selected_map):
        nonlocal manual_confirmation,last_nomotion
        if stopping or busy() or not still() or selected_map!=active_map():raise ValueError('手动定位确认期间状态改变，未发布位置，请停稳后重试')
        m=PoseWithCovarianceStamped();m.header.frame_id='map';m.header.stamp=node.get_clock().now().to_msg()
        m.pose.pose.position.x=values[0];m.pose.pose.position.y=values[1]
        m.pose.pose.orientation.z=math.sin(values[2]/2);m.pose.pose.orientation.w=math.cos(values[2]/2)
        m.pose.covariance[0]=.01;m.pose.covariance[7]=.01;m.pose.covariance[35]=math.radians(10)**2
        stamp=m.header.stamp.sec+m.header.stamp.nanosec/1e9
        localization_gate.request(stamp);localization_gate.initial(stamp,values)
        initial_publisher.publish(m);manual_confirmation=time.monotonic();last_nomotion=0.
        emit('nav_message',text='手动定位已提交，正在更新定位；不会移动机器人')
    try:
        while running and rclpy.ok():
            rclpy.spin_once(node,timeout_sec=.025);now=time.monotonic()
            if not (run/'heartbeat').exists() or time.time()-(run/'heartbeat').stat().st_mtime>1.5:
                stopping=True
                if goal:cancel_reason='抓取界面心跳中断，已请求取消本次导航'
            if goal and (run/'stop').exists():cancel_reason=cancel_reason or '用户停止演示'
            if manual_confirmation is not None:
                pose()  # Confirm fresh AMCL/TF before checking the bounded wait.
                if not localization_gate.pending:
                    manual_confirmation=None
                elif now-manual_confirmation>5.:
                    localization_gate.fail('手动定位更新超时，请检查雷达/AMCL后重新拖动定位')
                    manual_confirmation=None;emit('nav_error',text=localization_gate.error)
                elif now-last_nomotion>=.25 and (nomotion_future is None or nomotion_future.done()):
                    if clients['nomotion'].service_is_ready():
                        nomotion_future=clients['nomotion'].call_async(Empty.Request());last_nomotion=now

            if waiting_navigation and (stopping or (run/'stop').exists()):
                cmd=waiting_navigation[0];waiting_navigation=None
                emit('nav_done',success=False,leg=cmd['leg'],run_id=cmd['run_id'],text='等待停稳期间收到停止请求')
            if not stopping and now-last_voice_publish>=.5:
                proximity_publisher.publish(Bool(data=read_proximity_voice(run/'voice_settings.json')))
                last_voice_publish=now
            if stopping and not goal and pending is None and not runtime.starting:break
            try:
                try:cmd=commands.get_nowait()
                except queue.Empty:
                    if waiting_navigation and now>=waiting_navigation[2]:
                        cmd=waiting_navigation[0]
                    else:raise
                action=cmd['action']
                if waiting_navigation and cmd is not waiting_navigation[0] and action not in ('stop','quit'):
                    raise ValueError('正在等待本次导航出发确认，请勿重复提交')
                if action in ('stop','quit'):
                    if waiting_navigation:
                        queued=waiting_navigation[0];waiting_navigation=None
                        emit('nav_done',success=False,leg=queued['leg'],run_id=queued['run_id'],text='等待停稳期间收到停止请求')
                    if goal:cancel_reason='用户停止演示'
                    if action=='quit':stopping=True
                elif action=='start_navigation':
                    if goal or pending or map_op or busy() or not still():raise ValueError('请停止任务并等待底盘停稳后启动导航')
                    if clients['navigate'].service_is_ready():emit('nav_message',text='已连接现有导航后台，无需重复启动')
                    else:runtime.start(active_map())
                elif action=='navigate':
                    if map_fault:raise ValueError('地图切换尚未完整确认，请重新加载地图后再开始')
                    if goal or pending or map_op or busy():raise ValueError('其他导航或地图操作仍在进行，未接管')
                    if not still() or pose() is None:raise ValueError('底盘尚未停稳或定位未就绪')
                    if redis_client.get('shared_params:0:300:is_navigating') in (b'true',b'True'):
                        raise ValueError('底盘正被其他控制端使用')
                    if cmd['map']!=active_map() or cmd['name'] not in names:raise ValueError('地图或航点已改变，请重新选择')
                    step_posture=None
                    if cmd.get('workflow') is True:
                        from .task_flow import navigation_pose
                        context=json.loads((run/'task_flow.json').read_text())
                        record={'q':navigation_pose(cmd,context,arm_history,time.monotonic())}
                        # Read only the validated snapshot, never an arbitrary
                        # command-supplied exemption from joint holding checks.
                        step_posture=context['record']['steps'][int(cmd['leg'].split(':')[1])].get('posture')
                    else:
                        record=json.loads((run/('prepare_result.json' if cmd['leg']=='to_shelf' else 'demo_grasp_result.json')).read_text())
                        if record.get('status') not in ('prepared','completed'):raise ValueError('前一机械臂阶段未完成')
                    check_arm(record['q'])
                    req=NavigateToWaypoint.Request();req.name=cmd['name']
                    goal={**cmd,'step_posture':step_posture,'expected_q':record['q'],'progress':GoalProgress(req.name,node.get_clock().now().nanoseconds/1e9)}
                    waiting_navigation=None
                    ack=False;arrived=False;stable=None;cancel_reason=None;cancel_sent=False;sent_at=now
                    try:pending=call('navigate',req)
                    except Exception:goal=None;raise
                    pending_kind='navigate';pending_time=now
                elif action=='map':
                    if goal or pending or busy() or not still():raise ValueError('请停止导航并停稳后再更换地图')
                    path=str(Path(cmd['path']).resolve())
                    if path not in [m['path'] for m in map_catalog()]:raise ValueError('地图不在控制中心地图库')
                    if not clients['switch'].service_is_ready():raise ValueError('航点切换服务未就绪')
                    map_op=dict(new=path,old=active_map());req=LoadMap.Request();req.map_url=path
                    localization_gate.request(node.get_clock().now().nanoseconds/1e9)
                    pending=call('load',req);pending_kind='load';pending_time=now
                elif action=='initial_pose':
                    if goal or pending or map_op or busy() or not still():raise ValueError('请停止任务并停稳后手动重定位')
                    if not clients['nomotion'].service_is_ready() or not map_meta:raise ValueError('AMCL定位后台或地图尚未就绪')
                    if cmd.get('map')!=active_map():raise ValueError('地图已改变，请在新地图上重新选择位置')
                    values=[float(cmd[k]) for k in ('x','y','yaw')]
                    if not all(math.isfinite(v) for v in values):raise ValueError('定位坐标必须是有限数值')
                    from .navigation_map import world_to_pixel
                    px,py=world_to_pixel(map_meta,*values[:2])
                    if not 0<=px<map_meta['width'] or not 0<=py<map_meta['height']:raise ValueError('定位点超出地图范围')
                    if clients['cancel_localization'].service_is_ready():
                        manual_pending=(values,cmd['map'])
                        pending=call('cancel_localization',Trigger.Request());pending_kind='manual_cancel';pending_time=now
                    elif clients['relocalize'].service_is_ready():
                        raise ValueError('定位后台版本缺少手动接管接口，请重启导航后台加载已更新版本')
                    else:
                        publish_manual(values,cmd['map'])
                elif action=='relocalize':
                    if goal or pending or busy() or not still():raise ValueError('请停止导航并停稳后重定位')
                    if not clients['relocalize'].service_is_ready():raise ValueError('请先启动导航定位后台')
                    localization_gate.request(node.get_clock().now().nanoseconds/1e9)
                    pending=call('relocalize',Trigger.Request());pending_kind='relocalize';pending_time=now
            except queue.Empty:pass
            except NavigationSettling as exc:
                if waiting_navigation is None:
                    waiting_navigation=(cmd,now+3.,now+.05)
                    emit('nav_message',text='收回动作刚结束，等待新反馈确认停稳后自动出发')
                elif now<waiting_navigation[1]:
                    waiting_navigation=(cmd,waiting_navigation[1],now+.05)
                else:
                    waiting_navigation=None
                    emit('nav_error',text='等待停稳超时：'+str(exc),leg=cmd.get('leg'),run_id=cmd.get('run_id'))
            except Exception as exc:
                if waiting_navigation and cmd is waiting_navigation[0]:waiting_navigation=None
                if cmd.get('action')=='map' and pending is None:map_op=None
                emit('nav_error',text=str(exc),leg=cmd.get('leg'),run_id=cmd.get('run_id'))
            if pending is not None and pending.done():
                kind=pending_kind;future=pending;pending=None
                try:
                    response=future.result()
                    if kind=='navigate':
                        if not response.success:raise ValueError(response.message)
                        ack=True
                        emit('nav_started',run_id=goal['run_id'],leg=goal['leg'])
                    elif kind=='load':
                        if response.result!=0:raise ValueError('地图加载失败：'+str(response.result))
                        req=SwitchWaypointMap.Request();req.map_yaml=map_op['new']
                        pending=call('switch',req);pending_kind='switch';pending_time=now
                    elif kind=='switch':
                        if not response.success:raise ValueError(response.message)
                        write_active(map_op['new'],response.waypoints_file);names=list(response.names);map_op=None;map_fault=False
                        pending=call('relocalize',Trigger.Request());pending_kind='relocalize';pending_time=now
                        emit('nav_message',text='地图和航点库已同步，正在重新定位')
                    elif kind=='rollback':
                        if response.result!=0:raise ValueError('恢复原地图失败，请在控制中心检查')
                        req=SwitchWaypointMap.Request();req.map_yaml=map_op['old']
                        pending=call('switch',req);pending_kind='rollback_switch';pending_time=now
                    elif kind=='rollback_switch':
                        if not response.success:raise ValueError('恢复原航点库失败，请在控制中心检查')
                        names=list(response.names);map_op=None;map_fault=False
                        emit('nav_error',text='切换未完成，已恢复原地图和航点库，请重试')
                    elif kind=='cancel':
                        if not response.success:emit('nav_error',text='取消导航未确认：'+response.message)
                    elif kind=='manual_cancel':
                        values,selected_map=manual_pending;manual_pending=None
                        if not response.success:raise ValueError('全局搜索尚未停止，未覆盖定位：'+response.message)
                        publish_manual(values,selected_map)
                    elif kind=='relocalize':
                        if not response.success:raise ValueError(response.message)
                        emit('nav_message',text=response.message)
                except Exception as exc:
                    if goal:cancel_reason=str(exc)
                    if map_op and kind=='switch':
                        map_fault=True
                        req=LoadMap.Request();req.map_url=map_op['old']
                        try:pending=call('load',req);pending_kind='rollback';pending_time=now
                        except Exception:map_op=None
                    elif map_op:
                        map_fault=True;map_op=None
                    emit('nav_error',text=str(exc))
            if pending is not None and now-pending_time>10:
                # Do not forget a navigation request whose response might arrive late.
                if goal:cancel_reason=cancel_reason or '导航请求响应超时，等待取消确认'
                elif now-pending_time>20:
                    if map_op:map_fault=True
                    emit('nav_error',text='导航服务响应超时，请检查控制中心');pending=None;map_op=None
            if goal:
                if goal.get('workflow') and ((run/'task_pause').exists() or pause.confirmed or pause.future):sent_at+=max(0,now-last_pause_tick)
                wanted=goal.get('workflow',False) and (run/'task_pause').exists()
                if ack and not arrived and not cancel_reason and pending is None:
                    try:
                        changed=pause.tick(wanted,now,lambda key:call(key,Trigger.Request()))
                        if changed is not None:
                            if not changed:pause.last_resume_stamp=node.get_clock().now().nanoseconds/1e9
                            emit('nav_pause_state',paused=changed,leg=goal['leg'],run_id=goal['run_id'])
                    except Exception as exc:cancel_reason='导航暂停/继续不可用：'+str(exc)
                try:check_arm(goal['expected_q'])
                except ValueError as exc:cancel_reason=cancel_reason or str(exc)
                if now-sent_at>600:cancel_reason='导航超过10分钟，已停止本轮演示'
                try:
                    if active_map()!=goal['map']:cancel_reason='当前地图改变，演示停止'
                except Exception:cancel_reason='当前地图配置不可读，演示停止'
                if cancel_reason and pending is None and not cancel_sent:
                    # Never cancel a replacement target belonging to another operator.
                    if latest.get('active_waypoint') not in (None,'',goal['name']):
                        emit('nav_done',success=False,leg=goal['leg'],run_id=goal['run_id'],text=cancel_reason);goal=None
                    else:
                        try:
                            pending=call('cancel',Trigger.Request());pending_kind='cancel';pending_time=now
                            cancel_sent=True;cancel_time=now
                        except Exception as exc:emit('nav_error',text=str(exc));time.sleep(.1)
                if goal and cancel_sent and pending is None and still() and not any(actions.values()):
                    emit('nav_done',success=False,leg=goal['leg'],run_id=goal['run_id'],text=cancel_reason);goal=None
                elif goal and ack and arrived and not cancel_reason:
                    stable=(stable if stable is not None else now) if still() else None
                    if stable is not None and now-stable>.5:
                        emit('nav_done',success=True,leg=goal['leg'],run_id=goal['run_id'],name=goal['name'],map=goal['map'])
                        goal=None
            if goal is None:pause=NavigationPause()
            last_pause_tick=now
            if not pending and not map_op and now-last_list>2 and clients['list'].service_is_ready() and list_future is None:
                list_future=clients['list'].call_async(ListWaypoints.Request());last_list=now
            if list_future is not None and list_future.done():
                try:
                    response=list_future.result()
                    if response.success:names=list(response.names)
                except Exception:pass
                list_future=None
            if now-last_publish>.5:
                last_publish=now
                if runtime.process is not None and runtime.process.poll() is not None:
                    runtime.error='导航后台已退出，请查看 navigation_backend.log 后重试'
                try:current_map=active_map()
                except Exception:current_map=''
                current_pose=pose()
                snapshot=dict(maps=map_catalog(),active_map=current_map,names=names,
                     library=waypoint_poses(None),waypoints=waypoint_poses(names),pose=current_pose,meta=map_meta,
                     ready=clients['navigate'].service_is_ready() and not map_fault,busy=bool(goal or pending or map_op or waiting_navigation or busy()),
                     localization_ready=clients['relocalize'].service_is_ready(),localization=localization_status,
                     manual_localization_ready=bool(map_meta and clients['nomotion'].service_is_ready()),
                     startup_error=runtime.error,
                     starting=runtime.starting or bool(runtime.started and not runtime.error and now-runtime.started<45 and not clients['navigate'].service_is_ready()),
                     obstacle_meta=obstacle_meta if now-obstacle_time<3 else None,
                     service_ready=clients['navigate'].service_is_ready(),map_fault=map_fault,
                     pose_reason=pose_reason,localization_pending=localization_gate.pending,
                     pending_pose=localization_gate.requested_pose if localization_gate.pending else None,
                     snapshot_time=time.time(),navigation=latest)
                emit('nav_snapshot',**snapshot)
                from .core import atomic_json
                atomic_json(run/'navigation_state.json',snapshot)
    finally:
        if goal and latest.get('active_waypoint') in (None,'',goal['name']) and clients['cancel'].service_is_ready():
            try:
                future=clients['cancel'].call_async(Trigger.Request())
                deadline=time.monotonic()+3
                while not future.done() and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.05)
            except Exception:pass
        redis_client.close();node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
