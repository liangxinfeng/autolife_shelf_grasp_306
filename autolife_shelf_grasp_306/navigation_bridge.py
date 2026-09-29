"""ROS navigation client in its own process. Never publishes chassis velocity."""
import argparse,json,math,queue,sys,threading,time
from pathlib import Path
import numpy as np
from .navigation_logic import (active_map,map_catalog,write_active,waypoint_poses,
                               GoalProgress,LocalizationGate,IDLE)


def emit(event,**data):print(json.dumps(dict(event=event,**data),ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    run=Path(args.run)
    import cv2,rclpy,tf2_ros,redis
    from rclpy.qos import QoSProfile,DurabilityPolicy,qos_profile_sensor_data
    from nav_msgs.msg import OccupancyGrid,Odometry
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from std_msgs.msg import String,Bool
    from std_srvs.srv import Trigger
    from action_msgs.msg import GoalStatusArray
    from nav2_msgs.srv import LoadMap
    from control_center_navigation.srv import ListWaypoints,NavigateToWaypoint,SwitchWaypointMap
    rclpy.init();node=rclpy.create_node('grasp306_navigation_client')
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
        ('cancel',Trigger,'/cancel_navigation'),('load',LoadMap,'/map_server/load_map'),
        ('switch',SwitchWaypointMap,'/switch_waypoint_map'),
        ('relocalize',Trigger,'/cartographer_global_relocalization/relocalize')]}
    latest={};odom=None;odom_stamp=None;nav_bool=False;actions={};names=[];map_meta=None;arm=None
    goal=None;ack=False;arrived=False;stable=None;cancel_reason=None;cancel_sent=False
    sent_at=0.;last_publish=0.;last_list=0.;list_future=None;map_op=None;map_fault=False
    pending=None;pending_kind=None;pending_time=0.;stopping=False;cancel_time=0.
    localization_gate=LocalizationGate();localization_attempt=-1;required_attempt=-1;pose_reason='等待定位坐标'
    redis_client=redis.Redis(socket_timeout=.15,socket_connect_timeout=.15)
    def status(m):
        nonlocal latest,arrived,cancel_reason
        try:
            data=json.loads(m.data);latest=data
            localization_gate.navigation(data)
            if goal and not cancel_reason:
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
        nonlocal localization_attempt
        try:
            d=json.loads(m.data);localization_attempt=int(d.get('attempt',-1))
            if (localization_gate.pending and localization_attempt>required_attempt and d.get('state')=='succeeded'
                    and 'disabled' not in d.get('detail','')):localization_gate.pending=False
        except (ValueError,TypeError):pass
    def initial_pose(m):
        if m.header.frame_id=='map':localization_gate.initial(m.header.stamp.sec+m.header.stamp.nanosec/1e9)
    def amcl_pose(m):
        if m.header.frame_id!='map':return
        p=m.pose.pose.position;q=m.pose.pose.orientation;c=m.pose.covariance
        localization_gate.amcl(m.header.stamp.sec+m.header.stamp.nanosec/1e9,
            [p.x,p.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))],[c[0],c[7],c[35]])
    def joints(m):
        nonlocal arm
        from autolife_handeye_calibration_307.core import BODY,NECK,ARMS
        try:
            d=json.loads(m.data);q={}
            for key,ns in [('leg_waist_joint_state',BODY),('neck_joint_state',NECK),
                           ('left_arm_joint_state',ARMS['left']),('right_arm_joint_state',ARMS['right']),
                           ('left_gripper_state',['Joint_Left_Gripper'])]:
                if any(d[key].get('communication_lost',[])) or len(d[key]['position'])!=len(ns):return
                q.update(zip(ns,d[key]['position']))
            if all(math.isfinite(v) for v in q.values()):arm=(time.monotonic(),q)
        except (ValueError,KeyError,TypeError):pass
    def check_arm(expected):
        if arm is None or time.monotonic()-arm[0]>.5:raise ValueError('关节反馈过期，停止导航')
        if any(n not in expected or abs(v-expected[n])>(8. if n=='Joint_Left_Gripper' else 3.) for n,v in arm[1].items()):
            raise ValueError('手臂或身体已偏离运输姿态，停止导航')
    def map_received(m):
        nonlocal map_meta
        try:
            w,h=m.info.width,m.info.height
            grid=np.asarray(m.data,dtype=np.int16).reshape(h,w)
            pixels=np.full((h,w),38,dtype=np.uint8);pixels[grid==0]=230;pixels[grid>0]=0
            pixels=np.flipud(pixels);scale=min(1.,640/w,360/h)
            pixels=cv2.resize(pixels,(max(1,int(w*scale)),max(1,int(h*scale))),interpolation=cv2.INTER_NEAREST)
            temp=run/'map_next.png';cv2.imwrite(str(temp),pixels);temp.replace(run/'navigation_map.png')
            p=m.info.origin.position;q=m.info.origin.orientation
            map_meta=dict(width=w,height=h,resolution=m.info.resolution,x=p.x,y=p.y,
                          yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)))
        except (ValueError,ZeroDivisionError):pass
    qos=QoSProfile(depth=10,durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(String,'/waypoints/navigation_status',status,qos)
    node.create_subscription(String,'/global_relocalization/status',localization,qos)
    node.create_subscription(PoseWithCovarianceStamped,'/initialpose',initial_pose,10)
    node.create_subscription(PoseWithCovarianceStamped,'/amcl_pose',amcl_pose,10)
    node.create_subscription(OccupancyGrid,'/map',map_received,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    node.create_subscription(Odometry,'/topic_gv_wheel_odom_0_307',odometry,qos_profile_sensor_data)
    node.create_subscription(Bool,'/robot_navigation_0_307/is_navigating',vendor,10)
    node.create_subscription(String,'/topic_arm_whole_body_and_gripper_current_joints_status_0_307',joints,qos_profile_sensor_data)
    for topic in ('/navigate_to_pose/_action/status','/navigate_through_poses/_action/status','/dock_robot/_action/status'):
        node.create_subscription(GoalStatusArray,topic,lambda m,k=topic:actions.update({k:any(s.status in (1,2,3) for s in m.status_list)}),qos)
    def call(kind,request):
        if not clients[kind].service_is_ready():raise ValueError('请先在机器人控制中心启动导航模块（'+kind+'服务未就绪）')
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
                pose_reason='等待本轮定位确认（支持控制中心手动定位）';return None
            pose_reason='';return current
        except Exception as exc:pose_reason='未收到map到base_link坐标变换：'+str(exc)[:100];return None
    try:
        while running and rclpy.ok():
            rclpy.spin_once(node,timeout_sec=.025);now=time.monotonic()
            if not (run/'heartbeat').exists() or time.time()-(run/'heartbeat').stat().st_mtime>1.5:
                stopping=True
                if goal:cancel_reason='抓取界面心跳中断，已请求取消本次导航'
            if goal and (run/'stop').exists():cancel_reason=cancel_reason or '用户停止演示'
            if stopping and not goal and pending is None:break
            try:
                cmd=commands.get_nowait()
                action=cmd['action']
                if action in ('stop','quit'):
                    if goal:cancel_reason='用户停止演示'
                    if action=='quit':stopping=True
                elif action=='navigate':
                    if map_fault:raise ValueError('地图切换尚未完整确认，请重新加载地图后再开始')
                    if goal or pending or map_op or busy():raise ValueError('其他导航或地图操作仍在进行，未接管')
                    if not still() or pose() is None:raise ValueError('底盘尚未停稳或定位未就绪')
                    if redis_client.get('shared_params:0:300:is_navigating') in (b'true',b'True'):
                        raise ValueError('底盘正被其他控制端使用')
                    if cmd['map']!=active_map() or cmd['name'] not in names:raise ValueError('地图或航点已改变，请重新选择')
                    record=json.loads((run/('prepare_result.json' if cmd['leg']=='to_shelf' else 'demo_grasp_result.json')).read_text())
                    if record.get('status') not in ('prepared','completed'):raise ValueError('前一机械臂阶段未完成')
                    check_arm(record['q'])
                    req=NavigateToWaypoint.Request();req.name=cmd['name']
                    goal={**cmd,'expected_q':record['q'],'progress':GoalProgress(req.name,node.get_clock().now().nanoseconds/1e9)}
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
                    localization_gate.request(node.get_clock().now().nanoseconds/1e9);required_attempt=localization_attempt
                    pending=call('load',req);pending_kind='load';pending_time=now
                elif action=='relocalize':
                    if goal or pending or busy() or not still():raise ValueError('请停止导航并停稳后重定位')
                    localization_gate.request(node.get_clock().now().nanoseconds/1e9);required_attempt=localization_attempt
                    pending=call('relocalize',Trigger.Request());pending_kind='relocalize';pending_time=now
            except queue.Empty:pass
            except Exception as exc:
                if cmd.get('action')=='map' and pending is None:map_op=None
                emit('nav_error',text=str(exc),leg=cmd.get('leg'),run_id=cmd.get('run_id'))
            if pending is not None and pending.done():
                kind=pending_kind;future=pending;pending=None
                try:
                    response=future.result()
                    if kind=='navigate':
                        if not response.success:raise ValueError(response.message)
                        ack=True
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
                    elif kind=='relocalize':emit('nav_message',text=response.message)
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
                try:current_map=active_map()
                except Exception:current_map=''
                current_pose=pose()
                snapshot=dict(maps=map_catalog(),active_map=current_map,names=names,
                     library=waypoint_poses(None),waypoints=waypoint_poses(names),pose=current_pose,meta=map_meta,
                     ready=clients['navigate'].service_is_ready() and not map_fault,busy=bool(goal or pending or map_op),
                     service_ready=clients['navigate'].service_is_ready(),map_fault=map_fault,
                     pose_reason=pose_reason,localization_pending=localization_gate.pending,
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
