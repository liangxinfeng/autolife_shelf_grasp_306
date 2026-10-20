"""Full navigation client integration against loopback-only dummy ROS services."""
import os,json,queue,threading,time,tempfile,subprocess
from pathlib import Path
os.environ['ROS_DOMAIN_ID']='217'
os.environ['RMW_IMPLEMENTATION']='rmw_cyclonedds_cpp'
os.environ['CYCLONEDDS_URI']='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
import rclpy
from rclpy.qos import QoSProfile,DurabilityPolicy
from std_msgs.msg import String
from nav_msgs.msg import Odometry,OccupancyGrid
from geometry_msgs.msg import TransformStamped,PoseWithCovarianceStamped
from tf2_ros import TransformBroadcaster
from std_srvs.srv import Trigger,Empty
from control_center_navigation.srv import ListWaypoints,NavigateToWaypoint
from autolife_handeye_calibration_306.core import BODY,NECK,ARMS
from autolife_shelf_grasp_306.navigation_logic import active_map
rclpy.init();node=rclpy.create_node('dummy_pause_localization_test');tf=TransformBroadcaster(node)
latched=QoSProfile(depth=10,durability=DurabilityPolicy.TRANSIENT_LOCAL)
status=node.create_publisher(String,'/waypoints/navigation_status',latched)
amcl=node.create_publisher(PoseWithCovarianceStamped,'/amcl_pose',latched)
mp=node.create_publisher(OccupancyGrid,'/map',latched)
od=node.create_publisher(Odometry,'/topic_gv_wheel_odom_0_300',10)
ar=node.create_publisher(String,'/topic_arm_whole_body_and_gripper_current_joints_status_0_300',10)
run=Path(tempfile.mkdtemp(prefix='nav_pause_isolated_'));(run/'heartbeat').touch()
map_path=active_map();context=dict(run_id='isolated',map=map_path,record=dict(steps=[dict(kind='navigate',target='test_goal')]))
(run/'task_flow.json').write_text(json.dumps(context))
target=None;paused=False;end=float('inf');calls=[];stop_checks=0;location=None;amcl_due=None
def send_status(state):
    status.publish(String(data=json.dumps(dict(state=state,active_waypoint=target,stamp=node.get_clock().now().nanoseconds/1e9))))
def listing(req,res):res.success=True;res.names=['test_goal'];return res
def navigate(req,res):
    global target
    calls.append('navigate');target=req.name;send_status('started');res.success=True;return res
def pause(req,res):
    global paused
    calls.append('pause');paused=True;send_status('paused');res.success=True;return res
def ready(req,res):
    global stop_checks
    calls.append('stop_ready');stop_checks+=1
    res.success=stop_checks>=4;res.message='等待新鲜底盘反馈并连续停稳1秒';return res
def resume(req,res):
    global paused,end
    calls.append('resume');assert stop_checks>=4
    paused=False;end=time.monotonic()+1.;send_status('started');res.success=True;return res
def ok(req,res):res.success=True;return res
def initial(m):
    global location,amcl_due
    location=m;amcl_due=time.monotonic()+.8
node.create_service(ListWaypoints,'/list_waypoints',listing)
node.create_service(NavigateToWaypoint,'/navigate_to_waypoint',navigate)
for name,callback in [('pause_navigation',pause),('resume_navigation',resume),('navigation_stop_ready',ready),('cancel_navigation',ok),('cartographer_global_relocalization/relocalize',ok),('cartographer_global_relocalization/cancel',ok)]:node.create_service(Trigger,'/'+name,callback)
node.create_service(Empty,'/request_nomotion_update',lambda req,res:res)
node.create_subscription(PoseWithCovarianceStamped,'/initialpose',initial,latched)
def tick():
    global target,amcl_due
    stamp=node.get_clock().now().to_msg()
    t=TransformStamped();t.header.frame_id='map';t.child_frame_id='base_link';t.header.stamp=stamp;t.transform.rotation.w=1.
    if location and amcl_due is None:t.transform.translation.x=location.pose.pose.position.x;t.transform.translation.y=location.pose.pose.position.y
    tf.sendTransform(t)
    o=Odometry();o.header.stamp=stamp;o.pose.pose.orientation.w=1.;od.publish(o)
    ar.publish(String(data=json.dumps({key:dict(position=[0.]*len(ns),communication_lost=[False]*len(ns)) for key,ns in [
        ('leg_waist_joint_state',BODY),('neck_joint_state',NECK),('left_arm_joint_state',ARMS['left']),('right_arm_joint_state',ARMS['right']),('left_gripper_state',['Joint_Left_Gripper']),('right_gripper_state',['Joint_Right_Gripper'])]})))
    if amcl_due and time.monotonic()>=amcl_due:
        location.header.stamp=stamp;amcl.publish(location);amcl_due=None
    if target and not paused and time.monotonic()>end:send_status('succeeded');target=None
node.create_timer(.04,tick)
m=OccupancyGrid();m.header.frame_id='map';m.info.width=100;m.info.height=100;m.info.resolution=.1;m.info.origin.orientation.w=1.;m.data=[0]*10000;mp.publish(m)
code='import redis; from types import SimpleNamespace; redis.Redis=lambda **kw:SimpleNamespace(get=lambda key:None,close=lambda:None); from autolife_shelf_grasp_306.navigation_bridge import main; main()'
proc=subprocess.Popen(['/usr/bin/python3','-c',code,'--run',str(run)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
events=queue.Queue()
def read():
    for line in proc.stdout:
        try:events.put(json.loads(line))
        except ValueError:pass
threading.Thread(target=read,daemon=True).start()
def command(action,**kw):proc.stdin.write(json.dumps(dict(action=action,**kw))+'\n');proc.stdin.flush()
def wait(predicate,seconds=12):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        (run/'heartbeat').touch();rclpy.spin_once(node,timeout_sec=.02)
        try:
            e=events.get_nowait()
            if e.get('event')=='nav_error' or (e.get('event')=='nav_done' and not e['success']):raise AssertionError(e)
            if predicate(e):return e
        except queue.Empty:pass
    raise AssertionError('Timeout: '+str(proc.poll())+' calls='+str(calls))
try:
    wait(lambda e:e.get('event')=='nav_snapshot' and e.get('ready') and e.get('pose') and e.get('meta') and e.get('names'))
    command('initial_pose',x=1.,y=2.,yaw=0.,map=map_path)
    marker=wait(lambda e:e.get('event')=='nav_snapshot' and e.get('pending_pose'))
    assert marker['pose'] is None and marker['pending_pose']==[1.,2.,0.]
    confirmed=wait(lambda e:e.get('event')=='nav_snapshot' and e.get('pose') and not e.get('localization_pending'))
    assert confirmed['pose'][:2]==[1.,2.]
    command('navigate',workflow=True,run_id='isolated',leg='isolated:0',name='test_goal',map=map_path)
    wait(lambda e:e.get('event')=='nav_started')
    (run/'task_pause').touch();wait(lambda e:e.get('event')=='nav_pause_state' and e['paused'])
    (run/'task_pause').unlink();wait(lambda e:e.get('event')=='nav_pause_state' and not e['paused'])
    result=wait(lambda e:e.get('event')=='nav_done');assert result['success']
    assert calls.count('navigate')==1 and calls.count('resume')==1 and calls.count('stop_ready')==4,calls
    command('quit')
    deadline=time.monotonic()+3
    while proc.poll() is None and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.02)
    assert proc.poll()==0
finally:
    if proc.poll() is None:proc.terminate();proc.wait(timeout=3)
    node.destroy_node();rclpy.shutdown()
print('ISOLATED_PASS: manual pending marker -> fresh AMCL/TF; pause -> settling retries -> same goal -> arrival',calls)
