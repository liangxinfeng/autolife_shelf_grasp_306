"""ROS integration on isolated domain 101, dummy services only, no cmd_vel."""
import os,json,queue,threading,time,tempfile,subprocess
from pathlib import Path
os.environ['ROS_DOMAIN_ID']='101'
os.environ['RMW_IMPLEMENTATION']='rmw_cyclonedds_cpp'
os.environ['CYCLONEDDS_URI']='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
import rclpy
from rclpy.qos import QoSProfile,DurabilityPolicy
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from std_srvs.srv import Trigger
from control_center_navigation.srv import ListWaypoints,NavigateToWaypoint
from autolife_handeye_calibration_306.core import BODY,NECK,ARMS
from autolife_shelf_grasp_306.navigation_logic import active_map
rclpy.init();node=rclpy.create_node('dummy_navigation_test');tf=TransformBroadcaster(node)
status=node.create_publisher(String,'/waypoints/navigation_status',QoSProfile(depth=10,durability=DurabilityPolicy.TRANSIENT_LOCAL))
od=node.create_publisher(Odometry,'/topic_gv_wheel_odom_0_300',10)
ar=node.create_publisher(String,'/topic_arm_whole_body_and_gripper_current_joints_status_0_300',10)
run=Path(tempfile.mkdtemp(prefix='grasp_nav_isolated_'));(run/'heartbeat').touch()
q={n:0. for n in BODY+NECK+ARMS['left']+ARMS['right']+['Joint_Left_Gripper']}
(run/'prepare_result.json').write_text(json.dumps(dict(status='prepared',q=q)))
(run/'demo_grasp_result.json').write_text(json.dumps(dict(status='completed',q=q)))
target=None;end=0;cancel_count=0
def send_status(state,name):
    status.publish(String(data=json.dumps(dict(state=state,active_waypoint=name,stamp=node.get_clock().now().nanoseconds/1e9))))
def listing(req,res):res.success=True;res.names=['shelf','front'];return res
def navigate(req,res):
    global target,end
    target=req.name;end=time.monotonic()+(.7 if target=='shelf' else 20)
    send_status('started',target);res.success=True;return res
def cancel(req,res):
    global target,cancel_count
    cancel_count+=1;send_status('canceled',target);target=None;res.success=True;return res
node.create_service(ListWaypoints,'/list_waypoints',listing)
node.create_service(NavigateToWaypoint,'/navigate_to_waypoint',navigate)
node.create_service(Trigger,'/cancel_navigation',cancel)
def tick():
    global target
    now=node.get_clock().now().to_msg()
    t=TransformStamped();t.header.frame_id='map';t.child_frame_id='base_link';t.header.stamp=now;t.transform.rotation.w=1.;tf.sendTransform(t)
    o=Odometry();o.header.stamp=now;o.child_frame_id='base_link';o.pose.pose.orientation.w=1.;od.publish(o)
    ar.publish(String(data=json.dumps({key:dict(position=[0.]*len(ns),communication_lost=[False]*len(ns)) for key,ns in [
        ('leg_waist_joint_state',BODY),('neck_joint_state',NECK),('left_arm_joint_state',ARMS['left']),
        ('right_arm_joint_state',ARMS['right']),('left_gripper_state',['Joint_Left_Gripper'])]})))
    if target and time.monotonic()>end:send_status('succeeded',target);target=None
node.create_timer(.04,tick)
proc=subprocess.Popen(['/usr/bin/python3','-m','autolife_shelf_grasp_306.navigation_bridge','--run',str(run)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
events=queue.Queue()
def read():
    for line in proc.stdout:
        try:events.put(json.loads(line))
        except ValueError:pass
threading.Thread(target=read,daemon=True).start()
def command(action,**kw):proc.stdin.write(json.dumps(dict(action=action,**kw))+'\n');proc.stdin.flush()
def wait(predicate,seconds=8):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        (run/'heartbeat').touch();rclpy.spin_once(node,timeout_sec=.02)
        try:
            e=events.get_nowait()
            if e.get('event')=='nav_error':raise AssertionError(e)
            if predicate(e):return e
        except queue.Empty:pass
    raise AssertionError('Timed out; child exit='+str(proc.poll()))
try:
    wait(lambda e:e.get('event')=='nav_snapshot' and e.get('ready') and e.get('pose') and e.get('names'))
    command('navigate',run_id='test',leg='to_shelf',name='shelf',map=active_map())
    result=wait(lambda e:e.get('event')=='nav_done');assert result['success'],result
    command('navigate',run_id='test',leg='to_front',name='front',map=active_map())
    wait(lambda e:e.get('event')=='nav_snapshot' and e['navigation'].get('active_waypoint')=='front')
    command('stop')
    result=wait(lambda e:e.get('event')=='nav_done');assert not result['success'] and cancel_count==1,result
    command('quit')
    deadline=time.monotonic()+3
    while proc.poll() is None and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.02)
    assert proc.poll()==0
    print('ISOLATED_NAV_PASS: own-goal arrival, stationary handoff, cancellation; no real-robot ROS domain used')
finally:
    if proc.poll() is None:proc.terminate();proc.wait(timeout=3)
    node.destroy_node();rclpy.shutdown()
