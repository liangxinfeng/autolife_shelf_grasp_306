"""Real recorder process, synthetic state only, isolated ROS domain 87."""
import os
os.environ['ROS_DOMAIN_ID']='87'
import json,subprocess,sys,tempfile,time,uuid
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from test_teaching import pose,payload
from autolife_shelf_grasp_306.teaching_library import TeachingLibrary
from openarmx_teleop_vr_navigation_306.teaching_ipc import request,discovery_path

root=Path(tempfile.mkdtemp(prefix='capture-service-'));library=TeachingLibrary(root)
log=(root/'service.log').open('w')
p=subprocess.Popen([sys.executable,'-m','autolife_shelf_grasp_306.teaching_service','--robot-id','345','--library',str(root)],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
rclpy.init();node=Node('teaching_synthetic_feedback_test')
pub=node.create_publisher(String,'/topic_arm_whole_body_and_gripper_current_joints_status_0_345',qos_profile_sensor_data)
q=pose(Joint_Right_Elbow=-70,Joint_Left_Elbow=-70)
def feed():pub.publish(String(data=json.dumps(payload(q))));rclpy.spin_once(node,timeout_sec=0.)
def get():return request(345)
def command(op,**kw):
    return request(345,'/command',dict(op=op,request_id=uuid.uuid4().hex,session=get()['session'],**kw),desktop=True)
try:
    end=time.monotonic()+10
    while time.monotonic()<end:
        feed()
        try:
            if get()['feedback_fresh']:break
        except OSError:pass
        time.sleep(.04)
    else:raise AssertionError('Recorder did not subscribe: '+(root/'service.log').read_text())
    command('enable',mode='poses');ack=command('capture');assert ack['count']==1
    assert len(library.records())==1
    command('mode',mode='trajectory')
    request(345,'/command',dict(op='heartbeat',session=get()['session'],ready=True))
    command('start');begin=time.monotonic();last_heartbeat=0.
    while time.monotonic()-begin<5.2:
        dt=time.monotonic()-begin
        q['Joint_Right_Elbow']=-70.+max(0.,min(2.,dt-2.))*2.
        feed()
        if time.monotonic()-last_heartbeat>.2:
            request(345,'/command',dict(op='heartbeat',session=get()['session'],ready=True));last_heartbeat=time.monotonic()
        time.sleep(.02)
    command('stop');end=time.monotonic()+15
    while get()['phase']=='processing' and time.monotonic()<end:feed();time.sleep(.05)
    assert get()['phase']=='saved',get()
    assert len(library.records())==2
    # No command topic publishers exist in this test domain.
    assert not any('target_joints' in name for name,types in node.get_topic_names_and_types())
    command('new');request(345,'/command',dict(op='heartbeat',session=get()['session'],ready=True));command('start')
    begin=time.monotonic()
    while time.monotonic()-begin<2.6:
        feed();request(345,'/command',dict(op='heartbeat',session=get()['session'],ready=True));time.sleep(.03)
    assert get()['phase']=='recording'
    p.stdin.close();p.wait(timeout=12)
    assert p.returncode==0,(root/'service.log').read_text()
    assert not discovery_path(345).exists()
    assert list((root/'recordings_incomplete').glob('*.jsonl'))
    assert len(library.records())==2
    print('SERVICE_OK: passive ROS sampler, RPC, auto-save, no motor publishers, EOF partial recovery',flush=True)
finally:
    if p.poll() is None:p.stdin.close();p.wait(timeout=12)
    node.destroy_node();rclpy.shutdown();log.close()
