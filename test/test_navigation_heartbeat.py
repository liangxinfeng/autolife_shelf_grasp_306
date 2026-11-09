import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest
from autolife_shelf_grasp_306.navigation_heartbeat import SourceLease
from test_navigation_runtime import make_runtime


def test_concurrent_touch_is_not_a_future_timestamp(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import autolife_shelf_grasp_306.navigation_heartbeat as m
    clock=[100.]
    class File:
        name='heartbeat'
        def stat(self):
            clock[0]+=.001
            return SimpleNamespace(st_mtime=clock[0])
    monkeypatch.setattr(m.time,'time',lambda:clock[0])
    lease=SourceLease(tmp_path);lease.paths=[File(),File()]
    assert lease.check(now=1.) and lease.failed_at is None


@pytest.mark.parametrize('missing',['heartbeat','navigation_bridge_heartbeat'])
def test_either_source_missing_stops(missing,tmp_path):
    for name in ('heartbeat','navigation_bridge_heartbeat'):
        if name!=missing:(tmp_path/name).touch()
    assert not SourceLease(tmp_path).check()


@pytest.mark.parametrize('stale',['heartbeat','navigation_bridge_heartbeat'])
def test_stale_latches_and_cannot_resume_previous_goal(stale,tmp_path):
    for name in ('heartbeat','navigation_bridge_heartbeat'):
        (tmp_path/name).touch();os.utime(tmp_path/name,(100.,100.))
    lease=SourceLease(tmp_path)
    assert lease.check(wall=100.1,now=1.)
    os.utime(tmp_path/stale,(99.,99.))
    assert not lease.check(wall=100.1,now=1.1)
    os.utime(tmp_path/stale,(100.,100.))
    assert not lease.check(wall=100.1,now=1.2)
    assert not lease.should_exit(now=1.5) and lease.should_exit(now=1.61)


def test_launch_sets_private_owner_not_control_center(monkeypatch,tmp_path):
    r,p,c,e=make_runtime(monkeypatch,tmp_path);r._start(p)
    assert not r.error
    args=next(v[1] for v in c if v[0]=='spawn')
    assert 'operator_heartbeat_topic:=/grasp306/operator_heartbeat/'+tmp_path.name in args
    assert 'operator_liveness_run:='+str(tmp_path) in args


def test_child_launch_default_and_override():
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    from launch_ros.actions import Node
    path=Path('/tmp/task_heartbeat_test/control_center_navigation/launch/bringup_navigation.launch.py')
    spec=importlib.util.spec_from_file_location('test_nav_launch',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    desc=m.generate_launch_description();ctx=LaunchContext()
    for entity in desc.entities:
        if isinstance(entity,DeclareLaunchArgument):entity.execute(ctx)
    assert ctx.launch_configurations['operator_heartbeat_topic']=='/control_center/operator_heartbeat'
    ctx.launch_configurations['operator_heartbeat_topic']='/grasp306/operator_heartbeat/shelf306_test'
    from launch_ros.utilities import evaluate_parameters
    nodes=[e for e in desc.entities if isinstance(e,Node)]
    arbiter=next(n for n in nodes if any('cmd_vel_arbiter' in str(s) for s in n.cmd))
    params=evaluate_parameters(ctx,arbiter._Node__parameters)
    assert any(isinstance(p,dict) and p.get('operator_heartbeat_topic')=='/grasp306/operator_heartbeat/shelf306_test' for p in params)


def test_real_heartbeat_and_owner_loss(tmp_path):
    # Private DDS domain supplied by test runner; this process publishes only String.
    assert os.environ['ROS_DOMAIN_ID']=='219'
    import rclpy
    from std_msgs.msg import String
    run=tmp_path/'shelf306_isolated';run.mkdir();topic='/grasp306/operator_heartbeat/'+run.name
    for name in ('heartbeat','navigation_bridge_heartbeat'):(run/name).touch()
    rclpy.init();node=rclpy.create_node('task_heartbeat_test')
    messages=[];other=[]
    node.create_subscription(String,topic,lambda m:messages.append((time.monotonic(),m.data)),10)
    node.create_subscription(String,'/control_center/operator_heartbeat',other.append,10)
    p=subprocess.Popen([sys.executable,'-m','autolife_shelf_grasp_306.navigation_heartbeat','--run',str(run),'--topic',topic],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    try:
        end=time.monotonic()+3
        while time.monotonic()<end:
            for name in ('heartbeat','navigation_bridge_heartbeat'):(run/name).touch()
            rclpy.spin_once(node,timeout_sec=.02)
        assert p.poll() is None and len(messages)>20
        assert len({s for _,s in messages})==1 and not other
        initial=len(messages);end=time.monotonic()+1.5
        while time.monotonic()<end:
            # UI still alive but bridge unresponsive -> must stop.
            (run/'heartbeat').touch();rclpy.spin_once(node,timeout_sec=.02)
        assert p.poll()==0 and len(messages)-initial<12
        assert 'navigation_bridge_heartbeat' in p.stdout.read().decode()
    finally:
        if p.poll() is None:p.terminate();p.wait(timeout=3)
        node.destroy_node();rclpy.shutdown()


def test_launch_owner_exit_shuts_down_only_its_backend(tmp_path):
    import threading
    from launch import LaunchDescription, LaunchService, LaunchContext
    from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, SetLaunchConfiguration
    from launch.utilities import perform_substitutions
    path=Path('/tmp/task_heartbeat_test/control_center_navigation/launch/bringup_all.launch.py')
    spec=importlib.util.spec_from_file_location('test_all_launch',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m._active_map_defaults=lambda _:{k:str(tmp_path/k) for k in ('map_yaml','waypoints_file','waypoint_library_directory','active_map_config')}
    desc=m.generate_launch_description()
    args=[e for e in desc.entities if isinstance(e,DeclareLaunchArgument)]
    ctx=LaunchContext()
    for arg in args:arg.execute(ctx)
    watch=next(e for e in desc.entities if isinstance(e,ExecuteProcess) and 'navigation_heartbeat' in ' '.join(perform_substitutions(ctx,arg) for arg in e.cmd))
    handlers=[e for e in desc.entities if isinstance(e,RegisterEventHandler)]
    assert len(handlers)==1
    run=tmp_path/'shelf306_launch_test';run.mkdir();topic='/grasp306/operator_heartbeat/'+run.name
    stop=threading.Event()
    def feed():
        end=time.monotonic()+2
        while not stop.is_set() and time.monotonic()<end:
            for n in ('heartbeat','navigation_bridge_heartbeat'):(run/n).touch()
            time.sleep(.03)
    thread=threading.Thread(target=feed);thread.start()
    # Exercise the actual generated watchdog and exit handler, with a sleeping
    # dummy backend. Never include the robot's navigation nodes in this test.
    service=LaunchService()
    settings=[SetLaunchConfiguration('operator_liveness_run',str(run)),SetLaunchConfiguration('operator_heartbeat_topic',topic)]
    service.include_launch_description(LaunchDescription(settings+args+handlers+[watch,ExecuteProcess(cmd=[sys.executable,'-c','import time; time.sleep(30)'])]))
    started=time.monotonic()
    try:
        assert service.run()==0
        assert time.monotonic()-started<10
    finally:stop.set();thread.join(1)
