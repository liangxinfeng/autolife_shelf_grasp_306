"""Finite odometry sequence through the existing lease/deadman chassis helper."""
import argparse,json,math,subprocess,sys,threading,time
from pathlib import Path
import numpy as np
from .base_sequence import Controller,normalize,scan_points,clearance,MOVES,ObstacleHold
from .core import atomic_json

def request(run,token):
    context=json.loads((run/'task_flow.json').read_text('utf-8'))
    run_id,index=token.split(':');index=int(index)
    from .task_flow import validate
    rows=validate(context['record'])['steps']
    if run_id!=context['run_id'] or not 0<=index<len(rows) or rows[index]['kind']!='base':raise ValueError('底盘请求与流程快照不一致')
    return normalize(rows[index].get('base'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--token',required=True);a=p.parse_args()
    run=Path(a.run);result=dict(token=a.token,status='failed',completed_moves=0);helper=None;node=None;seq=0
    def send(values):
        nonlocal seq
        seq+=1
        if helper is not None and helper.poll() is None:
            helper.stdin.write(json.dumps(dict(sequence=seq,stamp=time.monotonic(),velocity=values))+'\n');helper.stdin.flush()
    try:
        config=request(run,a.token);controller=Controller(config)
        if not config['enabled']:result['status']='completed';return
        import rclpy
        from rclpy.qos import QoSProfile,ReliabilityPolicy
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import LaserScan
        rclpy.init();node=rclpy.create_node('shelf_base_sequence_306')
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
        state={'odom':None,'scans':{},'errors':{}};health={'received':0.,'data':None}
        def odom(m):
            stamp=(m.header.stamp.sec,m.header.stamp.nanosec)
            if state['odom'] is not None and state['odom'][3]==stamp:return
            q=m.pose.pose.orientation;p=m.pose.pose.position;v=m.twist.twist
            state['odom']=(time.monotonic(),[p.x,p.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))],
                           [v.linear.x,v.linear.y,v.angular.z],stamp)
        def scan(m,front):
            try:
                expected='front_laser_frame' if front else 'rear_laser_frame'
                if m.header.frame_id.lstrip('/')!=expected:raise ValueError('激光坐标系与306安装方向不匹配')
                stamp=(m.header.stamp.sec,m.header.stamp.nanosec)
                old=state['scans'].get(front)
                if old and old[2]==stamp:return
                points=scan_points(m.ranges,m.angle_min,m.angle_increment,m.range_min,m.range_max,front)
                state['scans'][front]=(time.monotonic(),points,stamp)
                state['errors'].pop(front,None)
            except ValueError as exc:state['errors'][front]=str(exc)
        node.create_subscription(Odometry,'/topic_gv_wheel_odom_0_300',odom,qos)
        for front,side in ((True,'front'),(False,'rear')):
            node.create_subscription(LaserScan,f'/topic_gv_{side}_lidar_0_300',lambda m,f=front:scan(m,f),qos)
        helper=subprocess.Popen([sys.executable,'-m','autolife_shelf_grasp_306.chassis','--fixed-sequence'],stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,stderr=sys.stderr,text=True,bufsize=1)
        def reports():
            for line in helper.stdout:
                try:health.update(received=time.monotonic(),data=json.loads(line))
                except ValueError:pass
        threading.Thread(target=reports,daemon=True).start()
        start=time.monotonic();started=False;last_odom=None;velocity=[0.,0.,0.];last_index=-1;obstacles=ObstacleHold();was_blocked=False
        while not controller.done:
            rclpy.spin_once(node,timeout_sec=.01);now=time.monotonic()
            if (run/'stop').exists():raise ValueError('用户停止')
            if not (run/'heartbeat').exists() or time.time()-(run/'heartbeat').stat().st_mtime>1.:raise ValueError('界面心跳中断')
            if helper.poll() is not None:raise ValueError('底盘控制进程退出')
            ready=(state['odom'] is not None and now-state['odom'][0]<.3 and len(state['scans'])==2 and
                   all(now-s[0]<.5 for s in state['scans'].values()) and now-health['received']<.6)
            if not ready:
                send([0.,0.,0.])
                if started or now-start>6.:raise ValueError('等待新鲜里程计、双雷达及底盘控制状态')
                continue
            if state['errors']:raise ValueError('；'.join(state['errors'].values()))
            h=health['data'] or {}
            if h.get('error'):
                if not started and now-start<6.:send([0.,0.,0.]);continue
                raise ValueError(h['error'])
            if not started:
                if not h.get('stationary'):
                    send([0.,0.,0.])
                    if now-start>6.:raise ValueError('开始底盘动作前未停稳')
                    continue
                started=True
            current=state['odom']
            if current[3]!=last_odom:
                velocity=controller.update(current[1],current[2],current[0],paused=obstacles.active or (run/'task_pause').exists());last_odom=current[3]
            points=np.concatenate([s[1] for s in state['scans'].values()])
            if (run/'task_pause').exists():
                velocity=[0.,0.,0.];obstacles=ObstacleHold();was_blocked=False
            else:velocity=obstacles.update(points,velocity,current[2],now)
            if obstacles.active!=was_blocked:
                was_blocked=obstacles.active
                message='路径近处有障碍，停车等待；清除后自动继续（最多30秒）' if was_blocked else '障碍已清除，继续本段定量移动'
                print(json.dumps(dict(event='progress',token=a.token,index=controller.index,total=len(config['moves']),message=message),ensure_ascii=False),flush=True)
                with (run/'base_guard.jsonl').open('a') as log:
                    nearest=points[np.argsort(np.linalg.norm(points,axis=1))[:12]].tolist()
                    log.write(json.dumps(dict(time=time.time(),blocked=was_blocked,index=controller.index,measured=current[2],probe=obstacles.probe,nearest_points=nearest))+'\n')
            send(velocity)
            if controller.index!=last_index:
                last_index=controller.index;result['completed_moves']=controller.index
                print(json.dumps(dict(event='progress',token=a.token,index=controller.index,total=len(config['moves']),
                    message='路径近处有障碍，停车等待；清除后自动继续（最多30秒）' if obstacles.active else ''),ensure_ascii=False),flush=True)
            time.sleep(.015)
        result['status']='completed'
    except Exception as exc:result['reason']=str(exc)
    finally:
        # Zero repeatedly while retaining the lease, then EOF closes the helper.
        # Helper's 250 ms watchdog also covers this supervisor dying abruptly.
        if helper is not None:
            for _ in range(5):
                try:send([0.,0.,0.])
                except (OSError,ValueError):break
                time.sleep(.04)
            try:helper.stdin.close();helper.wait(timeout=2.)
            except (OSError,subprocess.TimeoutExpired):helper.terminate()
        if node is not None:node.destroy_node();rclpy.shutdown()
        atomic_json(run/'base_result.json',result)
        print(json.dumps(dict(event='done',**result),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
