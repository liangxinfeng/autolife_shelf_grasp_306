"""Separate passive ROS sampler + local RPC, bound to its parent's stdin lifetime."""
import argparse
import fcntl
import hmac
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import signal
import socketserver
import sys
import threading
import uuid
from .teaching_feedback import FeedbackState, create_monitor
from .teaching_library import TeachingLibrary
from .teaching_recorder import Recorder
from openarmx_teleop_vr_navigation_306.teaching_ipc import namespace, discovery_path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--robot-id',type=int,required=True)
    parser.add_argument('--library',required=True);args=parser.parse_args()
    root,prefix=namespace(args.robot_id);root.mkdir(parents=True,exist_ok=True)
    lease=(root/(prefix+'.lock')).open('a+')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    owner=uuid.uuid4().hex;token=uuid.uuid4().hex;path=root/(prefix+'-'+owner[:10]+'.sock')
    discovery=discovery_path(args.robot_id);stop=threading.Event()
    state=FeedbackState();recorder=Recorder(TeachingLibrary(args.library),state)
    node=create_monitor(args.robot_id,state)
    class Server(socketserver.ThreadingMixIn,socketserver.UnixStreamServer):
        daemon_threads=True
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,data):
            payload=json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        def handle_request(self,post=False):
            try:
                if not hmac.compare_digest(self.headers.get('X-Recorder-Token',''),token):
                    self.reply(403,dict(success=False,message='录制服务令牌失效'));return
                if post:
                    if self.path!='/command':raise ValueError('无效接口')
                    size=int(self.headers.get('Content-Length','0'))
                    if not 0<size<=16384:raise ValueError('请求大小无效')
                    data=json.loads(self.rfile.read(size))
                    if not isinstance(data,dict):raise ValueError('请求格式无效')
                    result=recorder.command(data,desktop=self.headers.get('X-Recorder-Client')=='desktop')
                else:
                    if self.path!='/status':raise ValueError('无效接口')
                    result=recorder.status()
                self.reply(200,result)
            except (ValueError,KeyError,TypeError,OSError) as exc:
                self.reply(400,dict(success=False,message=str(exc)))
        def do_GET(self):self.handle_request()
        def do_POST(self):self.handle_request(True)
    server=Server(str(path),Handler);os.chmod(path,0o600)
    tmp=discovery.with_name(discovery.name+'.'+owner+'.tmp')
    tmp.write_text(json.dumps(dict(socket=str(path),pid=os.getpid(),token=token,owner=owner)))
    os.chmod(tmp,0o600);os.replace(tmp,discovery)
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    def lifeline():
        try:sys.stdin.buffer.read()
        finally:stop.set()
    threading.Thread(target=lifeline,daemon=True).start()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    try:
        import rclpy
        while not stop.is_set():
            with recorder.lock:
                rclpy.spin_once(node,timeout_sec=0.)
                recorder.tick()
            stop.wait(.02)
    finally:
        recorder.close();server.shutdown();server.server_close();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        try:
            if json.loads(discovery.read_text()).get('owner')==owner:discovery.unlink()
        except (OSError,ValueError):pass
        path.unlink(missing_ok=True);lease.close()


if __name__=='__main__':main()
