"""Own a headless VR stack; stdin EOF also closes it when the GUI disappears."""
import argparse,json,os,signal,ssl,subprocess,sys,threading,time,urllib.request
from pathlib import Path


def request_disable(url):
    req=urllib.request.Request(url+'/api/hardware-control',data=b'{"enabled":false}',headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,context=ssl._create_unverified_context(),timeout=3.) as response:
            result=json.load(response);print('DISABLE',json.dumps(result,ensure_ascii=False),flush=True)
        until=time.monotonic()+3.
        while time.monotonic()<until:
            with urllib.request.urlopen(url+'/api/status',context=ssl._create_unverified_context(),timeout=.5) as response:state=json.load(response)
            if state.get('teleopStatusFresh') and state.get('teleop',{}).get('hardware_enabled') is False:
                print('DISABLE_CONFIRMED',flush=True);return
            time.sleep(.1)
        print('DISABLE_NOT_CONFIRMED; closing owned command processes',flush=True)
    except Exception as exc:print('DISABLE_UNAVAILABLE',str(exc),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--entry',required=True);parser.add_argument('--url',default='https://127.0.0.1:8445')
    args=parser.parse_args();entry=Path(args.entry).resolve()
    if not entry.is_file():raise SystemExit('VR后台启动器不存在')
    if not args.url.startswith('https://127.0.0.1:'):raise SystemExit('控制地址必须是本机服务')
    stop=threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,lambda *_:stop.set())
    def stdin_watch():
        try:sys.stdin.buffer.readline()
        finally:stop.set()
    threading.Thread(target=stdin_watch,daemon=True).start()
    child=subprocess.Popen(['/bin/bash',str(entry)],stdin=subprocess.DEVNULL,start_new_session=True)
    print('BACKEND_PID',child.pid,flush=True)
    try:
        while child.poll() is None and not stop.wait(.2):pass
    finally:
        if child.poll() is None:
            request_disable(args.url)
            # Signal only the process created here. The existing stack owns and
            # shuts down its private ROS process group using its normal trap.
            child.send_signal(signal.SIGTERM)
            try:child.wait(timeout=25.)
            except subprocess.TimeoutExpired:
                print('BACKEND_SLOW_SHUTDOWN; continuing to wait for owned stack',flush=True)
                child.wait()
    code=child.returncode
    print('BACKEND_CLOSED',code,flush=True)
    return 0 if stop.is_set() else (code or 0)


if __name__=='__main__':raise SystemExit(main())
