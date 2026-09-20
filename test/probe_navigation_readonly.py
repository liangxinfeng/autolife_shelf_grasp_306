"""Observe map/list/TF only; no navigate, map switch or localization request."""
import os,json,subprocess,tempfile,time,threading,queue
from pathlib import Path
os.environ['ROS_DOMAIN_ID']='0';os.environ['RMW_IMPLEMENTATION']='rmw_cyclonedds_cpp'
os.environ['CYCLONEDDS_URI']='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
run=Path(tempfile.mkdtemp(prefix='grasp_nav_readonly_'));(run/'heartbeat').touch()
proc=subprocess.Popen(['/usr/bin/python3','-m','autolife_shelf_grasp_306.navigation_bridge','--run',str(run)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
messages=queue.Queue()
def read():
    for line in proc.stdout:
        try:messages.put(json.loads(line))
        except ValueError:pass
threading.Thread(target=read,daemon=True).start()
latest={}
try:
    deadline=time.monotonic()+8
    while time.monotonic()<deadline and proc.poll() is None:
        (run/'heartbeat').touch()
        try:latest=messages.get(timeout=.1)
        except queue.Empty:pass
    print(json.dumps(latest,ensure_ascii=False));print('READONLY_RUN',run)
finally:
    if proc.poll() is None:
        proc.stdin.write('{"action":"quit"}\n');proc.stdin.flush();proc.wait(timeout=4)
    print('EXIT',proc.returncode)
