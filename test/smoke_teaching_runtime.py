import time,json,socket,sys
from pathlib import Path
from PyQt5 import QtCore as C
from autolife_shelf_grasp_306.teaching_runtime import TeachingRuntime
app=C.QCoreApplication([]);r=TeachingRuntime(345,Path('/tmp/vr_teaching_20261030/runtime'),entry='/tmp/vr_teaching_20261030/isolated_stack.sh',url='https://127.0.0.1:19445')
start=time.monotonic();observed=False;stop_at=None;ticks=0;failure=None
def check():
    global observed,stop_at,ticks,failure
    ticks+=1
    if r.data and not observed:
        d=r.data;print('STATUS_KEYS',list(d),'TELEOP_KEYS',list(d.get('teleop',{})),flush=True)
        if not d.get('teleop',{}).get('dry_run'):failure='dry_run not confirmed'
        if d.get('teleop',{}).get('hardware_enabled'):failure='unexpected enabled state'
        observed=True;stop_at=time.monotonic()+2.
    if stop_at and time.monotonic()>=stop_at:
        stop_at=None
        if '--parent-closed' in sys.argv:r.process.stdin.close()
        else:r.stop()
    if observed and not r.owned and r.phase=='stopped':
        assert socket.socket().connect_ex(('127.0.0.1',19445))!=0
        print('RUNTIME_OK',not failure,'UI_TICKS',ticks,'SECONDS',round(time.monotonic()-start,2),'FAILURE',failure,flush=True)
        r.close();app.exit(1 if failure else 0)
    if r.phase=='error' or time.monotonic()-start>65:
        print('RUNTIME_ERROR',r.phase,r.message,flush=True);r.close();app.exit(1)
timer=C.QTimer();timer.timeout.connect(check);timer.start(40);r.start();raise SystemExit(app.exec_())
