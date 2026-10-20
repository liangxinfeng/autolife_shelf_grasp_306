"""Installed UI -> camera -> recognition -> planning; NEVER reset/execute."""
import os
import tempfile
from pathlib import Path
from PyQt5 import QtCore as C,QtWidgets as Q
from autolife_shelf_grasp_306.app import Window,STYLE

os.environ['ROS_DOMAIN_ID']='0';os.environ['RMW_IMPLEMENTATION']='rmw_cyclonedds_cpp'
os.environ['CYCLONEDDS_URI']='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
app=Q.QApplication([]);app.setStyleSheet(STYLE)
run=Path(tempfile.mkdtemp(prefix='shelf_plan_smoke_'))
w=Window('/home/ubuntu/ros2_ws/src/autolife_shelf_grasp_306',run);w.show()
state={'phase':'ready','result':None}
original=w.receive_event
def receive(data):
    original(data)
    if state['phase']=='plan' and data['event'] in ('planned','error','fatal'):
        state['result']=data
        if data['event']=='planned':
            # This readonly test intentionally never resets. Preview is allowed,
            # hardware execution must stay disabled without a reset receipt.
            assert not w.execute.isEnabled() and w.motion is None
            assert w.pending_plan and data['target']['product_id']
            w.grab().save('/tmp/shelf300_ui_planned.png')
        w.close();app.quit()
w.receive_event=receive
def tick():
    if state['phase']=='ready' and w.ready:
        state['phase']='scan';w.send('scan')
    elif state['phase']=='scan' and not w.busy and w.frozen:
        selected=next((v for v in w.items if v['name'] in ('怡保','怡宝')),None)
        if not selected:state['result']={'error':'target not visible'};w.close();app.quit();return
        state['phase']='plan';w.send('plan',id=selected['id'])
timer=C.QTimer();timer.timeout.connect(tick);timer.start(200)
C.QTimer.singleShot(60000,app.quit)
app.exec_();w.close()
print(state,flush=True)
assert state['result'] and state['result'].get('event')=='planned',state
