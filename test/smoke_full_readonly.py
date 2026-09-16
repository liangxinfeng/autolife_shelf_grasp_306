"""Actual Qt -> child worker -> ROS/SHM/YOLO/DINO -> Qt, without motion."""
import os
import sys
import tempfile
from pathlib import Path
from PyQt5 import QtCore as C,QtWidgets as Q
from autolife_shelf_grasp_306.app import Window,STYLE

os.environ['ROS_DOMAIN_ID']='0';os.environ['RMW_IMPLEMENTATION']='rmw_cyclonedds_cpp'
os.environ['CYCLONEDDS_URI']='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
errors=[];sys.excepthook=lambda *args:errors.append(str(args[1]))
app=Q.QApplication([]);app.setStyleSheet(STYLE)
run=Path(tempfile.mkdtemp(prefix='shelf_full_smoke_'))
window=Window('/home/ubuntu/ros2_ws/src/autolife_shelf_grasp_306',run);window.show()
phase={'scan':False,'done':False}
def tick():
    if window.ready and not phase['scan']:
        phase['scan']=True;window.send('scan')
    elif phase['scan'] and not window.busy and window.frozen:
        assert not window.video.pix.isNull()
        assert not window.execute.isEnabled()
        window.grab().save('/tmp/shelf306_ui_live.png')
        phase['done']=True;window.close();app.quit()
timer=C.QTimer();timer.timeout.connect(tick);timer.start(200)
C.QTimer.singleShot(35000,app.quit)
app.exec_()
window.close()
print('FULL_UI',phase,'errors',errors,'log',str(run/'worker.log'))
assert phase['done'] and not errors
