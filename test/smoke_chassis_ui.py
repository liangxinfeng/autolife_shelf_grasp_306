"""GUI test with fake process: no ROS initialization or movement."""
import tempfile,time,json
from pathlib import Path
from PyQt5 import QtWidgets as Q,QtCore as C,QtGui as G
from autolife_shelf_grasp_306.chassis_ui import ChassisDialog
from autolife_shelf_grasp_306.app import STYLE,Window
app=Q.QApplication([]);app.setStyleSheet(STYLE);owner=Q.QWidget();owner.run=Path(tempfile.mkdtemp())
dialog=ChassisDialog(owner,start_backend=False);sent=[]
dialog.send=lambda value:sent.append(value)
dialog.last_status=time.monotonic();dialog.isActiveWindow=lambda:True
dialog.mouse_press(C.Qt.Key_W);assert sent[-1]==(.05,0,0)
dialog.stop();assert sent[-1]==(0,0,0)
dialog.keyPressEvent(G.QKeyEvent(C.QEvent.KeyPress,C.Qt.Key_Q,C.Qt.NoModifier));assert sent[-1]==(0,.05,0)
dialog.keyReleaseEvent(G.QKeyEvent(C.QEvent.KeyRelease,C.Qt.Key_Q,C.Qt.NoModifier));assert sent[-1]==(0,0,0)
dialog.keyPressEvent(G.QKeyEvent(C.QEvent.KeyPress,C.Qt.Key_A,C.Qt.NoModifier));assert sent[-1]==(0,0,.2)
app.sendEvent(dialog,C.QEvent(C.QEvent.WindowDeactivate));assert sent[-1]==(0,0,0) and not dialog.keys
dialog.last_status=0;dialog.mouse_press(C.Qt.Key_W);dialog.tick();assert sent[-1]==(0,0,0)
dialog.show();app.processEvents();dialog.grab().save('/tmp/grasp306_chassis_dialog.png')
dialog.closing_since=time.monotonic()-.5;dialog.last_status=time.monotonic();dialog.stationary=True;dialog.active=False
assert dialog.can_close();dialog.active=True;assert not dialog.can_close();dialog.active=False
dialog.shutdown();dialog.close()
import autolife_shelf_grasp_306.chassis_ui as module
class FakeDialog(Q.QDialog):
    def exec_(self):return 0
    def shutdown(self):pass
module.ChassisDialog=FakeDialog
window=Window(Path('/home/ubuntu/ros2_ws/src/autolife_shelf_grasp_306'),owner.run,start_worker=False)
window.ready=True;window.busy=False;window.pending_plan={'old':'unsafe'};window.items=[{'id':1}]
window.adjust_chassis();assert not window.items and window.pending_plan is None and not window.execute.isEnabled()
window.pending_plan={'motion':'active'};window.motion=object();window.adjust_chassis()
assert window.pending_plan=={'motion':'active'}
window.motion=None;window.close();print('CHASSIS_GUI_OK')
