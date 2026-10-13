"""UI protocol only: no worker, ROS, or movement."""
import tempfile,sys
from pathlib import Path
from PyQt5 import QtWidgets as Q
from autolife_shelf_grasp_306.app import Window
app=Q.QApplication([]);errors=[];sys.excepthook=lambda *a:errors.append(str(a[1]))
w=Window('/home/ubuntu/ros2_ws/src/autolife_shelf_grasp_306',Path(tempfile.mkdtemp()),start_worker=False)
w.receive_event(dict(event='ready',blockers=[]));w.show();app.processEvents()
w.receive_event(dict(event='scene',items=[dict(id=3,name='未录入 · bottle',matched=False,product_id='temporary_x',box=[0,0,10,10])],shape=[720,1280,3],elapsed_ms=1))
assert w.plan.isEnabled() and '可直接规划' in w.list.item(0).text()
class Process:
    def write(self,data):self.data=data
w.process=Process();w.send('plan',id=3)
assert w.expected_target==dict(instance_id=3,product_id='temporary_x')
w.receive_event(dict(event='planned',request_id=w.request,plan_id='a'*32,filename='plan_'+'a'*32+'.json',
    target=dict(instance_id=3,product_id='temporary_x',name='未录入 · bottle'),pixels=[],shape=[720,1280,3],
    seconds=1.2,checked=100,blockers=[],estimated_dimensions_mm=[60,180,60]))
assert w.execute.isEnabled() and '60×180×60' in w.status.text()
w.process=None;w.close();assert not errors,errors
print('UNREGISTERED_UI_OK')
