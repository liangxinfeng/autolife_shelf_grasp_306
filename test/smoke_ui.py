import tempfile
import sys
from pathlib import Path
from PyQt5 import QtWidgets as Q
from autolife_shelf_grasp_306.app import Window,STYLE,ProductEditor,CatalogDialog

app=Q.QApplication([]);app.setStyleSheet(STYLE)
errors=[]
sys.excepthook=lambda *args:errors.append(str(args[1]))
root=Path('/home/ubuntu/ros2_ws/src/autolife_shelf_grasp_306')
run=Path(tempfile.mkdtemp(prefix='shelf_ui_test_'))
window=Window(root,run,start_worker=False)
window.receive_event({'event':'ready','blockers':['相机/TCP待实测验证'],'products':0})
window.show();app.processEvents()
window.grab().save('/tmp/shelf306_ui.png')
assert window.scan.isEnabled() and not window.execute.isEnabled()
assert not hasattr(window,'table')
launches=[]
window.launch_motion=lambda filename,preparing=False:launches.append((filename,preparing))
window.receive_event({'event':'reset_planned','request_id':window.request,'checked':10,'blockers':[]})
assert launches==[('last_reset_plan.json',True)]
window.request=1;window.busy=True;window.stop_motion()
window.receive_event({'event':'reset_planned','request_id':window.request,'checked':10,'blockers':[]})
assert len(launches)==1 and not window.busy
window.receive_event({'event':'scene','items':[
    {'id':1,'name':'冰红茶','product_id':'tea','matched':True,'box':[0,0,10,10]},
    {'id':2,'name':'怡保','product_id':'water','matched':True,'box':[20,0,30,10]}],
    'shape':[720,1280,3],'elapsed_ms':20.})
window.expected_target={'instance_id':1,'product_id':'tea'}
planned={'event':'planned','request_id':window.request,'pixels':[], 'shape':[720,1280,3],
    'plan_id':'a'*32,'filename':'plan_'+'a'*32+'.json','prepare_token':'reset-ok',
    'target':{'instance_id':1,'product_id':'tea','name':'冰红茶'},
    'seconds':1.4,'checked':1269,'blockers':[],'orientation_mode':'horizontal','approach_yaw_deg':30}
window.receive_event(planned)
assert window.execute.isEnabled() and not window.footer.isVisible()
assert '冰红茶' in window.execute.text()
window.list.setCurrentRow(1)
assert not window.execute.isEnabled() and window.pending_plan is None
window.expected_target={'instance_id':2,'product_id':'water'}
window.receive_event(planned)
assert not window.execute.isEnabled() and window.pending_plan is None
window.receive_event({'event':'scene','items':[
    {'id':3,'name':'测试袋子','product_id':'bag','matched':True,'geometry':'bag_edge',
     'box':[200,200,400,500],'grasp_fraction':[.5,.3]}],'shape':[720,1280,3],'elapsed_ms':50.})
assert window.rim.isVisible() and window.rim.isEnabled()
assert window.video.path==[[300.,290.]]
window.choose_rim()
assert window.picking_rim and not window.execute.isEnabled()
assert window.mode.count()==2 and window.manage.isEnabled()
product=dict(id='gum',name='口香糖',diameter_m=.08,height_m=.05,geometry='cylinder',depth_m=.08,examples=[])
editor=ProductEditor(window,product,editing=True)
editor.width.setValue(50);editor.height.setValue(80)
assert editor.values()['diameter_mm']==50 and editor.values()['height_mm']==80
editor.show();app.processEvents();editor.grab().save('/tmp/shelf306_editor.png');editor.close()
window.catalog_dialog=CatalogDialog(window);window.catalog_dialog.refresh([product]);window.catalog_dialog.show()
app.processEvents();assert window.catalog_dialog.current()['id']=='gum'
window.catalog_dialog.grab().save('/tmp/shelf306_catalog.png');window.catalog_dialog.close()
window.receive_event({'event':'scene','items':[
    {'id':1,'name':'甲','matched':False,'box':[0,0,10,10]},
    {'id':2,'name':'乙','matched':False,'box':[20,0,30,10]}],'shape':[720,1280,3],'elapsed_ms':1})
window.list.setCurrentRow(1)
window.receive_event({'event':'catalog','products':[product],'items':window.items[:]})
assert window.selected_id()==2 # library edits must not change the image being added
sent=[]
window.send=lambda action,**kw:sent.append((action,kw))
question=Q.QMessageBox.question
Q.QMessageBox.question=lambda *a,**kw:Q.QMessageBox.No
window.catalog_dialog.remove_product();window.prepare_robot();assert not sent
Q.QMessageBox.question=lambda *a,**kw:Q.QMessageBox.Yes
window.catalog_dialog.remove_product();window.prepare_robot()
assert sent==[('remove_product',{'product_id':'gum'}),('prepare',{})]
Q.QMessageBox.question=question
window.receive_event({'event':'catalog','products':[],'items':[]})
assert window.catalog_dialog.current() is None and not window.execute.isEnabled()
assert all(not b.isEnabled() for b in window.catalog_dialog.buttons)
window.catalog_dialog.close()
window.close()
assert not errors,errors
print('GUI_SMOKE_OK')
