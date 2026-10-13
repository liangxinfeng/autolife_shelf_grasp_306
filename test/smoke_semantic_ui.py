"""Offscreen UI only: never create a worker or a robot command publisher."""
import tempfile
from pathlib import Path
from PyQt5 import QtWidgets as Q
from PyQt5.QtGui import QPixmap
from autolife_shelf_grasp_306.app import Window,STYLE

app=Q.QApplication([]);app.setStyleSheet(STYLE)
root=Path('/home/ubuntu/grasp306_grounded_sam2_candidate_20260930')
window=Window(root,Path(tempfile.mkdtemp(prefix='grounded_ui_')),start_worker=False)
window.receive_event({'event':'ready','blockers':[],'products':0,'mode':'all',
                      'detector_backend':'grounded_sam2','semantic_prompt':''})
window.show();app.processEvents()
sent=[];window.send=lambda action,**kw:sent.append((action,kw))
window.semantic.setText('瓶子');window.change_prompt()
assert sent==[('set_prompt',{'text':'瓶子'})]
window.receive_event({'event':'scene','semantic_prompt':'瓶子','items':[
    {'id':1,'name':'未录入 · bottle','matched':False,'box':[757,202,879,426]}],
    'shape':[720,1280,3],'elapsed_ms':170.})
assert window.semantic.text()=='瓶子' and not window.execute.isEnabled()
window.busy=True;window.controls();assert not window.semantic_apply.isEnabled()
window.busy=False;window.controls();assert window.semantic_apply.isEnabled()
window.video.pix=QPixmap(str(root/'evidence/grounded_overlay.jpg'));window.video.update()
app.processEvents();window.grab().save(str(root/'evidence/ui.png'))
assert window.semantic_apply.geometry().bottom()<window.list.geometry().top()
window.close();print('SEMANTIC_GUI_OK')
