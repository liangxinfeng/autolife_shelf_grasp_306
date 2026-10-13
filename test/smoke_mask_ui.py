"""Render actual worker-shaped contours with production ImageView painter."""
import json,tempfile,shutil
from pathlib import Path
from PyQt5 import QtWidgets as Q,QtGui as G,QtCore as C
from autolife_shelf_grasp_306.app import Window,ImageView,STYLE
app=Q.QApplication([]);app.setStyleSheet(STYLE)
root=Path('/home/ubuntu/grasp306_mask_display_candidate_20260930');run=Path(tempfile.mkdtemp(prefix='mask_gui_'))
shutil.copy2(root/'evidence/scene.jpg',run/'scene.jpg')
window=Window(root,run,start_worker=False)
window.receive_event({'event':'ready','blockers':[],'products':0})
window.receive_event(json.loads((root/'evidence/scene.json').read_text()))
window.show();app.processEvents()
assert window.mask_toggle.isChecked() and window.video.show_masks
window.grab().save(str(root/'evidence/ui_masks.png'))
window.mask_toggle.setChecked(False);app.processEvents()
assert not window.video.show_masks
window.grab().save(str(root/'evidence/ui_original.png'))
window.mask_toggle.setChecked(True);window.live()
assert window.video.boxes==[] # never overlay stale masks on moving live RGB
window.close()
# Pixel-level check: ring interior is transparent, real object is tinted,
# background is untouched, and toggling removes the fill.
v=ImageView();v.resize(540,360);v.shape=[360,540,3]
v.pix=G.QPixmap(540,360);v.pix.fill(G.QColor(100,100,100))
v.boxes=[dict(id=1,name='ring',box=[40,40,160,160],mask_contours=[
 [[40,40],[160,40],[160,160],[40,160]],[[75,75],[125,75],[125,125],[75,125]]])]
v.show();app.processEvents();im=v.grab().toImage()
assert im.pixelColor(100,100).red()==100
assert im.pixelColor(50,100)!=G.QColor(100,100,100)
assert im.pixelColor(200,200)==G.QColor(100,100,100)
v.show_masks=False;v.update();app.processEvents()
assert v.grab().toImage().pixelColor(50,100)==G.QColor(100,100,100)
v.close();print('MASK_GUI_RENDER_OK')
