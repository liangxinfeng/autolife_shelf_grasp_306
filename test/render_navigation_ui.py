"""Offscreen layout verification with sample UI state; no ROS or motion."""
import json,tempfile
from pathlib import Path
from PyQt5 import QtWidgets as Q,QtGui as G
from autolife_shelf_grasp_306.app import Window,STYLE
app=Q.QApplication([]);app.setStyleSheet(STYLE)
root=Path(tempfile.mkdtemp());(root/'data').mkdir();run=root/'run';run.mkdir()
w=Window(root,run,start_worker=False);w.ready=True;w.busy=False;w.resize(1400,850)
w.navigation.data=dict(ready=True,pose=[0,0,0],active_map='demo',names=['前台','货架','迎宾区'])
w.navigation.refill(w.navigation.maps,[('当前 · 演示地图','demo')])
for box in (w.navigation.front,w.navigation.shelf):w.navigation.refill(box,[(x,x) for x in ['前台','货架','迎宾区']])
w.navigation.shelf.setCurrentIndex(1);w.navigation.refill(w.navigation.product,[('雪碧','sprite'),('冰红茶','tea')])
w.navigation.info.setText('导航已连接 · 定位正常');w.demo.setChecked(True)
w.video.pix=G.QPixmap(1280,720);w.video.pix.fill(G.QColor('#263846'))
w.status.setText('选择前台、货架与饮料，确认后开始演示');w.controls();w.show();app.processEvents()
w.grab().save('/tmp/grasp306_navigation_ui.png')
assert w.height()<=900,(w.width(),w.height())
assert w.execute.isEnabled()
w.close();print('LAYOUT_PASS',w.width(),w.height())
