from PyQt5 import QtWidgets as Q,QtGui as G
from autolife_shelf_grasp_306.navigation_ui import MapView


def test_pending_and_confirmed_robot_are_visible_in_map(tmp_path):
    app=Q.QApplication.instance() or Q.QApplication([])
    image=G.QImage(400,400,G.QImage.Format_RGB32);image.fill(G.QColor('#cccccc'))
    image.save(str(tmp_path/'navigation_map.png'))
    w=MapView(tmp_path,large=True);w.resize(600,600)
    w.data=dict(meta=dict(x=-2,y=-2,yaw=0,resolution=.01,width=400,height=400),pending_pose=[0,0,0])
    w.show();app.processEvents();frame=w.grab().toImage()
    assert frame.pixelColor(300,300).name()=='#ffb45c'
    frame.save('/tmp/grasp_pending_localization_test.png')
    w.data['pose']=[0,0,0];w.update();app.processEvents()
    assert w.grab().toImage().pixelColor(300,300).name()=='#45f59a'
    w.close()
