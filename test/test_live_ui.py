import os
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['ROS_DOMAIN_ID']='217'
import cv2
import numpy as np
from PyQt5 import QtWidgets as Q
from autolife_shelf_grasp_306.app import Window
from autolife_shelf_grasp_306.live_overlay import encode_packet


def test_live_frame_changes_even_with_selected_plan_and_keeps_annotations(tmp_path):
    app=Q.QApplication.instance() or Q.QApplication([])
    root=tmp_path/'root';root.mkdir();(root/'data').mkdir();run=tmp_path/'run';run.mkdir()
    w=Window(root,run,start_worker=False);w.timer.stop()
    try:
        w.frozen=True;w.overlay_revision=7;w.path_revision=7
        boxes=[dict(id=1,name='盒子',box=[3,4,30,40],mask_contours=[[[3,4],[30,4],[30,40],[3,40]]])]
        meta=dict(revision=7,path_revision=7,items=boxes,path=[[10,10],[20,20]],shape=[60,80,3],lost=0,path_status='projected')
        def send(color):
            _,jpg=cv2.imencode('.jpg',np.full((60,80,3),color,np.uint8))
            (run/'preview.npz').write_bytes(encode_packet(jpg,meta));w.preview_stamp=0;w.tick()
            return w.video.pix.toImage().pixelColor(0,0).red()
        a=send(30);b=send(210)
        assert b-a>150 and w.video.boxes==boxes and w.video.path==meta['path']
        w.overlay_revision=8;w.path_revision=None;send(100)
        assert not w.video.boxes and not w.video.path
        assert '颜色可选' in w.semantic.placeholderText()
    finally:w.close();app.processEvents()
