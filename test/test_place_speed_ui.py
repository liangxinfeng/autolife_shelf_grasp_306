import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('ROS_DOMAIN_ID','217')
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from autolife_shelf_grasp_306.point_cloud_filter import outside_boxes


def old_filter(world,poses,boxes):
    keep=np.ones(len(world),bool)
    for t,(lo,hi) in zip(poses,boxes):
        local=(world-t.translation)@t.rotation
        keep&=np.linalg.norm(local-np.clip(local,lo,hi),axis=1)>.015
    return keep


@pytest.mark.parametrize('seed',range(8))
def test_exact_broadphase_preserves_every_point(seed):
    rng=np.random.default_rng(seed);poses=[];boxes=[]
    world=rng.uniform(-1,1,(10000,3))
    for _ in range(12):
        rotation=Rotation.random(random_state=rng).as_matrix();translation=rng.uniform(-.5,.5,3)
        lo=rng.uniform(-.2,-.03,3);hi=rng.uniform(.03,.2,3)
        poses.append(SimpleNamespace(rotation=rotation,translation=translation));boxes.append((lo,hi))
        points=np.array([hi+[d,0,0] for d in (.01499,.015,.01501,.03)])
        world=np.r_[world,points@rotation.T+translation]
    assert np.array_equal(outside_boxes(world,poses,boxes),old_filter(world,poses,boxes))


def test_mismatched_geometry_rejected():
    with pytest.raises(ValueError):outside_boxes(np.zeros((1,3)),[None],[])


@pytest.mark.parametrize('cells',[np.empty((0,2),int),np.zeros((20,2),int),
    np.array([[-2,-3],[0,-5],[-2,1],[-2,-3]]),
    np.array([[-2**40,-2**40],[2**40,2**40]])])
def test_cell_keys_exact(cells):
    from autolife_shelf_grasp_306.point_cloud_filter import group_cells
    a,b=group_cells(cells);x,y=np.unique(cells,axis=0,return_inverse=True)
    assert np.array_equal(a,x) and np.array_equal(b,y)


def test_grouped_columns_identical_to_old():
    from autolife_shelf_grasp_306.demo_surface import surface_obstacles,obstacle_boxes
    rng=np.random.default_rng(2);points=rng.uniform([.4,-.5,.68],[1.2,.5,.92],(55000,3))
    table={'coef':[.02,-.01,.67],'minimum':[.4,-.5,.67],'maximum':[1.2,.5,.71],
           'support_method':'observed_semantic_plane'}
    p=surface_obstacles(points,table['coef']);lo=np.array(table['minimum']);hi=np.array(table['maximum'])
    p=p[np.all(p[:,:2]>lo[:2]-.05,axis=1)&np.all(p[:,:2]<hi[:2]+.05,axis=1)]
    unique,ids=np.unique(np.floor(p[:,:2]/.05).astype(int),axis=0,return_inverse=True)
    expected=[];a,b,c=table['coef']
    for idx,cell in enumerate(unique):
        cluster=p[ids==idx];lower=cluster.min(0)-.010;upper=cluster.max(0)+.010
        lower[2]=min(lower[2],float(np.min(cluster[:,0]*a+cluster[:,1]*b+c)))
        expected.append(dict(minimum=lower.tolist(),maximum=upper.tolist(),cell=cell.tolist(),padding_m=.010))
    assert obstacle_boxes(points,table)==expected
    assert obstacle_boxes(np.empty((0,3)),table)==[]


def test_wheel_ignored_focused_unfocused_and_direct_edit_preserved(tmp_path):
    from PyQt5 import QtWidgets as Q,QtCore as C,QtGui as G
    from autolife_shelf_grasp_306.app import Window
    from autolife_shelf_grasp_306.no_wheel import NoWheelDoubleSpinBox,NoWheelComboBox
    app=Q.QApplication.instance() or Q.QApplication([])
    (tmp_path/'data').mkdir();run=tmp_path/'run';run.mkdir()
    w=Window(tmp_path,run,start_worker=False)
    w.timer.stop();w.navigation.product_timer.stop();w.studio.timer.stop()
    w.ready=True;w.blockers=[];w.controls();w.show();app.processEvents()
    widgets=[w.grip_margin,*w.target_offsets.spins,w.target_offsets.arm]
    try:
        for widget in widgets:
            assert isinstance(widget,(NoWheelDoubleSpinBox,NoWheelComboBox))
            for focused in (False,True):
                widget.setFocus() if focused else widget.clearFocus()
                before=widget.value() if hasattr(widget,'value') else widget.currentIndex()
                for delta in (120,-120,480,-480):
                    event=G.QWheelEvent(C.QPointF(5,5),C.QPointF(5,5),C.QPoint(),C.QPoint(0,delta),
                                       C.Qt.NoButton,C.Qt.NoModifier,C.Qt.NoScrollPhase,False)
                    Q.QApplication.sendEvent(widget,event)
                    after=widget.value() if hasattr(widget,'value') else widget.currentIndex()
                    assert before==after
        w.grip_margin.setValue(12.5);assert w.grip_margin.value()==12.5
        w.target_offsets.spins[0].setValue(3.);assert w.target_offsets.spins[0].value()==3.
        w.target_offsets.arm.setCurrentIndex(1);assert w.target_offsets.arm.currentData()=='right'
    finally:
        w.hide();w.deleteLater();app.processEvents()
