from types import SimpleNamespace as NS
import itertools
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from autolife_shelf_grasp_306 import gripper_contact_geometry as g
from autolife_shelf_grasp_306.low_object_grasp import check_clearance,basis


def box(lo,hi):return np.array(list(itertools.product(*zip(lo,hi))))


def fixture(tmp_path,monkeypatch,side='right'):
    model=tmp_path/'robot.urdf';model.write_text('<robot/>')
    parts=[('Link_'+side.title()+'_Wrist_Lower_to_Gripper_0',box([-.10,-.03,-.02],[.02,.03,.02])),
           ('Link_'+side.title()+'_Gripper_Left_Finger_0',box([.02,.02,-.005],[.09,.04,.005])),
           ('Link_'+side.title()+'_Gripper_Right_Finger_0',box([.02,-.04,-.005],[.09,-.02,.005]))]
    monkeypatch.setattr(g,'_parts',lambda *args:parts)
    p=NS(cfg=dict(urdf_path=str(model),arm=side,gripper_max_width_m=.09,T_wrist_tool_quat_xyzw=[0,0,0,1]),guard=True)
    return p,parts


@pytest.mark.parametrize('side',['left','right'])
def test_model_includes_fingertip_not_only_palm(tmp_path,monkeypatch,side):
    p,parts=fixture(tmp_path,monkeypatch,side);v=g.gripper_geometry(p)['vertices']
    assert v[:,0].max()==pytest.approx(.09)
    assert parts[0][1][:,0].max()==pytest.approx(.02)
    # 手指宽20mm，90mm净开口：外缘总宽130mm。
    assert v[:,1].min()==pytest.approx(-.065)
    assert v[:,1].max()==pytest.approx(.065)


def test_contact_is_pad_front_and_idempotent(tmp_path,monkeypatch):
    p,_=fixture(tmp_path,monkeypatch);old=np.array([.02,.001,.002]);new=g.contact_tcp(p,old)
    assert new[0]==pytest.approx(.076)  # 根部20mm + 指长70mm×0.8
    assert np.allclose(new[1:],old[1:])
    assert np.allclose(g.contact_tcp(p,new),new)
    assert .09-new[0]==pytest.approx(.014)


def test_gripper_width_change_rebuilds_envelope(tmp_path,monkeypatch):
    p,_=fixture(tmp_path,monkeypatch);a=g.gripper_geometry(p)['vertices']
    p.cfg['gripper_max_width_m']=.10;b=g.gripper_geometry(p)['vertices']
    assert b[:,1].max()>a[:,1].max()


@pytest.mark.parametrize('fraction',[float('nan'),.2,1.0])
def test_contact_never_extrapolates_past_finger(tmp_path,monkeypatch,fraction):
    p,_=fixture(tmp_path,monkeypatch);p.cfg['low_object_finger_contact_fraction']=fraction
    with pytest.raises(ValueError,match='指腹'):g.contact_tcp(p,[.02,0,0])


def test_old_palm_pass_but_real_tip_crosses_table(tmp_path,monkeypatch):
    p,parts=fixture(tmp_path,monkeypatch);p._low_object_grasp_active=True
    R=basis([0,0,1],0,35);center=np.array([.4,0,.025]);old_tcp=np.array([.02,0,0])
    palm=(parts[0][1]-old_tcp)@R.T+center
    assert palm[:,2].min()>.003
    p.pose=lambda q,t:(np.asarray(q['position']),R)
    product=dict(height_m=.05,low_object_support=dict(normal=[0,0,1],offset=0.,center=center.tolist(),radius_m=.23))
    with pytest.raises(ValueError,match='局部桌面'):
        check_clearance(p,dict(position=center),old_tcp,product)
    new_tcp=g.contact_tcp(p,old_tcp)
    check_clearance(p,dict(position=center),new_tcp,product)


def test_geometry_envelope_contains_every_original_vertex(tmp_path,monkeypatch):
    p,parts=fixture(tmp_path,monkeypatch);v=g.gripper_geometry(p)['vertices']
    from scipy.spatial import ConvexHull
    hull=ConvexHull(v);points=np.concatenate([x for _,x in parts])
    assert np.max(points@hull.equations[:,:3].T+hull.equations[:,3])<1e-9


def test_finger_face_crossing_local_table_is_checked_even_vertices_outside():
    v=box([-.3,-.3,-.01],[.3,.3,.02])
    p=NS(_low_object_grasp_active=True,cfg=dict(arm='right'),
         _low_object_vertices=('Right',v),pose=lambda q,t:(np.zeros(3),np.eye(3)))
    product=dict(height_m=.03,low_object_support=dict(normal=[0,0,1],offset=0.,center=[0,0,0],radius_m=.23))
    assert np.all(np.linalg.norm(v[:,:2],axis=1)>.23)
    with pytest.raises(ValueError,match='局部桌面'):
        check_clearance(p,{},np.zeros(3),product)
    p.pose=lambda q,t:(np.array([1.,0,0]),np.eye(3))
    check_clearance(p,{},np.zeros(3),product)
