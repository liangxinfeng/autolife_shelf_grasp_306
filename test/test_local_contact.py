import numpy as np
import pytest
from autolife_shelf_grasp_306.local_contact import candidates,required_opening,held_tcp,merged_footprint
from autolife_shelf_grasp_306.object_extent import projected_width


def product():
    # Front is narrow, back remains too wide for centred grasp.
    return dict(bbox_axes_torso=np.eye(3).tolist(),bbox_extents_m=[.12,.12,.12],
                measured_footprint_torso=[[-.06,-.02,0],[-.06,.02,0],[.06,.06,0],[.06,-.06,0]])


def cfg():return dict(up_torso=[0,0,1],approach_yaw_deg=0,gripper_max_width_m=.095)


def test_front_contact_does_not_shrink_held_object():
    p=product();before=dict(p);choices=candidates(p,cfg())
    assert choices and choices[0]['yaw']==0 and choices[0]['contact'][0]<0
    assert choices[0]['required_opening_m']<=.095
    assert projected_width(p,[0,1,0])==.12 and p==before


def test_full_width_square_cannot_use_front_to_cheat():
    p=product();p['measured_footprint_torso']=[[-.06,-.06,0],[-.06,.06,0],[.06,.06,0],[.06,-.06,0]]
    assert not candidates(p,cfg())


def test_fingertip_reach_and_symmetric_center_checked():
    p=product();a=np.array([-.03,0,0.])
    assert required_opening(p,a,np.eye(3))>required_opening(p,a-[.01,0,0],np.eye(3))
    assert required_opening(p,a+[0,.02,0],np.eye(3))>required_opening(p,a,np.eye(3))


def test_no_local_fallback_without_measured_hull():
    p=product();del p['measured_footprint_torso'];assert candidates(p,cfg())==[]


def test_offsets_survive_place_and_bad_calibration_is_rejected():
    physical=np.array([.2,0,0]);offset=np.array([.03,.01,0])
    grasp=dict(physical_tcp=physical.tolist(),tcp=(physical+offset).tolist(),product=dict(object_center_offset_wrist=offset.tolist()))
    np.testing.assert_allclose(held_tcp(grasp,physical),physical+offset)
    with pytest.raises(ValueError):held_tcp(grasp,physical+[.01,0,0])
    np.testing.assert_allclose(held_tcp({},physical),physical)


def test_tilt_checks_whole_height_not_only_a_thin_plane():
    from scipy.spatial.transform import Rotation
    p=product();a=np.array([-.03,0,0.])
    assert required_opening(p,a,Rotation.from_euler('x',30,degrees=True).as_matrix())>required_opening(p,a,np.eye(3))


def test_all_frames_contribute_to_union_hull():
    p=product();p['footprint_vertices_torso']=p.pop('measured_footprint_torso')
    hull=np.array(merged_footprint([p,p],[np.zeros(3),np.array([0,.003,0])],np.zeros(3),[0,0,1]))
    assert np.max(hull[:,1])==pytest.approx(.063)


def test_planner_local_fallback_retains_full_product_and_virtual_center():
    from autolife_shelf_grasp_306.planning import Planner
    from types import SimpleNamespace
    p=product();c=cfg();c['T_wrist_tool_quat_xyzw']=[0,0,0,1]
    source={};physical=np.array([.2,0,0]);center=np.array([.5,0,0]);calls=[]
    def route(q,tcp,goal,obj,flexible,started,yaw,profile):
        calls.append((tcp,goal,obj,flexible))
        return dict(tcp=np.asarray(tcp).tolist(),product=obj)
    stub=SimpleNamespace(cfg=c,pose=lambda q,tcp:(np.zeros(3),np.eye(3)),_route=route)
    plan=Planner._local_contact_plan(stub,source,physical,center,p,0.)
    assert plan['grasp_contact_mode']=='front_local' and not calls[0][3]
    assert plan['product']['bbox_extents_m']==p['bbox_extents_m']
    np.testing.assert_allclose(calls[0][1],center)
    np.testing.assert_allclose(np.asarray(plan['tcp'])-physical,center-np.asarray(plan['contact_center']))
    np.testing.assert_allclose(held_tcp(plan,physical),plan['tcp'])


def test_local_geometry_never_bypasses_path_collision():
    from autolife_shelf_grasp_306.planning import Planner,CollisionFailure
    from types import SimpleNamespace
    c=cfg();c['T_wrist_tool_quat_xyzw']=[0,0,0,1]
    def route(*args):raise CollisionFailure('机器人自碰撞')
    stub=SimpleNamespace(cfg=c,pose=lambda *args:(np.zeros(3),np.eye(3)),_route=route)
    with pytest.raises(ValueError,match='开度已通过.*自碰撞'):
        Planner._local_contact_plan(stub,{},np.zeros(3),np.zeros(3),product(),0.)
