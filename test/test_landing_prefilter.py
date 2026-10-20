import numpy as np
from scipy.spatial.transform import Rotation
from autolife_shelf_grasp_306.demo_planning import landing_candidate_mask,CLUTTER_MARGIN_M


def test_keeps_empty_scene_and_clearly_free_candidates():
    points=np.array([[0,0,0],[.2,0,0]])
    assert landing_candidate_mask(points,np.eye(4),{},[]).all()
    boxes=[dict(minimum=[-.01,-.01,-.01],maximum=[.01,.01,.01])]
    assert landing_candidate_mask(points,np.eye(4),{},boxes).tolist()==[False,True]


def test_retains_borderline_ik_tolerance_and_local_grasp_offsets():
    boxes=[dict(minimum=[-.1,-.1,-.1],maximum=[0,.1,.1])]
    points=np.array([[.044,0,0],[.047,0,0],[.051,0,0]])
    assert landing_candidate_mask(points,np.eye(4),{},boxes).tolist()==[False,True,True]
    assert landing_candidate_mask(points,np.eye(4),dict(object_center_offset_wrist=[.01,0,0]),boxes).all()


def test_uses_ground_frame_not_torso_axes():
    T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',[170,5,20],degrees=True).as_matrix();T[:3,3]=[.1,.2,1.]
    ground=np.array([[0,0,0],[.2,0,0]]);torso=(ground-T[:3,3])@T[:3,:3]
    boxes=[dict(minimum=[-.01,-.01,-.01],maximum=[.01,.01,.01])]
    np.testing.assert_array_equal(landing_candidate_mask(torso,T,{},boxes),[False,True])


def test_rejected_candidates_cannot_escape_with_offset_and_ik_error():
    rng=np.random.default_rng(23);points=rng.uniform(-.08,.08,(1000,3));offset=np.array([.012,.003,0])
    boxes=[dict(minimum=[-.02]*3,maximum=[.02]*3)]
    rejected=points[~landing_candidate_mask(points,np.eye(4),dict(object_center_offset_wrist=offset),boxes)]
    for _ in range(20):
        r=Rotation.random(random_state=rng);error=rng.normal(size=(len(rejected),3));error*=.004/np.maximum(np.linalg.norm(error,axis=1)[:,None],1e-9)
        physical=rejected-r.apply(offset)+error
        distance=np.linalg.norm(physical-np.clip(physical,-.02,.02),axis=1)
        assert np.all(distance<.045+CLUTTER_MARGIN_M+1e-8)
