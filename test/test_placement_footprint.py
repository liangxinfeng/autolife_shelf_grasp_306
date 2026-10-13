import numpy as np
import pytest
from autolife_shelf_grasp_306.placement_support import footprint_outline,support_candidates,pack_support,unpack_support,supported
from autolife_shelf_grasp_306.demo_planning import check_release_departure,CollisionFailure


def test_measured_rectangle_fits_where_enclosing_circle_does_not():
    product=dict(bbox_axes_torso=np.eye(3).tolist(),bbox_extents_m=[.10,.05,.05])
    outline=footprint_outline(product,np.eye(3))
    grid=np.zeros((50,50),np.uint8);grid[17:33,11:39]=1
    assert support_candidates(grid,outline,.005)[25,25]
    from scipy.ndimage import distance_transform_edt
    assert distance_transform_edt(grid)[25,25]*.005 < np.hypot(.10,.05)/2
    assert supported(grid,[0,0],.005,[.1275,.1275,0],outline)


def test_unknown_holes_and_true_edges_remain_rejected():
    outline=np.array([[-.04,-.025],[.04,-.025],[.04,.025],[-.04,.025]])
    grid=np.ones((50,50),np.uint8)
    assert not supported(grid,[0,0],.005,[.01,.12,0],outline)
    grid[25,25]=0
    assert not supported(grid,[0,0],.005,[.1275,.1275,0],outline)


def test_support_codec_and_pose_rotation():
    grid=np.zeros((50,50),np.uint8);grid[12:38,18:32]=1
    decoded,origin,res=unpack_support(pack_support(grid,[1.,2.],.005));np.testing.assert_array_equal(decoded,grid)
    outline=np.array([[-.05,-.02],[.05,-.02],[.05,.02],[-.05,.02]])
    assert not supported(decoded,origin,res,[1.1275,2.1275,0],outline)
    assert supported(decoded,origin,res,[1.1275,2.1275,0],outline[:,::-1])


def test_calibration_shift_is_directional_not_added_to_radius():
    grid=np.zeros((50,50),np.uint8);grid[15:35,10:40]=1
    outline=np.array([[-.03,-.02],[.03,-.02],[.03,.02],[-.03,.02]])
    plain=support_candidates(grid,outline,.005)
    shifted=support_candidates(grid,outline,.005,[.015,0])
    assert shifted.sum()==plain.sum()
    assert np.where(shifted)[1].mean()<np.where(plain)[1].mean()


def test_release_direct_path_cannot_sweep_through_placed_object():
    def check(delta,R=np.eye(3)):
        check_release_departure(np.asarray(delta),R,np.eye(3),np.eye(3),np.array([0,0,1]),.12,.18)
    check([-.02,0,.01]);check([-.13,.10,0])
    for delta in ([.01,0,0],[-.02,.03,0],[-.02,0,-.01]):
        with pytest.raises(CollisionFailure):check(delta)
