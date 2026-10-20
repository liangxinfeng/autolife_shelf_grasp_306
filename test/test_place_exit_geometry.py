import time
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from autolife_shelf_grasp_306.demo_planning import held_box_rotation,release_displacement,rank_place_ik


def test_complete_box_follows_grasped_object_without_shrinking():
    axes=Rotation.from_euler('z',17,degrees=True).as_matrix()
    original=Rotation.from_euler('xyz',[30,-10,20],degrees=True).as_matrix()
    wrist=Rotation.from_euler('xyz',[-20,10,45],degrees=True).as_matrix()
    product=dict(bbox_axes_torso=axes.tolist(),bbox_extents_m=[.085,.095,.157])
    rotation,extent=held_box_rotation(wrist,original,product)
    np.testing.assert_allclose(extent,product['bbox_extents_m'])
    np.testing.assert_allclose(rotation,wrist@original.T@axes)
    for corner in np.ndindex(2,2,2):
        point=(np.array(corner)*2-1)*extent/2
        np.testing.assert_allclose(rotation@point,wrist@original.T@(axes@point))


@pytest.mark.parametrize('angle',[-40,0,25,40])
def test_exit_tracks_tool_heading_not_torso_x(angle):
    rotation=Rotation.from_euler('z',angle,degrees=True).as_matrix()
    displacement=release_displacement(rotation,[0,0,0,1],[0,0,1],.12,.08)
    np.testing.assert_allclose(displacement,-rotation[:,0]*.12+[0,0,.08],atol=1e-12)
    assert np.linalg.norm(displacement[:2])==pytest.approx(.12)


def test_exit_accounts_for_tool_axis_and_rejects_vertical_jaws():
    tool=Rotation.from_euler('z',90,degrees=True)
    np.testing.assert_allclose(release_displacement(np.eye(3),tool.as_quat(),[0,0,-1],.12,.03),[0,-.12,-.03],atol=1e-12)
    with pytest.raises(ValueError):
        release_displacement(Rotation.from_euler('y',90,degrees=True).as_matrix(),[0,0,0,1],[0,0,1],.12,.08)


def rank(planner,seed=()):
    diagnostics=[]
    result=rank_place_ik(planner,{'a':0.},np.zeros(3),[np.array([.5,0,.8])],np.eye(3),
        np.array([0,0,1.]),{},dict(approach_height_m=.1),SimpleNamespace(check=lambda *a:None),
        time.monotonic()+8,diagnostics,alternative_seeds=seed)
    return result


def test_alternative_seed_is_not_a_commanded_intermediate_pose():
    class Planner:
        names=['a']
        seen=[]
        def solve(self,source,*a,**kw):
            if source['a']<0:raise ValueError('bad seed')
            if source['a']==0:raise ValueError('unreachable branch')
            return {'a':10.}
        def check(self,q,*a):self.seen.append(q['a'])
    p=Planner();result=rank(p,[{'a':30.}])
    assert result and result[0]['above']['a']==10 and result[0]['seed_index']==1
    assert 30. not in p.seen and all(0<=v<=10 for v in p.seen)


def test_endpoint_success_does_not_hide_a_blocked_approach():
    class Planner:
        names=['a']
        def solve(self,*a,**kw):return {'a':12.}
        def check(self,q,*a):
            if 4<q['a']<8:raise ValueError('obstacle')
    assert not rank(Planner())


def test_search_includes_small_angles_between_old_twenty_degree_grid():
    class Planner:
        names=['a']
        def solve(self,source,tcp,target,rotation,*a,**kw):
            yaw=Rotation.from_matrix(rotation).as_euler('xyz',degrees=True)[2]
            if abs(yaw+25)>1e-6:raise ValueError('not feasible')
            return {'a':1.}
        def check(self,*a):pass
    assert rank(Planner())[0]['yaw']==-25
