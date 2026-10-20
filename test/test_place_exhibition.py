import time
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from autolife_shelf_grasp_306.demo_surface import obstacle_boxes
from autolife_shelf_grasp_306.demo_planning import upright_heading,rank_place_ik

def scene():
    return dict(coef=[0.,0.,.72],minimum=[.40,-.28,.71],maximum=[1.02,.39,.73],support_method='observed_semantic_plane')

def cluster(x,z):
    return np.array([[x+dx,-.17+dy,z+dz] for dx in (0,.002,.004) for dy in (0,.002,.004) for dz in (0,.002)])

def test_outside_table_return_keeps_measured_volume_not_extrapolated_pillar():
    p=cluster(.355,1.03);box=obstacle_boxes(p,scene())[0]
    np.testing.assert_allclose(box['minimum'],p.min(axis=0)-.01)
    np.testing.assert_allclose(box['maximum'],p.max(axis=0)+.01)
    assert box['minimum'][2]>1.

def test_inside_table_obstacle_still_protects_occluded_body():
    p=cluster(.555,1.03);box=obstacle_boxes(p,scene())[0]
    assert box['minimum'][2]==pytest.approx(.72)
    assert box['maximum'][2]>=p[:,2].max()+.0099

def test_multiframe_obstacle_is_not_erased_or_reextruded():
    frames=[cluster(.355,1.03)+[0,0,i*.001] for i in range(5)]
    boxes=obstacle_boxes(np.concatenate(frames),scene(),frames=frames)
    assert len(boxes)==1
    assert boxes[0]['minimum'][2]>1.
    for p in frames:
        assert np.all(p>=boxes[0]['minimum']) and np.all(p<=boxes[0]['maximum'])

def test_boundary_overlap_remains_conservative():
    box=obstacle_boxes(cluster(.398,1.03),scene())[0]
    assert box['minimum'][2]==pytest.approx(.72)

@pytest.mark.parametrize('up',[[0,0,1],[0,0,-1],[.1,0,np.sqrt(.99)]])
@pytest.mark.parametrize('yaw',[-130.,-60.,13.,70.,150.])
def test_heading_can_be_oblique_without_tipping_object(up,yaw):
    up=np.array(up);reference=Rotation.from_euler('xyz',[12,34,56],degrees=True).as_matrix()
    actual=Rotation.from_rotvec(up*np.deg2rad(yaw)).as_matrix()@reference
    desired,angle=upright_heading(actual,reference,up)
    np.testing.assert_allclose(desired,actual,atol=1e-10)
    np.testing.assert_allclose(desired.T@up,reference.T@up,atol=1e-10)
    assert angle==pytest.approx(yaw)

def test_landing_locks_solved_oblique_heading_and_checks_support():
    R=Rotation.from_euler('z',65,degrees=True).as_matrix()
    class Planner:
        names=['a']
        calls=[]
        def solve(self,source,tcp,point,rotation,*args,**kwargs):
            self.calls.append((rotation,kwargs));return {'a':10. if kwargs.get('free_heading') else 11.}
        def pose(self,q,tcp):return np.zeros(3),R
        def check(self,*args):pass
    class Guard:
        count=0
        def check_support(self,q):self.count+=1
        def check(self,*args):pass
    p=Planner();guard=Guard()
    result=rank_place_ik(p,{'a':0.},np.zeros(3),[np.array([.4,0,.8])],np.eye(3),np.array([0,0,1.]),{},
        {'approach_height_m':.1},guard,time.monotonic()+2,[],free_heading=True)
    assert result and result[0]['yaw']==pytest.approx(65.)
    assert guard.count>0
    np.testing.assert_allclose(p.calls[1][0],R)
    assert p.calls[0][1]['free_heading'] is True
    assert 'free_heading' not in p.calls[1][1]
