"""The fast exclusion must never discard a possible narrow-phase collision."""
import numpy as np
import hppfcl as fcl
from scipy.spatial.transform import Rotation
from autolife_shelf_grasp_306.demo_planning import box_separation


def test_box_lower_bound_against_exact_rotated_shapes():
    rng=np.random.default_rng(306)
    shapes=[fcl.Box(.2,.08,.3),fcl.Sphere(.045),fcl.Cylinder(.035,.23)]
    for shape in shapes:
        shape.computeLocalAABB();lo=np.array(shape.aabb_local.min_);hi=np.array(shape.aabb_local.max_)
        for _ in range(200):
            rot=Rotation.random(random_state=rng).as_matrix();boxrot=Rotation.random(random_state=rng).as_matrix()
            pos=rng.uniform(-.4,.4,3);center=rng.uniform(-.2,.2,3);half=rng.uniform(.01,.3,3)
            lower=box_separation(rot,pos,lo,hi,boxrot,center,half)
            exact=fcl.distance(shape,fcl.Transform3f(rot,pos),fcl.Box(*map(float,2*half)),
                fcl.Transform3f(boxrot,center),fcl.DistanceRequest(),fcl.DistanceResult())
            assert lower<=max(0.,exact)+1e-7
            if exact<.01:assert lower<=.01+1e-7


def test_touching_and_clearance_margin_are_not_excluded():
    for gap in (0.,.0005,.005,.01):
        lower=box_separation(np.eye(3),np.array([.2+gap,0,0]),np.full(3,-.1),np.full(3,.1),
                             np.eye(3),np.zeros(3),np.full(3,.1))
        assert abs(lower-gap)<1e-12
