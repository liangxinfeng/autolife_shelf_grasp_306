import numpy as np
from scipy.spatial.transform import Rotation
from autolife_shelf_grasp_306.demo_planning import geometry_box_separations,box_separations


def test_batch_preserves_every_scalar_collision_candidate():
    rng=np.random.default_rng(345)
    for _ in range(100):
        r=Rotation.random(28,random_state=rng).as_matrix();p=rng.normal(size=(28,3))
        low=-rng.uniform(.01,.3,size=(28,3));high=rng.uniform(.01,.3,size=(28,3))
        targets=Rotation.random(8,random_state=rng).as_matrix();centers=rng.normal(size=(8,3))
        extents=rng.uniform(.01,1,size=(8,3))
        actual=geometry_box_separations(r,p,low,high,targets,centers,extents)
        expected=np.array([box_separations(*args,targets,centers,extents) for args in zip(r,p,low,high)])
        assert np.allclose(actual,expected,atol=1e-12)
        assert np.array_equal(actual<=.005,expected<=.005)
