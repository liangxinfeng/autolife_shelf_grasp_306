"""Transparent-shell recovery must not turn missing/ambiguous depths into targets."""
import numpy as np
import pytest
from autolife_shelf_grasp_306.bottle import foreground_cylinder,center_from_points,DepthQualityError


def cloud(fraction=.36):
    rng=np.random.default_rng(4)
    theta,z=np.meshgrid(np.linspace(-1.25,1.25,48),np.linspace(0,.18,60))
    points=np.c_[.60-.03*np.cos(theta.ravel()),.1+.03*np.sin(theta.ravel())]
    points+=rng.normal(0,.0007,points.shape)
    dirty=rng.random(len(points))<fraction
    points[dirty,0]+=rng.uniform(.018,.028,dirty.sum())
    return points,z.ravel()


def solve(p,z,total=None):
    return foreground_cylinder(p,z,.03,np.array([.6,.1]),np.array([1.,0.]),0,.18,len(p) if total is None else total)


def test_foreground_recovers_refracted_depth_without_changing_radius():
    p,z=cloud();result=solve(p,z)
    assert result is not None
    center,stats=result
    np.testing.assert_allclose(center,[.6,.1],atol=.002)
    assert stats['cylinder_support_ratio']>=.5


def test_normal_opaque_fit_keeps_fast_path():
    p,z=cloud(0);center,stats,_=center_from_points(np.c_[p,z],[0,0,1],
                {'diameter_m':.06,'height_m':.18},[0,.1,.09])
    assert stats['cylinder_method']=='soft_l1'
    np.testing.assert_allclose(center,[.6,.1,.09],atol=.002)


def test_front_shell_end_to_end_uses_observed_height():
    p,z=cloud(.42);center,stats,_=center_from_points(np.c_[p,z],[0,0,1],
                {'diameter_m':.06,'height_m':.25},[0,.1,.09])
    assert stats['cylinder_method']=='foreground_consensus'
    np.testing.assert_allclose(center,[.6,.1,.09],atol=.003)
    assert 175<stats['observed_height_mm']<185


def test_too_few_or_minority_points_are_not_enough():
    p,z=cloud(0)
    assert solve(p[:50],z[:50]) is None
    assert solve(p,z,total=len(p)*3) is None


def test_single_side_or_single_height_fragment_is_not_enough():
    p,z=cloud(0)
    one_side=p[:,1]<.09
    assert solve(p[one_side],z[one_side]) is None
    assert solve(p,np.full(len(p),.09)) is None


def test_noise_and_wrong_width_rejected():
    rng=np.random.default_rng(7);p=rng.uniform([.52,.04],[.65,.16],(3000,2));z=rng.uniform(0,.18,len(p))
    assert solve(p,z) is None
    p,z=cloud(0);p[:,1]=.1+(p[:,1]-.1)*2
    assert solve(p,z) is None


def test_deterministic_and_inputs_unchanged():
    p,z=cloud();original=p.copy();a=solve(p,z);b=solve(p,z)
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(p,original)


def test_two_competing_shells_do_not_override_majority_requirement():
    p,z=cloud(0);p[::2,0]+=.06
    assert solve(p,z) is None
