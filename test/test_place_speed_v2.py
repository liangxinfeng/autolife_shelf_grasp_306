import copy
import threading
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_vendor_place import fixture
from autolife_shelf_grasp_306.vendor_place_trajectory import build_blocks,validate_blocks
from autolife_shelf_grasp_306.vendor_grasp_trajectory import timed_samples
from autolife_handeye_calibration_306.verification_trajectory import JointPath
from autolife_handeye_calibration_306.core import BODY
from autolife_shelf_grasp_306.held_broadphase import sphere_box_candidates
from autolife_shelf_grasp_306.trajectory_preload import TrajectoryPreload


def test_new_free_approach_uses_exact_grasp_timing_but_descent_stays_unchanged():
    p=fixture();old=copy.deepcopy(p['vendor_blocks']);p['placement_speed_policy']='grasp_transfer_v2'
    p['vendor_blocks']=build_blocks(p,p['config']);validate_blocks(p)
    assert p['vendor_blocks'][0]['duration_sec']<old[0]['duration_sec']
    assert p['vendor_blocks'][1]==old[1]
    assert p['vendor_blocks'][2]==old[2]  # 缺少退出凭据时全段保留局部限制。
    names=p['controlled_joints'];caps=np.array([20. if n in BODY else 60. for n in names])
    accels=np.array([80. if n in BODY else 240. for n in names])
    _,expected=timed_samples(JointPath(names,p['segments'][0]['waypoints']),caps,accels)
    assert p['vendor_blocks'][0]['duration_sec']==expected


def test_return_frees_only_after_checked_exit_and_slider_still_applies():
    p=fixture();p['placement_speed_policy']='grasp_transfer_v2'
    p['segments'][2]['transfer_from_s']=0.
    fast=build_blocks(p,p['config']);slow=build_blocks(p,{**p['config'],'grasp_speed_percent':50.})
    assert fast[2]['duration_sec']<p['vendor_blocks'][2]['duration_sec']
    assert all(s['duration_sec']>f['duration_sec'] for s,f in zip(slow,fast))
    p['segments'][2]['transfer_from_s']=-1
    with pytest.raises(ValueError,match='退出边界'):build_blocks(p,p['config'])


def test_vector_bounds_preserve_scalar_candidates_in_rotated_scenes():
    rng=np.random.default_rng(345)
    for _ in range(100):
        rotations=Rotation.random(40,random_state=rng).as_matrix()
        translations=rng.uniform(-1,1,(40,3));low=rng.uniform(-.5,-.01,(40,3));high=-low
        center=rng.uniform(-1,1,3);radius=rng.uniform(.01,.5)
        expected=[]
        for i in range(40):
            local=rotations[i].T@(center-translations[i])
            if np.linalg.norm(local-np.clip(local,low[i],high[i]))<=radius:expected.append(i)
        actual=sphere_box_candidates(center,radius,rotations,translations,low,high)
        np.testing.assert_array_equal(actual,expected)


def test_preload_poll_keeps_cancel_and_does_not_swallow_import_error():
    release=threading.Event()
    def loader(cfg):release.wait(1.);return 'function'
    task=TrajectoryPreload(loader,{})
    try:
        def cancel():raise ValueError('用户停止')
        with pytest.raises(ValueError,match='用户停止'):task.result(cancel)
    finally:release.set();task.close()
    def broken(cfg):raise ImportError('缺少原厂模块')
    task=TrajectoryPreload(broken,{})
    try:
        with pytest.raises(ImportError):task.result(lambda:None)
    finally:task.close()


def test_preload_success_returns_only_loaded_function():
    done=threading.Event();function=lambda:None
    def loader(cfg):done.set();return function
    task=TrajectoryPreload(loader,{})
    try:
        assert task.result(lambda:done.wait(.01)) is function
    finally:task.close()
