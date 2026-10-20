import numpy as np
import pytest
from autolife_shelf_grasp_306.motion import Follower,configure_grasp_path
from autolife_handeye_calibration_306.verification_trajectory import JointPath


def curved():
    return configure_grasp_path(JointPath(['a','b'],[{'a':0.,'b':0.},{'a':5.,'b':-5.},{'a':10.,'b':-4.},{'a':30.,'b':5.}]),{})


def test_chord_bound_contains_dense_curve_error():
    p=curved();f=Follower(p,0,5.)
    for lo,hi in [(0.,8.),(4.,7.),(3.9,11.),(5.,5.5)]:
        u=np.linspace(0,1,1000);s=lo+(hi-lo)*u
        chord=p.at(lo)+(p.at(hi)-p.at(lo))*u[:,None]
        assert np.max(np.abs(p.curve(s)-chord))<=f.chord_bound(lo,hi)+1e-9


def test_corner_lookahead_reduced_not_path_or_limit():
    p=curved();f=Follower(p,0,5.);q=dict(zip(p.names,p.at(0)))
    geometry=p.curve(np.linspace(0,p.length,100)).copy()
    for i in range(1,150):f.step(q,i/60)
    assert f.progress<5 and f.chord_bound(0,f.progress)<=.15+1e-8
    np.testing.assert_array_equal(geometry,p.curve(np.linspace(0,p.length,100)))


def test_real_offpath_still_stops_with_joint_diagnostic():
    f=Follower(curved(),0,5.)
    with pytest.raises(ValueError,match='超过1°'):f.step({'a':2.,'b':2.},.02)
    assert f.tracking['path_error_deg']==2. and f.tracking['worst_joint'] in ('a','b')


def test_straight_path_retains_speed_headroom_inside_hard_lead():
    p=configure_grasp_path(JointPath(['a','b'],[{'a':0.,'b':0.},{'a':90.,'b':45.}]),{})
    f=Follower(p,0,5.)
    for i in range(1,200):f.step({'a':0.,'b':0.},i/60)
    # A stalled motor must not be driven up against the hard lead boundary.
    # The continuous governor now reserves 15% before that unchanged limit.
    assert 4.24<f.progress<=4.25+1e-6 and f.chord_bound(0,5.)<1e-10
