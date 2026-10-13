import math
from autolife_shelf_grasp_306.navigation_logic import LocalizationGate


def test_three_initial_repeats_keep_first_boundary():
    g=LocalizationGate();g.request(100.)
    g.initial(101.547);g.initial(101.697)
    g.amcl(101.755,[9.05,-13.78,-1.92],[.00026,.00006,.00002])
    g.initial(101.847)
    assert g.initial_stamp==101.547
    assert g.check([9.05,-13.78,-1.92],120.)
    assert not g.pending


def test_each_retry_requires_its_own_measurement():
    # Search attempt_count resets and is deliberately absent from this gate.
    g=LocalizationGate()
    for t in (100.,110.,120.):
        g.request(t);g.initial(t-1)
        g.amcl(t-1,[1.,2.,0.],[.01,.01,.01])
        assert not g.check([1.,2.,0.],t+.5)
        g.initial(t+.1)
        assert not g.check([1.,2.,0.],t+.5)
        g.amcl(t+.2,[1.,2.,0.],[.01,.01,.01])
        assert g.check([1.,2.,0.],t+20.)


def test_stationary_sample_still_requires_transform_agreement():
    g=LocalizationGate();g.request(100.);g.initial(101.)
    g.amcl(101.1,[1.,2.,0.],[.01,.01,.01])
    assert not g.check(None,130.)
    assert not g.check([1.3,2.,0.],130.)
    assert not g.check([1.,2.,.3],130.)
    assert g.check([1.,2.,0.],130.)


def test_invalid_uncertain_and_future_measurements_rejected():
    g=LocalizationGate();g.request(100.);g.initial(101.)
    for stamp,p,cov in [(101.1,[math.nan,2.,0.],[.01]*3),(105.,[1.,2.,0.],[.01]*3),
                        (101.1,[1.,2.,0.],[.3,.01,.01]),(101.1,[1.,2.,0.],[-.01,.01,.01])]:
        g.amcl(stamp,p,cov)
        assert not g.check([1.,2.,0.],102.)
