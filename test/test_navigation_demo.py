import pytest
from autolife_shelf_grasp_306.navigation_logic import GoalProgress,LocalizationGate,select_product
from autolife_shelf_grasp_306.demo_config import validate


def test_ignore_old_success_and_require_this_goal_start():
    p=GoalProgress('货架',100)
    assert p.update(dict(stamp=99,state='succeeded',active_waypoint='货架')) is None
    assert p.update(dict(stamp=101,state='succeeded',active_waypoint='货架')) is None
    assert p.update(dict(stamp=102,state='started',active_waypoint='货架'))=='running'
    assert p.update(dict(stamp=103,state='succeeded',active_waypoint='货架'))=='arrived'


@pytest.mark.parametrize('state',['failed','rejected','canceled','paused','undock_failed'])
def test_no_grasp_after_navigation_failure(state):
    with pytest.raises(ValueError):GoalProgress('货架',100).update(dict(stamp=101,state=state,active_waypoint='货架'))


def test_changed_goal_is_not_our_arrival():
    with pytest.raises(ValueError,match='其他操作'):GoalProgress('货架',100).update(dict(stamp=101,state='succeeded',active_waypoint='前台'))


def test_match_requested_sku_only():
    items=[dict(id=1,product_id='tea',matched=True,score=.99),dict(id=2,product_id='water',matched=True,score=.85)]
    assert select_product(items,'water')==2
    with pytest.raises(ValueError):select_product(items,'sprite')


def test_old_fixed_route_settings_are_not_used():
    settings=validate(dict(turn_deg=180,forward_m=.5))
    assert 'turn_deg' not in settings and 'forward_m' not in settings


def test_manual_relocalization_can_complete_pending_gate():
    gate=LocalizationGate();gate.request(100.)
    assert not gate.check([1,2,.3],101.)
    gate.initial(100.1);gate.amcl(100.2,[1,2,.3],[.01,.01,.02])
    assert gate.check([1,2,.3],100.5) and not gate.pending


@pytest.mark.parametrize('stamp,pose,cov',[
    (99.,[1,2,.3],[.01,.01,.02]),
    (100.2,[8,2,.3],[.01,.01,.02]),
    (100.2,[1,2,.3],[1.,1.,1.]),
    (100.2,[1,2,float('nan')],[.01,.01,.02])])
def test_stale_or_inconsistent_localization_does_not_unlock(stamp,pose,cov):
    gate=LocalizationGate();gate.request(100.);gate.initial(100.1);gate.amcl(stamp,pose,cov)
    assert not gate.check([1,2,.3],100.5)


def test_real_navigation_arrival_clears_old_local_wait_but_old_arrival_does_not():
    gate=LocalizationGate();gate.request(100.)
    gate.navigation(dict(state='succeeded',stamp=99.));assert gate.pending
    gate.navigation(dict(state='succeeded',stamp=101.));assert not gate.pending
