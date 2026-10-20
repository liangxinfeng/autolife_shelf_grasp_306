from concurrent.futures import Future
from types import SimpleNamespace as NS
import pytest
from autolife_shelf_grasp_306.navigation_logic import NavigationPause, LocalizationGate


def done(success=True,message=''):
    f=Future();f.set_result(NS(success=success,message=message));return f


def test_resume_waits_for_stop_and_retains_goal():
    p=NavigationPause();calls=[];ready=[False,False,True]
    def call(key):
        calls.append(key)
        return done(ready.pop(0),'等待新鲜底盘反馈并连续停稳1秒') if key=='stop_ready' else done()
    p.tick(True,0,call);assert p.tick(True,.1,call) is True
    for t in (.2,.3,.7,.8,1.2,1.3,1.4):p.tick(False,t,call)
    assert not p.confirmed and calls==['pause','stop_ready','stop_ready','stop_ready','resume']


def test_pause_before_nav2_accept_is_retried_not_aborted():
    p=NavigationPause();calls=[]
    def call(key):
        calls.append(key);return done(len(calls)>1,'导航请求尚在确认，请稍候再暂停')
    p.tick(True,0,call);p.tick(True,.1,call);p.tick(True,.5,call)
    assert p.tick(True,.6,call) is True


def test_rapid_toggle_never_sends_resume_before_pause_ack():
    p=NavigationPause();f=Future();calls=[]
    def call(key):calls.append(key);return f if key=='pause' else done()
    p.tick(True,0,call);p.tick(False,.1,call);assert calls==['pause']
    f.set_result(NS(success=True,message=''))
    p.tick(False,.2,call);p.tick(False,.3,call);p.tick(False,.4,call);p.tick(False,.5,call)
    assert calls==['pause','stop_ready','resume'] and not p.confirmed


def test_real_failure_and_hung_service_not_ignored():
    p=NavigationPause();p.tick(True,0,lambda _:done(False,'暂停目标不存在'))
    with pytest.raises(ValueError,match='目标不存在'):p.tick(True,.1,lambda _:None)
    p=NavigationPause();p.tick(True,0,lambda _:Future())
    with pytest.raises(ValueError,match='响应超时'):p.tick(True,6,lambda _:None)


def test_manual_marker_is_not_a_confirmed_pose():
    g=LocalizationGate();g.request(100);g.initial(100,[1,2,.3])
    assert g.requested_pose==[1,2,.3] and not g.check([1,2,.3],101)
    g.amcl(99,[1,2,.3],[.01]*3);assert not g.check([1,2,.3],101)
    g.amcl(101,[1,2,.3],[.01]*3);assert g.check([1,2,.3],101.1)
    g.request(102);assert g.requested_pose is None


def test_late_pause_status_after_resume_is_not_a_new_failure():
    p=NavigationPause();p.last_resume_stamp=100.
    assert p.owns_paused_status(False,99.)
    assert not p.owns_paused_status(False,101.)  # another operator's NEW pause
