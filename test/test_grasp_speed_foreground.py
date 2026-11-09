import json
import numpy as np
import pytest
from test_auto_grasp import window,qt,plan
from autolife_shelf_grasp_306.grasp_speed import checked_percent,read_percent
from autolife_shelf_grasp_306.bottle import cylinder_foreground,DepthQualityError


@pytest.mark.parametrize('value',[0,19,101,float('nan'),float('inf')])
def test_invalid_speed(value):
    with pytest.raises(ValueError):checked_percent(value)


def test_slider_replans_actual_request_without_enrollment(window,qt):
    w=window;pid='temporary_live_object'
    w.receive_event(dict(event='scene',items=[dict(id=7,name='未录入 · 水瓶',product_id=pid,matched=False,score=.9)],shape=[480,640,3],elapsed_ms=1))
    assert w.plan.isEnabled() and not (w.root/'data/products.json').exists()
    w.grasp_speed.setValue(80)
    assert read_percent(w.root)==80
    w.send('plan',id=7)
    assert w.process.commands[-1]['grasp_speed_percent']==80
    assert not w.grasp_speed.isEnabled()
    event=plan(w);event['target'].update(instance_id=7,product_id=pid,name='未录入 · 水瓶')
    w.receive_event(event);qt.processEvents()
    assert w.pending_plan and w.execute.isEnabled()
    w.grasp_speed.setValue(50)
    assert w.pending_plan is None and not w.execute.isEnabled()
    assert read_percent(w.root)==50


def test_foreground_preserves_full_vertical_extent_and_rejects_background():
    rng=np.random.default_rng(42)
    # 同一瓶体的不同高度不能按相机深度误删；少数远背景应被剔除。
    body=np.c_[rng.uniform(-.025,.025,1000),rng.uniform(.55,.58,1000),rng.uniform(-.1,.15,1000)]
    background=np.c_[rng.uniform(-.025,.025,100),rng.uniform(.8,1.,100),rng.uniform(-.45,-.3,100)]
    points=np.r_[body,background]
    keep=cylinder_foreground(points,None,[0,0,1],.09)
    assert keep[:1000].all() and not keep[1000:].any()
    # 不能将大多数杂乱数据当成稳定瓶身。
    with pytest.raises(DepthQualityError):
        cylinder_foreground(rng.uniform(-1,1,(1000,3)),None,[0,0,1],.09)
