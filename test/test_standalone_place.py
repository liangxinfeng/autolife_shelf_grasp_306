import json
import pytest
from autolife_shelf_grasp_306.core import atomic_json
from autolife_shelf_grasp_306.placement_request import held_grasp,new_standalone_request,standalone_source,check_held_gripper
from autolife_shelf_grasp_306.demo_protocol import validate_place
from autolife_shelf_grasp_306.demo_config import validate


def make_setup(run):
    pid='a'*32
    product=dict(id='water',name='水',geometry='cylinder',height_m=.2,diameter_m=.07)
    grasp=dict(plan_id=pid,run_id=run.name,product=product)
    record=dict(status='completed',plan_id=pid,q={'Joint_Left_Gripper':110.})
    atomic_json(run/('plan_'+pid+'.json'),grasp);atomic_json(run/'held_object.json',record)
    context=new_standalone_request(run,validate({}))
    plan=dict(kind='place',placement_mode='standalone',placement_request_id=context['run_id'],
              place_protocol_version=1,table_height_checked=True,parent_plan_id=pid,
              product=product,demo_settings=context['settings'],source_q=record['q'],
              segments=[dict(phase=p) for p in ('place_approach','place_lower','place_release_retreat','place_return')])
    return context,plan


def test_no_navigation_required(tmp_path):
    context,plan=make_setup(tmp_path);validate_place(plan,tmp_path)
    assert not (tmp_path/'demo_base_result.json').exists()
    assert not (tmp_path/'demo_run.json').exists()
    assert standalone_source(tmp_path)[0]==context


@pytest.mark.parametrize('mutation',['missing','stopped','old_session','bad_pid','nan','wrong_product'])
def test_invalid_held_source(tmp_path,mutation):
    context,plan=make_setup(tmp_path)
    p=tmp_path/'held_object.json';record=json.loads(p.read_text())
    if mutation=='missing':p.unlink()
    elif mutation=='old_session':
        gp=tmp_path/('plan_'+'a'*32+'.json');g=json.loads(gp.read_text());g['run_id']='old';atomic_json(gp,g)
    elif mutation=='wrong_product':
        context['product_id']='different';atomic_json(tmp_path/'standalone_place.json',context)
    else:
        if mutation=='stopped':record['status']='stopped'
        if mutation=='bad_pid':record['plan_id']='../bad'
        if mutation=='nan':record['q']['Joint_Left_Gripper']='nan'
        atomic_json(p,record)
    with pytest.raises(ValueError):validate_place(plan,tmp_path)


@pytest.mark.parametrize('field,value',[('placement_request_id','old'),('parent_plan_id','old'),('demo_settings',{}),('segments',[])])
def test_stale_or_incomplete_plan(tmp_path,field,value):
    _,plan=make_setup(tmp_path);plan[field]=value
    with pytest.raises(ValueError):validate_place(plan,tmp_path)


def test_jaw_change_or_release_invalidates(tmp_path):
    _,plan=make_setup(tmp_path);plan['source_q']={'Joint_Left_Gripper':140}
    with pytest.raises(ValueError):validate_place(plan,tmp_path)
    (tmp_path/'held_object.json').unlink()
    with pytest.raises(ValueError):held_grasp(tmp_path)


def test_observation_receipt(tmp_path):
    context,plan=make_setup(tmp_path)
    plan.update(kind='observe',plan_id='b'*32,observation_protocol_version=1,
                held_gripper_deg=110,segments=[dict(phase='observe_lift')])
    atomic_json(tmp_path/'observation_attempt.json',dict(plan_id=plan['plan_id'],run_id=context['run_id']))
    validate_place(plan,tmp_path)
    atomic_json(tmp_path/'demo_observe_result.json',dict(status='observed'))
    with pytest.raises(ValueError):validate_place(plan,tmp_path)


def test_demo_still_requires_navigation(tmp_path):
    _,plan=make_setup(tmp_path);plan['placement_mode']='demo'
    with pytest.raises(OSError):validate_place(plan,tmp_path)


def test_cropped_bottle_cannot_make_request(tmp_path):
    make_setup(tmp_path);p=tmp_path/('plan_'+'a'*32+'.json');g=json.loads(p.read_text())
    g['geometry']={'image_cropped':True};atomic_json(p,g)
    with pytest.raises(ValueError):new_standalone_request(tmp_path,validate({}))
