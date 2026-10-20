import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from autolife_shelf_grasp_306.task_recognition import task_detection_request,infer_scene
from autolife_shelf_grasp_306.task_flow import select_candidates,Cursor


@pytest.mark.parametrize('target,prompt',[('盒子','box.'),('绿色盒子','green box.'),('白色纸巾','white tissue.'),('mouse','mouse.')])
@pytest.mark.parametrize('registered',[False,True])
def test_description_not_disabled_by_catalog(target,prompt,registered):
    products=[dict(id='box',name=target,category='beverage',geometry='solid')] if registered else []
    request=task_detection_request(target,products)
    assert request['prompt']==prompt and request['mode']=='all' and request['semantic']


@pytest.mark.parametrize('target',['怡宝','冰红茶','CocaCola'])
def test_brand_keeps_catalog_identity(target):
    products=[dict(id='brand',name=target,geometry='cylinder')]
    request=task_detection_request(target,products)
    assert not request['semantic'] and 'bottle' in request['prompt']
    with pytest.raises(ValueError):select_candidates([dict(id=1,product_id='other')],target,products,request['semantic'])
    assert select_candidates([dict(id=2,product_id='brand')],target,products,request['semantic'])==[2]


def test_unknown_chinese_constraint_not_silently_dropped():
    with pytest.raises(ValueError):task_detection_request('魔法限定盒子',[])


def test_task_context_does_not_modify_manual_state_or_disappear_on_plan():
    calls=[];cfg={'semantic_prompt_overrides':{'beverages':'白色水瓶'}};before=copy.deepcopy(cfg)
    class Detector:
        def infer(self,*args,**kwargs):calls.append((args[2],kwargs));return [],0
    request=task_detection_request('盒子',[dict(id='box',name='盒子')])
    infer_scene(Detector(),{},None,'beverages',request)  # initial scan
    infer_scene(Detector(),{},None,'beverages',request)  # fresh planning frame
    infer_scene(Detector(),{},None,'beverages')          # manual scan restores scope
    assert calls==[('all',{'prompt':'box.'}),('all',{'prompt':'box.'}),('beverages',{})]
    assert cfg==before


def test_real_workflow_start_uses_task_command(tmp_path):
    from autolife_shelf_grasp_306.task_flow_ui import Workflow
    (tmp_path/'data').mkdir()
    (tmp_path/'data/products.json').write_text(json.dumps({'products':[dict(id='b',name='盒子',category='beverage')]}))
    calls=[]
    state=SimpleNamespace(owner=SimpleNamespace(root=tmp_path,send=lambda *a,**kw:calls.append((a,kw))),
        cursor=Cursor({'steps':[dict(kind='grasp',target='盒子')]}))
    Workflow.start_scan(state)
    assert state.semantic_target and state.phase=='scan'
    assert calls==[(('scan_task',),{'target':'盒子'})]


def test_model_receives_explicit_task_prompt():
    from autolife_shelf_grasp_306.perception import Perception
    calls=[]
    detector=SimpleNamespace(proposals=lambda *a,**kw:(calls.append((a[1],kw)) or []))
    state=SimpleNamespace(segmenter=detector,cfg={},previous=[])
    Perception.infer(state,{},SimpleNamespace(products=[]),'all',prompt='box.')
    assert calls==[('all',{'prompt':'box.'})]


def test_worker_keeps_scope_through_fresh_plan_and_logs_recognition():
    import autolife_shelf_grasp_306.worker as worker
    source=Path(worker.__file__).read_text()
    assert 'infer_scene(perception,scene,store,mode,task_recognition)' in source
    assert 'infer_scene(perception,frame,store,mode,task_recognition)' in source
    assert "recognition_diagnostics.json" in source
    section=source.split("if action=='scan_task':",1)[1].split("if action=='set_prompt':",1)[0]
    assert 'runtime_path' not in section and 'semantic_prompt_overrides' not in section
