"""自然名称、物体/包装区分与任务识别保持一致。无机器人接口。"""
import pytest
from autolife_shelf_grasp_306.grounded_sam2 import (
    normalize_prompt,detection_query,category_label_matches,semantic_display_name)
from autolife_shelf_grasp_306.task_recognition import task_detection_request


@pytest.mark.parametrize('name,text',[
    ('纸团','crumpled paper.'),('废纸团','crumpled paper.'),('一团纸','crumpled paper.'),
    ('纸巾团','crumpled tissue.'),('揉皱的纸巾','crumpled tissue.'),
    ('皱巴巴的纸巾','crumpled tissue.'),('白色揉成团的纸巾','white crumpled tissue.'),
    ('请帮我找一下一个白色纸团','white crumpled paper.'),
    ('帮我识别一下白色海绵','white sponge.'),('压扁的水瓶','flattened water bottle.'),
    ('空的塑料瓶','empty plastic bottle.'),('纸巾盒','tissue box.'),
    ('纸巾包','tissue pack.'),('瓶盖','bottle cap.'),('抹布','cleaning cloth.'),
    ('海绵和纸团','sponge. crumpled paper.'),('垃圾袋','trash bag.'),
])
def test_everyday_description_preserves_meaning(name,text):
    assert normalize_prompt(name)==text


@pytest.mark.parametrize('name',['纸巾','白色纸巾','揉皱的纸巾','白色揉皱的纸巾'])
def test_tissue_never_becomes_container(name):
    query,_=detection_query(name,{'grounded_box_threshold':.35})
    assert not any(x in query.split() for x in ['box.','pack.','container.'])
    assert not category_label_matches(name,'tissue box')
    assert not category_label_matches(name,'pack tissue box')
    assert category_label_matches(name,'crumpled tissue')


def test_qualified_synonyms_keep_color_and_no_generic_fallback():
    query,threshold=detection_query('白色纸巾',{'grounded_box_threshold':.35})
    assert all(s.startswith('white ') for s in query.rstrip('.').split('. '))
    assert threshold==.35
    query,_=detection_query('白色纸巾盒',{'grounded_box_threshold':.35})
    assert query=='white tissue box.'


def test_tissue_boxes_remain_available_when_explicitly_requested():
    for request in ['纸巾盒','纸巾包','盒子','纸巾,纸巾盒']:
        assert category_label_matches(request,'tissue box')


@pytest.mark.parametrize('name',['不是纸团','纸团旁边的瓶子','水瓶左边的海绵'])
def test_dont_silently_drop_negation_or_spatial_constraints(name):
    with pytest.raises(ValueError):normalize_prompt(name)


def test_generic_task_does_not_require_enrollment():
    request=task_detection_request('纸团',[])
    assert request['semantic'] and request['prompt']=='crumpled paper.'
    # A saved product with generic name must not lock recognition to one instance.
    request=task_detection_request('纸团',[dict(id='old',name='纸团')])
    assert request['semantic']


def test_display_name_does_not_invent_brand_or_identity():
    assert semantic_display_name('crumpled tissue')=='纸巾团'
    assert semantic_display_name('tissue box')=='纸巾盒'
    assert semantic_display_name('unseen shape')=='unseen shape'
