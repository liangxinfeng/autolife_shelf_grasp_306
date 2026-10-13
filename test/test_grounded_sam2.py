"""Offline prompt, proposal, mask and SKU compatibility contracts."""
import numpy as np
import pytest
from types import SimpleNamespace
from autolife_shelf_grasp_306.grounded_sam2 import normalize_prompt,prompt_for,select_boxes,local_model_paths
from autolife_shelf_grasp_306.perception import Perception,model_paths


@pytest.mark.parametrize('text,result',[
 ('瓶子、杯子，袋子','bottle. cup. bag.'),(' 红色的杯子 ','red cup.'),
 ('red bottle; BLUE cup.','red bottle. blue cup.'),('瓶子,bottle','bottle.'),('',''),
 ('口香糖罐，西瓜模型','chewing gum container. watermelon shaped object.')])
def test_prompt(text,result):assert normalize_prompt(text)==result


@pytest.mark.parametrize('text,result',[
 ('白色鼠标','white mouse.'),('白色的鼠标','white mouse.'),('白色电脑鼠标','white mouse.'),
 ('白鼠标','white mouse.'),('白色的无线鼠标','white wireless mouse.'),
 ('帮我识别一下白色鼠标','white mouse.'),('白色 鼠标','white mouse.'),
 ('白色鼠标和黑色键盘','white mouse. black keyboard.'),
 ('透明塑料瓶子','transparent plastic bottle.'),('白色的小杯子','white small cup.'),
 ('白色鼠标，white mouse','white mouse.'),('粉红色纸袋','pink paper bag.')])
def test_chinese_attributes_match_english(text,result):assert normalize_prompt(text)==result


@pytest.mark.parametrize('text',['瓶子旁边的鼠标','不是白色的鼠标','特殊品牌鼠标'])
def test_unknown_qualifier_never_silently_discarded(text):
 with pytest.raises(ValueError,match='未能完整理解'):normalize_prompt(text)


@pytest.mark.parametrize('text',['未知的新商品名字','a'*301,'bottle <script>','\x00',','.join('object '+str(i) for i in range(17))])
def test_invalid_text_is_explicit_not_fake_translation(text):
 with pytest.raises(ValueError):normalize_prompt(text)


def test_prompt_modes_and_blank_restore_default():
 cfg={'grounded_sam2_prompts':{'all':'bottle. bag.','beverages':'bottle.'},'semantic_prompt_overrides':{'all':'杯子'}}
 assert prompt_for(cfg,'all')=='cup.' and prompt_for(cfg,'beverages')=='bottle.'
 cfg['semantic_prompt_overrides']['all']=''
 assert prompt_for(cfg,'all')=='bottle. bag.'


def test_nms_synonyms_keeps_separate_instances_and_clips_partial_box():
 boxes=[[-3,0,20,40],[0,0,20,40],[40,2,60,40],[0,0,float('nan'),10],[2,2,1,3]]
 selected=select_boxes(boxes,[.9,.8,.7,.99,.95],['bottle','drink','bottle','bad','bad'],(60,60,3))
 assert len(selected)==2 and np.allclose(selected[0]['box'],[0,0,20,40])


def test_no_detection_is_valid_and_no_fake_mask():
 assert select_boxes([],[],[],(100,100,3))==[]


def test_model_paths_never_require_yolo_in_new_backend(tmp_path):
 cfg={'detector_backend':'grounded_sam2','grounding_dino_model':'gd','sam2_model':'sam',
      'dino_repo':'dino','dino_weights':'dino.pth'}
 for folder in ('gd','sam'):
  p=tmp_path/folder;p.mkdir()
  for f in ['config.json','model.safetensors','preprocessor_config.json','tokenizer_config.json','tokenizer.json']:(p/f).write_text('{}')
 (tmp_path/'dino').mkdir();(tmp_path/'dino/hubconf.py').write_text('# test')
 (tmp_path/'dino.pth').write_text('test')
 paths=model_paths(cfg,tmp_path)
 assert 'yolo_weights' not in paths and 'generic_weights' not in paths
 (tmp_path/'gd/model.safetensors').unlink()
 with pytest.raises(ValueError,match='未完整'):model_paths(cfg,tmp_path)


def test_new_masks_keep_existing_product_matching_without_yolo():
 p=Perception.__new__(Perception);p.cfg={'match_threshold':.72,'match_margin':.06};p.previous=[];p.counter=0
 mask=np.ones((20,20),bool)
 p.segmenter=SimpleNamespace(proposals=lambda frame,mode:[dict(box=[0,0,20,20],mask=mask,
     crop=np.zeros((20,20,3),np.uint8),semantic_label='bottle',source='grounded_sam2',detection_score=.9)])
 p.embed=lambda crops:np.array([[1.,0.] for c in crops])
 product=dict(id='water',name='现有矿泉水',features=[[1.,0.]],diameter_m=.07,height_m=.20)
 store=SimpleNamespace(products=[product])
 first,_=p.infer({'image':None},store,'all');second,_=p.infer({'image':None},store,'beverages')
 assert first[0]['product'] is product and first[0]['name']=='现有矿泉水'
 assert first[0]['id']==second[0]['id'] and first[0]['mask'] is mask
 store.products=[]
 unknown,_=p.infer({'image':None},store)
 assert unknown[0]['product'] is None and unknown[0]['name'].startswith('未录入')
