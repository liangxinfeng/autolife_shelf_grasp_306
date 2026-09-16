import json
from types import SimpleNamespace
import numpy as np
import pytest
from autolife_shelf_grasp_306.core import ProductStore,match,product_category
from autolife_shelf_grasp_306.perception import Perception


def test_height_edit_persists_without_extra_picture(tmp_path):
    s=ProductStore(tmp_path);im=np.zeros((25,25,3),np.uint8)
    p=s.add('口香糖',80,50,[1,0],im)
    old=p['examples'][:]
    changed=s.update(p['id'],name='口香糖',diameter_mm=50,height_mm=80,geometry='cylinder',category='general')
    assert changed['diameter_m']==.05 and changed['height_m']==.08 and changed['depth_m']==.05
    assert changed['examples']==old and len(changed['features'])==1
    assert ProductStore(tmp_path).products[0]['height_m']==.08
    assert list((tmp_path/'history').glob('*.json'))


def test_delete_removes_matching_feature_and_keeps_recoverable_image(tmp_path):
    s=ProductStore(tmp_path);im=np.zeros((25,25,3),np.uint8)
    p=s.add('水',50,180,[1,0],im);p=s.add('水',50,180,[0,1],im)
    deleted=p['examples'][0];s.remove_example(p['id'],0)
    assert s.products[0]['features']==[[0.,1.]]
    assert len(s.products[0]['examples'])==1 and (tmp_path/deleted).is_file()
    s.remove_example(p['id'],0)
    assert match([1,0],s.products)['id'] is None


def test_invalid_edit_does_not_change_file(tmp_path):
    s=ProductStore(tmp_path);im=np.zeros((25,25,3),np.uint8)
    a=s.add('甲',50,80,[1,0],im);s.add('乙',50,80,[0,1],im)
    original=s.path.read_bytes()
    with pytest.raises(ValueError):s.update(a['id'],name='乙',diameter_mm=50,height_mm=80)
    with pytest.raises(ValueError):s.remove_example(a['id'],100)
    assert s.path.read_bytes()==original


def test_legacy_drinks_and_new_objects_have_distinct_categories():
    assert product_category({'name':'水'})=='beverage'
    assert product_category({'name':'口香糖','geometry':'cylinder'})=='general'
    assert product_category({'geometry':'cylinder','category':'beverage'})=='beverage'


def test_drink_mode_skips_generic_inference():
    class Detector:
        def __init__(self):self.calls=[]
        def predict(self,*args,**kwargs):self.calls.append(kwargs);return [SimpleNamespace(masks=None)]
    p=Perception.__new__(Perception);p.yolo=Detector();p.generic=Detector();p.device='cpu'
    p.cfg=dict(image_size=640,confidence=.3,classes=[24,39],match_threshold=.72,match_margin=.06)
    p.previous=[];p.counter=0
    frame={'image':np.zeros((30,30,3),np.uint8)};store=SimpleNamespace(products=[])
    p.infer(frame,store,'beverages');assert p.yolo.calls[-1]['classes']==[39,40,41] and not p.generic.calls
    p.infer(frame,store,'all');assert len(p.generic.calls)==1
    with pytest.raises(ValueError):p.infer(frame,store,'bad')


def test_modes_share_same_sku_dimensions_and_samples():
    class Array:
        def __init__(self,a):self.a=np.asarray(a)
        def cpu(self):return self
        def numpy(self):return self.a
    mask=np.zeros((30,30));mask[3:27,3:27]=1
    detected=SimpleNamespace(masks=SimpleNamespace(data=Array([mask])),
        boxes=SimpleNamespace(xyxy=Array([[3,3,27,27]]),conf=Array([.9])))
    p=Perception.__new__(Perception);p.yolo=SimpleNamespace(predict=lambda *a,**k:[detected])
    p.generic=SimpleNamespace(predict=lambda *a,**k:[SimpleNamespace(masks=None)])
    p.cfg=dict(image_size=640,confidence=.3,classes=[39],match_threshold=.72,match_margin=.06)
    p.device='cpu';p.previous=[];p.counter=0;p.embed=lambda crops:np.array([[1.,0.] for c in crops])
    product=dict(id='shared',name='饮料',diameter_m=.05,height_m=.08,category='general',features=[[1.,0.]],examples=['test.jpg'])
    store=SimpleNamespace(products=[product]);frame={'image':np.zeros((30,30,3),np.uint8)}
    for mode in ('all','beverages','all'):
        items,_=p.infer(frame,store,mode)
        assert len(items)==1 and items[0]['product'] is product
        assert items[0]['product']['height_m']==.08 and items[0]['name']=='饮料'
