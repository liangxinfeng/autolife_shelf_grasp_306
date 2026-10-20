import numpy as np
import pytest
from autolife_shelf_grasp_306.core import ProductStore
from autolife_shelf_grasp_306.object_geometry import solid_center,bag_edge_target


def test_old_cylinder_and_new_shapes_share_product_store(tmp_path):
    store=ProductStore(tmp_path);image=np.zeros((30,30,3),np.uint8)
    can=store.add('口香糖罐',50,90,[1,0,0],image)
    solid=store.add('西瓜模型',45,120,[0,1,0],image,'solid',30)
    bag=store.add('袋子',130,260,[0,0,1],image,'bag_edge',3)
    assert can['geometry']=='cylinder' and solid['depth_m']==.03 and bag['diameter_m']==.13
    assert len(ProductStore(tmp_path).products)==3
    with pytest.raises(ValueError):store.add('宽物体',130,100,[1,0,0],image,'solid',30)
    changed=store.add('西瓜模型',45,120,[0,1,0],image,'solid',40)
    assert changed['id']==solid['id'] and changed['depth_m']==.04


@pytest.mark.parametrize('reference_height',[.04,.12,.4])
def test_non_cylinder_center_uses_user_thickness_not_radius_fit(reference_height):
    y,z=np.meshgrid(np.linspace(-.025,.025,30),np.linspace(0,.12,60))
    p=np.column_stack([np.full(y.size,.6),y.ravel(),z.ravel()])
    c,stats=solid_center(p,[0,0,1],{'height_m':reference_height,'depth_m':.03},np.zeros(3))
    np.testing.assert_allclose(c,[.615,0,.06],atol=.002)
    assert stats['geometry']=='solid'


def test_bag_rim_is_local_not_whole_bag_center():
    x,y=np.meshgrid(np.arange(20,40),np.arange(20,40))
    pixels=np.column_stack([x.ravel(),y.ravel()])
    points=np.column_stack([np.full(x.size,.6),(x.ravel()-30)*.0005,(y.ravel()-30)*.0005+.2])
    c,_=bag_edge_target(points,pixels,[30,30],[0,0,1],{'height_m':.3,'depth_m':.004},np.zeros(3))
    np.testing.assert_allclose(c,[.602,0,.2],atol=.001)
    with pytest.raises(ValueError,match='有效深度不足'):
        bag_edge_target(points,pixels,[80,80],[0,0,1],{'height_m':.3,'depth_m':.004},np.zeros(3))
