import numpy as np
from autolife_shelf_grasp_306.perception import duplicate_mask


def test_support_does_not_swallow_smaller_object():
    support=np.ones((100,100),bool);bag=np.zeros_like(support);bag[20:65,40:60]=True
    assert not duplicate_mask(bag,[40,20,60,65],[{'box':[0,0,100,100],'mask':support}])


def test_overlapping_instance_duplicate_is_removed():
    mask=np.ones((100,100),bool)
    assert duplicate_mask(mask,[0,0,100,100],[{'box':[1,1,99,99],'mask':mask}])
