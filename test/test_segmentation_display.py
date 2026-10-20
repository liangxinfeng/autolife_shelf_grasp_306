import cv2
import numpy as np
from autolife_shelf_grasp_306.segmentation_display import mask_contours,display_contours


def test_empty_and_missing_mask():
    assert mask_contours(None)==[]
    assert mask_contours(np.zeros((50,80),bool))==[]


def test_real_mask_not_rectangle_and_no_mutation():
    mask=np.zeros((120,120),np.uint8);cv2.circle(mask,(60,60),30,1,-1)
    original=mask.copy();polygons=mask_contours(mask)
    assert len(polygons)==1 and len(polygons[0])>4
    assert 2600<cv2.contourArea(np.array(polygons[0],np.int32))<3000
    assert np.array_equal(mask,original)


def test_holes_and_separate_parts_preserved():
    mask=np.zeros((140,140),np.uint8);mask[10:90,10:90]=1;mask[30:70,30:70]=0;mask[105:130,105:130]=1
    assert len(mask_contours(mask))==3


def test_display_cache_not_raw_mask():
    item={'mask':np.ones((25,35),bool)}
    a=display_contours(item);assert a is display_contours(item)
    assert isinstance(a[0][0][0],int) and item['mask'].sum()==875
