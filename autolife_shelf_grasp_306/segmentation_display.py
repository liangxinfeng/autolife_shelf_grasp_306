"""Compact display contours only; original masks remain untouched for RGB-D."""
import cv2
import numpy as np


def mask_contours(mask):
    if mask is None:return []
    binary=np.asarray(mask,dtype=np.uint8)
    if binary.ndim!=2:raise ValueError('分割掩膜必须是二维图像')
    # RETR_TREE keeps holes and disconnected components. Qt uses odd-even fill.
    contours,_=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    result=[]
    for contour in contours:
        polygon=cv2.approxPolyDP(contour,.65,True).reshape(-1,2)
        if len(polygon)>=3:result.append(polygon.tolist())
    return result


def display_contours(item):
    # An inference item owns an immutable mask; repeated catalog messages reuse
    # the contour encoding rather than copying a full image through JSON.
    if '_display_contours' not in item:
        item['_display_contours']=mask_contours(item.get('mask'))
    return item['_display_contours']
