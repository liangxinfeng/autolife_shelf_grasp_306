"""Lightweight non-cylinder RGB-D targets; never infer hidden thickness blindly."""
import numpy as np
from .bottle import object_points,horizontal_basis,image_cropped,DepthQualityError
from .core import unit


def solid_center(points,up,product,camera):
    p=np.asarray(points,float);u,v,n=horizontal_basis(up)
    z=p@n;lo,hi=np.percentile(z,[1,99]);span=hi-lo
    if span<.025:raise DepthQualityError('可见物体有效深度高度不足25mm')
    middle=p[(z>=lo+.25*span)&(z<=lo+.75*span)]
    if len(middle)<40:raise ValueError('物体中央有效深度不足')
    xy=np.column_stack([middle@u,middle@v]);surface=np.median(xy,axis=0)
    away=unit(surface-np.array([camera@u,camera@v]))
    center_xy=surface+away*float(product['depth_m'])/2
    center=center_xy[0]*u+center_xy[1]*v+(lo+hi)*.5*n
    return center,{'observed_height_mm':float(span*1000),'geometry':'solid',
        'center_source':'visible_midpoint_plus_user_thickness','points':len(p)}


def bag_edge_target(points,pixels,point,up,product,camera):
    distances=np.linalg.norm(pixels-np.asarray(point),axis=1)
    local=points[distances<=10]
    if len(local)<18:raise ValueError('袋口选点附近有效深度不足，请选实体边缘而非细提绳')
    center=np.median(local,axis=0)
    keep=np.linalg.norm(local-center,axis=1)<.018
    if keep.mean()<.70:raise ValueError('袋口附近混入背景，请重新选点')
    center=np.median(local[keep],axis=0)
    up=unit(up);away=center-camera;away=unit(away-up*np.dot(away,up))
    center+=away*product['depth_m']/2
    return center,{'observed_height_mm':product['height_m']*1000,'geometry':'bag_edge',
        'center_source':'operator_selected_rim_patch','points':int(keep.sum())}


def estimate_object(frame,mask,T,up,product,cfg,grasp_pixel=None):
    bag=product.get('geometry')=='bag_edge'
    w,h,d=(float(product[k]) for k in ('diameter_m','height_m','depth_m'))
    if not (.015<=w<=(.35 if bag else .09) and .04<=h<=.4 and
            (.001 if bag else .005)<=d<=(.02 if bag else .15)):
        raise ValueError('物品尺寸配置无效')
    points,pixels,_=object_points(frame,mask,T,cfg)
    if product.get('geometry')=='bag_edge':
        if grasp_pixel is None:raise ValueError('请先在图像中选择袋口实体边缘夹取点')
        return bag_edge_target(points,pixels,grasp_pixel,up,product,T[:3,3])
    center,stats=solid_center(points,up,product,T[:3,3])
    stats['image_cropped']=image_cropped(mask)
    return center,stats
