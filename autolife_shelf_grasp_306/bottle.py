"""Bottle-only 3D geometry. No support-plane estimate or clicked height.

Vectors are in the URDF torso frame, whose +Z is NOT assumed to point up.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from .core import unit


class DepthQualityError(ValueError):
    """A fresh frame may recover; invalid configuration/alignment is not this."""


def horizontal_basis(up,yaw=0.):
    up=unit(up);forward=np.array([np.cos(yaw),np.sin(yaw),0.])
    forward=unit(forward-up*np.dot(forward,up))
    return forward,unit(np.cross(up,forward)),up


def object_points(frame,mask,T,cfg):
    from autolife_handeye_calibration_306.depth_target import aligned_depth
    if frame.get('depth') is None:raise ValueError('等待同帧深度')
    aligned_depth(frame['image'],frame['depth'],frame['frame_id'],frame['depth_id'],frame['intrinsics'],frame['depth_intrinsics'])
    if mask.shape!=frame['depth'].shape:raise ValueError('分割与深度尺寸不同')
    if np.any(np.abs(frame['D'])>1e-9) and frame['intrinsics']['model_id'] not in (0,4):
        raise ValueError('深度畸变模型未支持')
    y,x=np.where(mask)
    if len(x)<100:raise ValueError('分割区域过小')
    h,w=mask.shape
    safe=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    # Cropped objects are allowed; omit image-edge depth rather than rejecting
    # the entire visible object or padding unobserved pixels with fake geometry.
    safe[:2,:]=False;safe[-2:,:]=False;safe[:,:2]=False;safe[:,-2:]=False
    rows,cols=np.where(safe);rows,cols=rows[::2],cols[::2]
    z=frame['depth'][rows,cols].astype(float)*cfg['depth_scale_m']
    lo,hi=cfg['depth_range_m'];valid=np.isfinite(z)&(z>=lo)&(z<=hi)
    if valid.sum()<80 or valid.mean()<.50:raise DepthQualityError('瓶身有效深度不足，请避开反光')
    rows,cols,z=rows[valid],cols[valid],z[valid]
    # Unlike the old global +/-65mm optical-depth cut, preserve vertical bottle
    # extent at oblique viewing angles. Radial rejection happens in 3D below.
    rays=cv2.undistortPoints(np.column_stack([cols,rows]).astype(float)[:,None,:],frame['K'],frame['D']).reshape(-1,2)
    points=np.column_stack([rays*z[:,None],z])@T[:3,:3].T+T[:3,3]
    return points,np.column_stack([cols,rows]),(int(y.min()),int(y.max()))


def center_from_points(points,up,product,camera):
    """Robust known-radius cylinder fit + observed vertical midpoint.

    Recorded height is reference-only; target height is the visible midpoint.
    Neck/cap points are excluded from the radius fit but used for vertical extent.
    """
    p=np.asarray(points,float);u,v,n=horizontal_basis(up)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<80 or not np.isfinite(p).all():raise ValueError('瓶身点云不足')
    diameter=float(product['diameter_m']);height=float(product['height_m']);r=diameter/2
    if not .015<=diameter<=.09 or not .04<=height<=.40:raise ValueError('商品尺寸无效')
    z=p@n;low,high=np.percentile(z,[1,99]);xy=np.column_stack([p@u,p@v])
    body=(z>low+(high-low)*.18)&(z<low+(high-low)*.70)
    if body.sum()<40:raise DepthQualityError('瓶身中央深度不足')
    middle=np.median(xy[body],axis=0)
    camera_xy=np.array([np.dot(camera,u),np.dot(camera,v)])
    away=unit(middle-camera_xy);seed=middle+r*away
    body_xy=xy[body]
    # Only local foreground data participates in the fit, not a reflection's
    # distant background. A substantial inlier population remains mandatory.
    body_xy=body_xy[np.linalg.norm(body_xy-middle,axis=1)<diameter*1.3]
    if len(body_xy)<40:raise DepthQualityError('瓶身深度混入背景')
    def residual(c):return np.r_[np.linalg.norm(body_xy-c,axis=1)-r,.08*(c-seed)]
    fit=least_squares(residual,seed,bounds=(seed-r*.8,seed+r*.8),loss='soft_l1',f_scale=.003,max_nfev=40)
    err=np.abs(np.linalg.norm(body_xy-fit.x,axis=1)-r)
    if not fit.success or np.median(err)>.006 or np.percentile(err,80)>.012:
        raise DepthQualityError(f'瓶身深度拟合不稳定：录入直径{diameter*1000:.0f}mm，'
                         f'拟合偏差{np.median(err)*1000:.1f}mm/P80 {np.percentile(err,80)*1000:.1f}mm；'
                         '请检查瓶身遮挡、反光及实际尺寸')
    if np.dot(fit.x-middle,away)<r*.15:raise DepthQualityError('瓶身中心方向不确定')
    inliers=np.linalg.norm(xy-fit.x,axis=1)<r+.010
    low,high=np.percentile(z[inliers],[.5,99.5]);span=high-low
    if span<.025:raise DepthQualityError('可见瓶身有效深度高度不足25mm，无法稳定确定夹取中心')
    center=fit.x[0]*u+fit.x[1]*v+(low+high)*.5*n
    stats={'observed_height_mm':float(span*1000),'height_source':'visible_bottle_midpoint',
           'reference_height_mm':height*1000,
           'radius_m':r,'cylinder_median_mm':float(np.median(err)*1000),'points':len(p)}
    return center,stats,inliers


def estimate(frame,mask,T,up,product,cfg):
    points,pixels,bounds=object_points(frame,mask,T,cfg)
    center,stats,keep=center_from_points(points,up,product,T[:3,3])
    ys=pixels[keep,1];y0,y1=bounds;span=max(1,y1-y0)
    if len(ys)<80 or np.percentile(ys,1)>y0+span*.12 or np.percentile(ys,99)<y1-span*.12:
        raise DepthQualityError('当前可见区域的上下边缘深度缺失过多，无法稳定确定可见部分中心')
    stats['image_cropped']=image_cropped(mask)
    return center,stats


def image_cropped(mask):
    return bool(np.any(mask[:2]) or np.any(mask[-2:]) or np.any(mask[:,:2]) or np.any(mask[:,-2:]))


def planning_product(product,geometry):
    """Do not overwrite the catalog or shrink the unseen carried-object volume.

    Full-view objects use measured height. For image-cropped objects, catalog
    height remains only a conservative hidden-extent estimate for collisions;
    it never moves the measured visible midpoint or blocks height mismatch.
    """
    result=dict(product)
    if product.get('geometry','cylinder')=='bag_edge':return result
    observed=float(geometry['observed_height_mm'])/1000.
    if not np.isfinite(observed) or observed<.025:raise ValueError('可见高度无效')
    result['reference_height_m']=product['height_m']
    result['height_m']=observed
    # A clipped top/bottom may put the grasp point away from the whole object's
    # midpoint. Bound that asymmetry instead of assuming symmetric half-height.
    if geometry.get('image_cropped'):
        result['collision_height_m']=observed+2*max(0.,product['height_m']-observed)
    else:result['collision_height_m']=observed
    return result


def combine(samples):
    if len(samples)<5:raise ValueError('有效深度帧不足，请重试')
    points=np.array([s[0] for s in samples]);center=np.median(points,axis=0)
    spread=float(np.max(np.linalg.norm(points-center,axis=1)))
    if spread>.010:raise ValueError('多帧瓶子中心波动超过10mm，请保持物体不动')
    shape=samples[0][1].get('geometry','cylinder')
    return center,{'source':'bottle_only_rgbd' if shape=='cylinder' else shape+'_rgbd','frames':len(samples),'center_spread_mm':spread*1000,
                  'observed_height_mm':float(np.median([s[1]['observed_height_mm'] for s in samples])),
                  'height_source':samples[0][1].get('height_source','visible_midpoint'),
                  'image_cropped':any(s[1].get('image_cropped',False) for s in samples),
                  'support_required':False}
