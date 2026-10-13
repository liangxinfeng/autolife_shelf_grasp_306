"""Session-only RGB-D geometry. Enrollment is optional, evidence is not."""
import re
import uuid
import numpy as np
from .bottle import object_points,horizontal_basis,image_cropped,estimate,DepthQualityError
from .core import unit
from .object_extent import coherent_component,oriented_bounds,footprint_vertices,bbox


def target_id(item,session):
    if item.get('product'):return item['product']['id']
    return 'temporary_'+uuid.uuid5(uuid.NAMESPACE_URL,str(session)+':'+str(item['id'])).hex


def same_target(chosen,fresh,session):
    if fresh is None or chosen['id']!=fresh['id'] or target_id(chosen,session)!=target_id(fresh,session):return False
    if chosen.get('product'):return True
    # A nearest bounding box alone is not sufficient identity for unknown goods.
    a,b=chosen['mask'],fresh['mask']
    if a.shape!=b.shape or np.logical_and(a,b).sum()/max(1,np.logical_or(a,b).sum())<.65:return False
    x,y=np.asarray(chosen['feature']),np.asarray(fresh['feature'])
    return bool(np.dot(x,y)/max(1e-12,np.linalg.norm(x)*np.linalg.norm(y))>=.85)


def temporary_product(item,session):
    label=str(item.get('semantic_label','')).lower()
    words=set(re.findall(r'[a-z]+',label))
    if words&{'person','hand','arm','human','cat','dog','table','tabletop','chair','bag','cloth','towel','rope'}:
        raise ValueError('该目标不是可直接自动估尺寸的小型硬质物品；袋口等柔性目标请使用专门夹取点设置')
    cylinder=bool(words&{'bottle','can'})
    return dict(id=target_id(item,session),name=item.get('name') or label or '未录入物品',
        temporary=True,geometry='cylinder' if cylinder else 'solid',category='beverage' if cylinder else 'general',
        dimension_source='multi_frame_rgbd',semantic_label=label)


def measured_bounds(points,up,camera):
    p=np.asarray(points,float);u,v,n=horizontal_basis(up)
    horizontal=np.median(p,axis=0)-camera;away=unit(horizontal-n*(horizontal@n));side=unit(np.cross(n,away))
    basis=np.column_stack([side,away,n]);local=p@basis
    low,high=np.percentile(local,[1,99],axis=0)
    return local,low,high,basis


def visible_surface(local,low,high,basis,camera):
    """Certify measured extent, not shape class. Never invent hidden thickness."""
    width,depth,height=high-low
    def coverage(fraction):
        top=local[local[:,2]>high[2]-max(.008,height*fraction)]
        if len(top)<40:return None
        spans=np.diff(np.percentile(top[:,:2],[5,95],axis=0),axis=0)[0]
        if spans[0]>=width*.5 and spans[1]>=max(.006,depth*.65):
            return (spans/np.array([width,depth])).tolist()
        return None
    # Preserve the original flat-top path for boxes.
    for fraction in (.12,.30):
        spans=coverage(fraction)
        if spans is not None:
            return dict(surface_evidence='visible_top',surface_coverage=spans)
    # A single PCA normal describes a plane, not a wrinkled or tilted object.
    # Check actual upper surface patches instead of requiring the whole cloud
    # (including vertical sides) to have a horizontal best-fit plane.
    view=np.asarray(camera)@basis-(low+high)*.5
    elevation=float(np.degrees(np.arcsin(np.clip(view[2]/max(1e-9,np.linalg.norm(view)),-1,1))))
    singular=np.linalg.svd(local-np.median(local,axis=0),compute_uv=False)
    volume=float(singular[-1]/max(singular[1],1e-9))
    thickness=float(singular[-1]/np.sqrt(len(local)))
    support=upper_surface_support(local,low,high) if elevation>=45 and volume>=.20 and thickness>=.004 else None
    if support is not None:
        return dict(surface_evidence='irregular_visible_surface',**support,
                    view_elevation_deg=elevation,surface_volume_ratio=volume)
    raise DepthQualityError('已有物体深度，但上部表面覆盖不足或仅看见单面，尚不能确认前后尺寸；请调整观察角度')


def upper_surface_support(local,low,high):
    """Bounded local PCA; evidence points only, no filling unseen geometry.

    Consistent upper-facing patches must span both horizontal dimensions.
    Keeping the lower 30% out avoids accepting a table fringe as the top of
    the object. A broad noisy wall cannot pass the local planarity test.
    """
    from scipy.spatial import cKDTree
    if len(local)<80:return None
    centers=local[np.linspace(0,len(local)-1,min(512,len(local)),dtype=int)]
    distances,neighbors=cKDTree(local).query(centers,k=24)
    patches=local[neighbors]
    patches=patches-patches.mean(axis=1,keepdims=True)
    values,axes=np.linalg.eigh(np.einsum('nki,nkj->nij',patches,patches)/24)
    good=(centers[:,2]>high[2]-(high[2]-low[2])*.70)
    good&=(distances[:,-1]<=.015)&(values[:,1]>1e-8)
    good&=(values[:,0]/np.maximum(values[:,1],1e-12)<.25)&(np.abs(axes[:,2,0])>=.65)
    if good.sum()<max(40,len(centers)*.20):return None
    spans=np.ptp(np.percentile(centers[good,:2],[5,95],axis=0),axis=0)
    if np.any(spans<np.maximum(.006,(high-low)[:2]*.50)):return None
    return dict(surface_coverage=(spans/(high-low)[:2]).tolist(),
                upper_surface_fraction=float(good.mean()),upper_surface_patches=int(good.sum()))


def estimate_temporary(frame,mask,T,up,template,cfg,reference=None):
    if image_cropped(mask):
        return estimate_partial(frame,mask,T,up,template,cfg,reference)
    points,pixels,bounds=object_points(frame,mask,T,cfg)
    if template['geometry']=='cylinder':
        local,low,high,basis=measured_bounds(points,up,T[:3,3]);width,depth,height=high-low
        body=local[(local[:,2]>low[2]+height*.18)&(local[:,2]<low[2]+height*.7)]
        if len(body)<80:raise DepthQualityError('未录入瓶身中央深度不足')
        width=float(np.diff(np.percentile(body[:,0],[.5,99.5]))[0]+.002)
        if not .015<=width<=.09 or not .04<=height<=.4:
            raise DepthQualityError('实测瓶身尺寸超出夹爪可用范围，请检查分割和深度')
        product={**template,'diameter_m':width,'height_m':float(height),'depth_m':width}
        center,stats=estimate(frame,mask,T,up,product,cfg)
    else:
        keep=coherent_component(points);points=points[keep];pixels=pixels[keep]
        local,low,high,basis=oriented_bounds(points,up);width,depth,height=high-low
        if not (.015<=width<=.35 and .005<=depth<=.35 and .025<=height<=.4):
            raise DepthQualityError('实测物体过薄、过大或深度混入背景，无法建立可靠夹取几何')
        evidence=visible_surface(local,low,high,basis,T[:3,3])
        center=basis@((low+high)*.5)
        product={**template,'diameter_m':float(width),'height_m':float(height),'depth_m':float(depth),
            'bbox_axes_torso':basis.tolist(),'bbox_extents_m':[float(width+.004),float(depth+.004),float(height)],
            'footprint_vertices_torso':footprint_vertices(points,up,center).tolist()}
        stats=dict(observed_height_mm=float(height*1000),geometry='solid',height_source='observed_object_midpoint',
            center_source='coherent_oriented_body_bounds',image_cropped=False,points=len(points),
            depth_component_ratio=float(keep.mean()),**evidence)
    ys=pixels[:,1];y0,y1=bounds
    if np.percentile(ys,1)>y0+.12*(y1-y0) or np.percentile(ys,99)<y1-.12*(y1-y0):
        raise DepthQualityError('未录入物品上下边界深度不足，无法确认中心和放置高度')
    stats.update(temporary_product=product,dimension_source='multi_frame_rgbd')
    return center,stats


def estimate_partial(frame,mask,T,up,template,cfg,reference):
    """Fresh visible-part target, with full-view reference for hidden extent."""
    if reference is None:
        raise DepthQualityError('局部可见物品尚无完整尺寸记录：请先完整入镜规划一次，之后可按可见部分夹取，无需录入商品')
    if reference.get('geometry')!=template['geometry']:raise DepthQualityError('物体形状与近期完整观察不一致')
    if np.any(mask[:,:2]) or np.any(mask[:,-2:]):
        raise DepthQualityError('夹取部位左右边界出画面，尚不能确认指间开度；请露出两侧边缘')
    product={**template,**{k:reference[k] for k in ('diameter_m','height_m','depth_m')},
             'geometry_reference':'recent_full_view','reference_age_s':reference['reference_age_s']}
    if product['geometry']=='cylinder':center,stats=estimate(frame,mask,T,up,product,cfg)
    else:
        from .object_geometry import estimate_object
        center,stats=estimate_object(frame,mask,T,up,product,cfg)
    if stats['observed_height_mm']/1000>product['height_m']+.015:
        raise DepthQualityError('当前可见尺寸超过完整观察记录，物体可能改变，请重新完整观察')
    stats.update(temporary_product=product,image_cropped=True,geometry_reference='recent_full_view',
                 dimension_source='recent_full_view_with_live_depth',height_source='visible_part_midpoint')
    return center,stats


def combine_product(template,samples):
    if len(samples)<5:raise ValueError('未录入物品需要至少5帧可靠尺寸')
    products=[s[1]['temporary_product'] for s in samples]
    if any(p['id']!=template['id'] or p['geometry']!=template['geometry'] for p in products):
        raise ValueError('临时物品数据不一致，请重新识别')
    dimensions=np.array([[p[k] for k in ('diameter_m','height_m','depth_m')] for p in products])
    if not np.isfinite(dimensions).all() or np.any(dimensions<=0):raise ValueError('物体尺寸数据无效')
    reference=None
    if any('bbox_axes_torso' in p for p in products):
        if not all('bbox_axes_torso' in p for p in products):raise ValueError('物体包络数据不完整')
        axes=[bbox(p)[0] for p in products];directions=np.array([a[:,0] for a in axes])
        reference=axes[int(np.argmin(np.arccos(np.clip(np.abs(directions@directions.T),0,1)).sum(axis=1)))]
        for i,p in enumerate(products):
            vertices=np.asarray(p['footprint_vertices_torso'],float)
            if vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)<3 or not np.isfinite(vertices).all():
                raise ValueError('物体轮廓数据无效')
            span=np.ptp(vertices@reference,axis=0)
            dimensions[i,[0,2]]=span[:2]
    from .multiframe import consensus_indices
    indices=consensus_indices([s[0] for s in samples],dimensions)
    raw_spread=np.ptp(dimensions,axis=0)*1000
    observed=len(samples);samples=[samples[i] for i in indices];products=[products[i] for i in indices]
    dimensions=dimensions[indices];median=np.median(dimensions,axis=0)
    result={**template,**dict(zip(('diameter_m','height_m','depth_m'),map(float,median))),
            'dimension_frames':len(samples),'dimension_observed_frames':observed,'dimension_inlier_indices':indices,
            'dimension_fusion':'common_axes_majority','dimension_raw_spread_mm':raw_spread.tolist(),
            'dimension_spread_mm':(np.ptp(dimensions,axis=0)*1000).tolist()}
    references=[s[1].get('geometry_reference') for s in samples]
    if any(references):
        if not all(v=='recent_full_view' for v in references):raise ValueError('完整与局部观测混合，请停稳后重新规划')
        result.update(geometry_reference='recent_full_view',dimension_source='recent_full_view_with_live_depth')
    if reference is not None:
        center=np.median([s[0] for s in samples],axis=0);cloud=[];heights=[]
        for p,(c,_) in zip(products,samples):
            vertices=np.asarray(p['footprint_vertices_torso']);cloud.extend((vertices+c-center)@reference)
            heights.append(abs((c-center)@reference[:,2])+p['height_m']/2)
        cloud=np.asarray(cloud);extent=np.r_[2*np.max(np.abs(cloud[:,:2]),axis=0)+.004,2*max(heights)]
        result.update(bbox_axes_torso=reference.tolist(),bbox_extents_m=extent.tolist())
        from .local_contact import merged_footprint
        result['measured_footprint_torso']=merged_footprint(products,[s[0] for s in samples],center,reference[:,2])
    return result
