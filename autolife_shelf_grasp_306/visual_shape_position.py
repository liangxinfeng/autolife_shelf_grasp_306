"""已确认物体身份后，用完整轮廓与分布式深度对应估计平移。

不把补录时的摆放位置当作初始位置，也不拿部分点云的质心差当运动。
本方法只服务“只纠正位置”：轮廓相似且视角接近时，在归一化轮廓内
取分散的深度对应，求一致的平移。无法观测的绕轴角不伪装成已测量。
"""
import cv2
import numpy as np
from .visual_reference import matrix


def _outline(arrays):
    mask=np.asarray(arrays['mask'],bool)
    if mask.ndim!=2 or mask.shape!=np.asarray(arrays['depth']).shape:
        raise ValueError('物体轮廓与深度尺寸不一致，请重新取景')
    from .visual_box_selection import dominant_component
    mask=dominant_component(mask)
    y,x=np.nonzero(mask)
    if len(x)<100:raise ValueError('请框选完整物体，当前轮廓太小')
    h,w=mask.shape
    lo=np.array([x.min(),y.min()]);hi=np.array([x.max(),y.max()])
    if np.min(hi-lo)<8:raise ValueError('物体轮廓太窄，请靠近观察')
    crop=mask[lo[1]:hi[1]+1,lo[0]:hi[0]+1].astype(np.uint8)
    shape=cv2.resize(crop,(64,96),interpolation=cv2.INTER_NEAREST)>0
    return mask,lo,hi,shape


def _edge_flags(outline):
    mask,lo,hi,_=outline[:4]
    h,w=mask.shape
    # 贴边一像素并不意味着物体已不可识别。
    return lo<=1,hi>=np.array([w-2,h-2])


def _partial_outline(reference,current,oa,ob,cfg):
    """轻微裁边时用完整参考的投影尺寸恢复采样坐标，不给缺失区域伪造深度。

    原归一化包围框会把裁短后的物体拉伸，产生半个缺口大小的位置偏差。
    使用深度与相机内参推算尺寸，只在仍可见的另一端锚定；采样仍仅取真实像素。
    """
    left,right=_edge_flags(ob)
    if not (left.any() or right.any()):return ob,False
    if any(x.any() for x in _edge_flags(oa)):
        raise ValueError('参考物体已超出画面，缺少完整尺寸；请补录一个完整参考视角')
    if np.any(left&right):raise ValueError('物体超出画面较多，请稍微后退让物体完整些')
    def depth_of(arr,mask):
        d=np.asarray(arr['depth'],float)*float(cfg['depth_scale_m'])
        low,high=cfg['depth_range_m'];v=d[mask&np.isfinite(d)&(d>=low)&(d<=high)]
        if len(v)<100:raise ValueError('物体轻微裁边，但有效深度不足，请调整视角')
        return float(np.median(v))
    za=depth_of(reference,oa[0]);zb=depth_of(current,ob[0])
    scale=np.diag(current['K'])[:2]/np.diag(reference['K'])[:2]*za/zb
    expected=(oa[2]-oa[1])*scale
    lo=ob[1].astype(float).copy();hi=ob[2].astype(float).copy()
    measured=hi-lo
    # 最多容忍各方向完整尺寸的8%，且最多12像素；不是取消所有裁边判断。
    for axis in (0,1):
        gap=expected[axis]-measured[axis]
        if left[axis] or right[axis]:
            if gap < -max(2.,expected[axis]*.04) or gap>min(12.,expected[axis]*.08)+1e-6:
                raise ValueError('物体超出画面较多或尺寸不一致，请稍微调整取景')
            if gap>0:
                if left[axis]:lo[axis]-=gap
                else:hi[axis]+=gap
        elif abs(gap)>max(2.,expected[axis]*.08):
            raise ValueError('裁边物体与参考轮廓尺寸不一致，请调整观察方向')
    return (ob[0],lo,hi,ob[3]),True


def _oriented_outline(arrays):
    """按完整轮廓主方向展开，避免横向包围框把透视倾斜当作形状变化。"""
    mask,lo,hi,shape=_outline(arrays)
    y,x=np.nonzero(mask);xy=np.column_stack((x,y)).astype(float);center=xy.mean(0)
    values,vectors=np.linalg.eigh(np.cov((xy-center).T))
    if values[1]<1.35*max(values[0],1.):return mask,lo,hi,shape
    along=vectors[:,1]
    if along[1]<0:along=-along
    across=np.array([along[1],-along[0]]);basis=np.column_stack((across,along))
    uv=(xy-center)@basis;lo=uv.min(0);hi=uv.max(0)
    gx,gy=np.meshgrid(np.linspace(0,1,64),np.linspace(0,1,96))
    pixels=(lo+np.column_stack((gx.ravel(),gy.ravel()))*(hi-lo))@basis.T+center
    px,py=np.rint(pixels).astype(int).T;h,w=mask.shape
    valid=(px>=0)&(px<w)&(py>=0)&(py<h);normalized=np.zeros(len(px),bool)
    normalized[valid]=mask[py[valid],px[valid]]
    return mask,lo,hi,normalized.reshape(96,64),basis,center


def _samples(arrays,T,cfg,grid,outline):
    mask,lo,hi,_=outline[:4]
    pixels=lo+grid*(hi-lo)
    if len(outline)==6:pixels=pixels@outline[4].T+outline[5]
    x,y=np.rint(pixels).astype(int).T;h,w=mask.shape
    inside=(x>=1)&(x<w-1)&(y>=1)&(y<h-1);x=x.clip(1,w-2);y=y.clip(1,h-2)
    depth=np.asarray(arrays['depth'],float)*float(cfg['depth_scale_m'])
    z=depth[y,x];low,high=cfg['depth_range_m']
    valid=inside&mask[y,x]&np.isfinite(z)&(z>=low)&(z<=high)
    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
        near=depth[y+dy,x+dx]
        valid&=np.isfinite(near)&(np.abs(near-z)<.015)
    rays=cv2.undistortPoints(pixels[:,None,:].astype(float),arrays['K'],arrays['D']).reshape(-1,2)
    points=np.column_stack((rays*z[:,None],z))@T[:3,:3].T+T[:3,3]
    return points,pixels,valid


def _estimate(reference,reference_camera,current,current_camera,cfg,oriented=False):
    """调用前必须已通过某个保存视角认出物体；返回初始帧到当前帧的平移。"""
    aT=matrix(reference_camera);bT=matrix(current_camera)
    cosine=np.clip((np.trace(aT[:3,:3].T@bT[:3,:3])-1)/2,-1,1)
    if np.degrees(np.arccos(cosine))>12:
        raise ValueError('物体已认出，但观察方向变化较大；请从接近初始观察方向预览位置纠偏')
    oa=_outline(reference);ob=_outline(current)
    ob,partial=_partial_outline(reference,current,oa,ob,cfg)
    if not partial and any(x.any() for x in _edge_flags(oa)):
        raise ValueError('参考物体已超出画面，请补录完整参考视角')
    if oriented:
        if partial:raise ValueError('轻微裁边使用原始投影方向，不从残缺轮廓推测旋转')
        oa=_oriented_outline(reference);ob=_oriented_outline(current)
    sa,sb=oa[3],ob[3];iou=float(np.sum(sa&sb)/max(1,np.sum(sa|sb)))
    aspect=(oa[2]-oa[1])[0]/(oa[2]-oa[1])[1]/((ob[2]-ob[1])[0]/(ob[2]-ob[1])[1])
    # 二维形状仅作诊断评分；是否可纠偏由分布式深度对应和三维重叠决定。
    grid=np.array([(x,y) for y in np.linspace(.04,.96,40) for x in np.linspace(.08,.92,24)])
    ap,_,av=_samples(reference,aT,cfg,grid,oa)
    bp,pixels,bv=_samples(current,bT,cfg,grid,ob)
    valid=av&bv
    diff=(bp-ap)[valid];uv=grid[valid];pixels=pixels[valid]
    if len(diff)<40:raise ValueError('物体已认出，但有效深度覆盖不足；请调整相机视角再预览')
    # 共识平移剔除透明表面透出的背景深度；每个格子一票，不重复计数。
    best=None
    for shift in diff[::max(1,len(diff)//120)]:
        good=np.linalg.norm(diff-shift,axis=1)<.012
        if best is None or good.sum()>best.sum():best=good
    shift=np.median(diff[best],axis=0)
    good=np.linalg.norm(diff-shift,axis=1)<.012
    if good.sum()<40 or good.mean()<.60:
        raise ValueError('物体已认出，但深度对应不一致；请让物体完整可见后预览')
    shift=np.median(diff[good],axis=0)
    rms=float(np.sqrt(np.mean(np.sum((diff[good]-shift)**2,axis=1))))
    spread=np.ptp(uv[good],axis=0)
    # 上下左右都必须有支持，避免仅匹配到标签或背景的一小片。
    quadrants={(int(x>=.5),int(y>=.5)) for x,y in uv[good]}
    if rms>.008 or min(spread)<.55 or len(quadrants)<4:
        raise ValueError('物体已认出，但深度只覆盖局部；请露出物体上下两端再预览')
    delta=np.eye(4);delta[:3,3]=shift
    return delta,dict(method='shape_position',method_label='方向对齐＋深度位置纠偏' if oriented else '整体轮廓＋深度位置纠偏',
        silhouette_alignment='principal_axes' if oriented else 'image_axes',
        orientation_observed=False,angle_resolution_deg=180.,partial_outline=partial,
        position_resolution_mm=max(2.,2*rms*1000),rms_mm=rms*1000,
        inliers=int(good.sum()),matches=len(diff),silhouette_overlap=iou,silhouette_aspect_ratio=float(aspect),
        pixels=pixels[good].tolist())


def position_from_shape(reference,reference_camera,current,current_camera,cfg):
    """保持原估计优先；形状/深度不符时先纠正轮廓方向，再核验同样的深度证据。"""
    try:return _estimate(reference,reference_camera,current,current_camera,cfg)
    except ValueError as original:
        try:return _estimate(reference,reference_camera,current,current_camera,cfg,oriented=True)
        except ValueError:
            raise original
