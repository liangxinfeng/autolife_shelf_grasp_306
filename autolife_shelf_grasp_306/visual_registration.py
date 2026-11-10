"""用同一物体上的RGB特征及其配对深度估计刚体变化；不依赖商品名或类别。

不以单帧点云质心差冒充物体运动：遮挡会改变质心。用对应的实体表面点
做RANSAC，缺少纹理/深度或多解时给出可操作提示，不返回伪零偏差。
"""
import cv2
import numpy as np
from .visual_reference import matrix


def image_features(image,mask=None,augment=False):
    """整块分割对象提取特征，并模拟侧视压缩；所有像素映回真实原图。"""
    grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);h,w=grey.shape
    mask=None if mask is None else cv2.erode((np.asarray(mask)>0).astype(np.uint8)*255,np.ones((3,3),np.uint8))
    detector=cv2.SIFT_create(nfeatures=6000,contrastThreshold=.012)
    transforms=[np.eye(3)]
    if augment:
        # 对整物体而非局部纹理保存多个投影描述。深度始终取原图，不生成假3D点。
        for axis in (0,1):
            for shear in (-.25,.25):
                a=np.eye(3);a[axis,axis]=.78;a[axis,1-axis]=shear
                a[:2,2]=np.array([w,h])/2-a[:2,:2]@np.array([w,h])/2
                transforms.append(a)
    pixels=[];descriptors=[]
    for a in transforms:
        view=cv2.warpPerspective(grey,a,(w,h));m=None if mask is None else cv2.warpPerspective(mask,a,(w,h),flags=cv2.INTER_NEAREST)
        keys,d=detector.detectAndCompute(view,m)
        if d is None:continue
        pts=cv2.perspectiveTransform(np.asarray([k.pt for k in keys],np.float32)[:,None,:],np.linalg.inv(a)).reshape(-1,2)
        good=(pts[:,0]>=1)&(pts[:,0]<w-1)&(pts[:,1]>=1)&(pts[:,1]<h-1)
        pixels.extend(pts[good]);descriptors.extend(d[good])
    if not pixels:raise ValueError('物体表面暂无可用纹理，请调整视角或补录物体另一面')
    return np.asarray(pixels),np.asarray(descriptors,np.float32)


def features(frame,cfg,T,mask=None,augment=False):
    from autolife_handeye_calibration_306.depth_target import aligned_depth
    aligned_depth(frame['image'],frame['depth'],frame['frame_id'],frame['depth_id'],frame['intrinsics'],frame['depth_intrinsics'])
    if np.any(np.abs(frame['D'])>1e-9) and frame['intrinsics']['model_id'] not in (0,4):raise ValueError('参照深度畸变模型不支持')
    pixels,desc=image_features(frame['image'],mask,augment)
    return project_features(frame,cfg,T,pixels,desc)


def project_features(frame,cfg,T,pixels,desc):
    x,y=np.rint(pixels).astype(int).T
    h,w=frame['depth'].shape;x=x.clip(0,w-1);y=y.clip(0,h-1)
    depth=np.asarray(frame['depth'],float)*cfg['depth_scale_m'];z=depth[y,x]
    low,high=cfg['depth_range_m'];good=np.isfinite(z)&(z>=low)&(z<=high)
    # 排除深度边缘，防止背景深度被当成标签点。
    for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
        near=depth[(y+dy).clip(0,h-1),(x+dx).clip(0,w-1)]
        good&=np.isfinite(near)&(np.abs(near-z)<.015)
    pixels=pixels[good];z=z[good];desc=desc[good]
    rays=cv2.undistortPoints(pixels[:,None,:],frame['K'],frame['D']).reshape(-1,2)
    points=np.column_stack((rays*z[:,None],z));t=matrix(T);points=points@t[:3,:3].T+t[:3,3]
    if len(points)<8:raise ValueError('参照有效纹理/深度点不足8个，请改善视角或选择更有纹理的区域')
    return dict(pixels=pixels,points=points,descriptors=desc)


def fit(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if len(a)<3 or np.linalg.svd(a-a.mean(0),compute_uv=False)[1]<.003:raise ValueError('参照匹配点过于集中或共线')
    u,_,vt=np.linalg.svd((a-a.mean(0)).T@(b-b.mean(0)));r=vt.T@u.T
    if np.linalg.det(r)<0:vt[-1]*=-1;r=vt.T@u.T
    t=np.eye(4);t[:3,:3]=r;t[:3,3]=b.mean(0)-r@a.mean(0);return t


def consensus(a,b,threshold=.008):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.shape!=b.shape or len(a)<8 or not np.isfinite([a,b]).all():raise ValueError('物体对应点不足或深度无效')
    rng=np.random.default_rng(345);best=None;count=0
    for _ in range(400):
        ids=rng.choice(len(a),3,replace=False)
        try:t=fit(a[ids],b[ids])
        except ValueError:continue
        error=np.linalg.norm(a@t[:3,:3].T+t[:3,3]-b,axis=1);ok=error<threshold
        if ok.sum()>count:best=ok;count=int(ok.sum())
    if best is None or count<8 or count/len(a)<.6:raise ValueError('物体对应点不一致，请清除遮挡并重新定位参照')
    t=fit(a[best],b[best]);error=np.linalg.norm(a@t[:3,:3].T+t[:3,3]-b,axis=1);ok=error<threshold
    if ok.sum()<8:raise ValueError('参照精配准未通过')
    t=fit(a[ok],b[ok]);rms=float(np.sqrt(np.mean(np.sum((a[ok]@t[:3,:3].T+t[:3,3]-b[ok])**2,axis=1))))
    if rms>.006:raise ValueError('参照深度配准残差过大')
    # 小面积或近共线特征可能有很低残差，却无法稳定确定旋转。
    span=float(np.linalg.svd(a[ok]-a[ok].mean(0),compute_uv=False)[1]/np.sqrt(ok.sum()))
    angle_resolution=float(np.degrees(np.arctan2(2*rms,max(span,1e-9))))
    return t,dict(inliers=int(ok.sum()),matches=len(a),rms_mm=rms*1000,angle_resolution_deg=angle_resolution),ok


def register(reference,current):
    a=reference['descriptors'].astype(np.float32);b=current['descriptors'].astype(np.float32)
    if len(a)<8 or len(b)<8:raise ValueError('该视角已保存，但可用纹理不足以建立几何对应')
    # 查询方向使用当前点→参考库；参考库的仿射投影可能包含同一实体点的多个描述，
    # 计算比值时必须跳过同一3D点的描述，不能把它误认成第二个不同物体候选。
    matcher=cv2.BFMatcher(cv2.NORM_L2);matches=matcher.knnMatch(b,a,k=min(16,len(a)))
    unique_current={}
    for row in matcher.knnMatch(a,b,k=min(16,len(b))):
        if len(row)<2:continue
        first=row[0];point=current['points'][first.trainIdx]
        other=next((n for n in row[1:] if np.linalg.norm(current['points'][n.trainIdx]-point)>.003),None)
        if other is not None and first.distance<.72*other.distance:unique_current[first.queryIdx]=point
    pairs=[];used=set()
    for row in sorted(matches,key=lambda x:x[0].distance if x else float('inf')):
        if len(row)<2:continue
        m=row[0];point=reference['points'][m.trainIdx]
        if m.trainIdx not in unique_current or np.linalg.norm(unique_current[m.trainIdx]-current['points'][m.queryIdx])>.003:continue
        other=next((n for n in row[1:] if np.linalg.norm(reference['points'][n.trainIdx]-point)>.004),None)
        if other is None or m.distance>=.72*other.distance:continue
        # 同一点的多个尺度/投影只能贡献一票，不虚增8点RANSAC门槛。
        cell=tuple(np.rint(point/.003).astype(int))
        current_cell=tuple(np.rint(current['points'][m.queryIdx]/.003).astype(int))
        if ('r',cell) in used or ('c',current_cell) in used:continue
        used.add(('r',cell));used.add(('c',current_cell));pairs.append((m.trainIdx,m.queryIdx))
    if len(pairs)<8:raise ValueError('没有找到唯一且足够的物体纹理对应，请让已记录的表面朝向相机；相同物品避免并排遮挡')
    left,right=np.asarray(pairs).T
    t,stats,ok=consensus(reference['points'][left],current['points'][right])
    stats['pixels']=current['pixels'][right[ok]].tolist()
    return t,stats


def register_views(views,current):
    accepted=[];errors=[]
    for meta,arrays in views:
        try:
            t,stats=register(arrays,current);accepted.append((t,stats,meta,arrays))
        except ValueError as exc:errors.append(str(exc))
    if not accepted:raise ValueError('整物体参照未找到可靠对应；请用“补录所选物体视角”逐步记录转动后的表面。'+(errors[0] if errors else ''))
    accepted.sort(key=lambda r:(-r[1]['inliers'],r[1]['rms_mm']))
    best=accepted[0];anchor=matrix(views[0][0]['object_pose'])
    for other in accepted[1:]:
        if np.linalg.norm((other[0]@anchor)[:3,3]-(best[0]@anchor)[:3,3])>.015:
            raise ValueError('不同视角定位到不同物体，请移除同款物体遮挡后再定位')
    best[1]['view_count']=len(views)
    return best


def validate_surface(reference_cloud,current_cloud,delta):
    """整物体3D表面佐证纹理匹配；只检查可见重叠，不以易受遮挡影响的质心纠偏。"""
    from scipy.spatial import cKDTree
    a=np.asarray(reference_cloud,float).reshape(-1,3);b=np.asarray(current_cloud,float).reshape(-1,3)
    a=a[np.isfinite(a).all(1)];b=b[np.isfinite(b).all(1)]
    if len(a)<30 or len(b)<30:raise ValueError('整物体有效深度不足，请露出更多物体表面')
    a=a[::max(1,len(a)//3000)];b=b[::max(1,len(b)//3000)]
    moved=a@delta[:3,:3].T+delta[:3,3];distance=cKDTree(b).query(moved)[0]
    overlap=float(np.mean(distance<.015))
    if overlap<.30:raise ValueError('纹理与整物体深度不一致，请重新取景确认物体')
    return overlap
