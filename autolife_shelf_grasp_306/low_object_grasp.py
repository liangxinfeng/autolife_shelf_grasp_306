"""低矮物体斜上方抓取。纯几何计算；角度相对重力，不随腰部改变。"""
from functools import wraps
import numpy as np
from scipy.spatial.transform import Rotation
from .bottle import horizontal_basis


def grasp_scope(method):
    """局部抓取桌面只对本次抓取生效，不能污染放置/复位复用的规划器。"""
    @wraps(method)
    def wrapped(self,*args,**kwargs):
        old=getattr(self,'_low_object_grasp_active',False)
        self._low_object_grasp_active=True
        if not old:self._request_gripper_vertices=None
        try:return method(self,*args,**kwargs)
        finally:
            self._low_object_grasp_active=old
            if not old:self._request_gripper_vertices=None
    return wrapped


def enabled(product,cfg):
    # 可见局部瓶身不能冒充完整矮物体；部分观测使用已有保守高度。
    h=max(float(product.get('height_m',0)),float(product.get('collision_height_m',0)))
    return product.get('geometry')!='bag_edge' and 0<h<float(cfg.get('low_object_height_m',.10))


def basis(up,yaw,pitch):
    f,s,u=horizontal_basis(up,yaw);a=np.deg2rad(pitch)
    return np.column_stack((f*np.cos(a)-u*np.sin(a),s,u*np.cos(a)+f*np.sin(a)))


def support(product,center,up):
    result=dict(product);u=np.array(up,dtype=float,copy=True);u/=np.linalg.norm(u)
    if 'low_object_support' not in result:
        # 原始物体底部的局部平面估计，不是“已识别整张桌面”。
        # 在手动补偿之前固定，不能随抓点偏移而移动桌面。
        result['low_object_support']=dict(normal=u.tolist(),offset=float(np.dot(center,u)-float(product['height_m'])/2),
            center=np.asarray(center).tolist(),radius_m=.23,source='object_bottom_estimate')
    return result


def fit_support(points,product,center,up):
    """只拟合物体周边近水平支撑面，排除较高物体；不增加拍照等待。"""
    result=support(product,center,up);surface=result['low_object_support']
    u=np.array(up,dtype=float,copy=True);u/=np.linalg.norm(u);c=np.asarray(center,float)
    p=np.asarray(points,float);p=p[np.isfinite(p).all(axis=1)]
    f,s,_=horizontal_basis(u);basis2=np.column_stack([f,s])
    delta=p-c;xy=delta@basis2;z=p@u
    keep=(np.linalg.norm(xy,axis=1)<.23)&(z>surface['offset']-.05)&(z<surface['offset']+.012)
    p,xy,z=p[keep],xy[keep],z[keep]
    if len(p)<100:return result
    bins=np.floor(z/.004).astype(int);ids,counts=np.unique(bins,return_counts=True)
    peak=(ids[np.argmax(counts)]+.5)*.004
    good=np.abs(z-peak)<.006
    if good.sum()<100:return result
    A=np.c_[xy[good],np.ones(good.sum())];coef=np.linalg.lstsq(A,z[good],rcond=None)[0]
    residual=np.abs(np.c_[xy,np.ones(len(xy))]@coef-z);good=residual<.004
    if good.sum()<100 or np.linalg.norm(coef[:2])>np.tan(np.deg2rad(8)):return result
    if np.min(np.ptp(xy[good],axis=0))<.06:return result
    # z=a*x_local+b*y_local+c，换回固定躯干坐标的平面n·p=d。
    normal=u-basis2@coef[:2];length=np.linalg.norm(normal)
    offset=float(coef[2]-np.dot(basis2@coef[:2],c))/length
    result['low_object_support']={**surface,'normal':(normal/length).tolist(),'offset':offset,
        'source':'local_rgbd_plane','plane_points':int(good.sum()),'fit_residual_mm':float(np.median(residual[good])*1000)}
    return result


def observe_support(frame,object_mask,T,product,center,up,cfg):
    from .bottle import object_points,DepthQualityError
    import cv2
    result=support(product,center,up)
    expanded=cv2.dilate(np.asarray(object_mask,dtype=np.uint8),np.ones((9,9),np.uint8)).astype(bool)
    depth=np.asarray(frame['depth'],float)*cfg['depth_scale_m']
    mask=(~expanded)&np.isfinite(depth)&(depth>cfg['depth_range_m'][0])&(depth<cfg['depth_range_m'][1])
    if np.count_nonzero(mask)<100:return result
    try:points,_,_=object_points(frame,mask,T,cfg)
    except DepthQualityError:return result
    return fit_support(points,product,center,up)


def gripper_vertices(planner):
    # 一次同步规划中模型和开口配置不变；跨请求重新核对几何键。
    active=getattr(planner,'_low_object_grasp_active',False)
    frozen=getattr(planner,'_request_gripper_vertices',None)
    if active and frozen is not None:return frozen
    side=planner.cfg['arm'].title()
    cached=getattr(planner,'_low_object_vertices',None)
    if cached is not None and cached[0]==side and not hasattr(planner,'guard'):return cached[1]
    from .gripper_contact_geometry import gripper_geometry
    result=gripper_geometry(planner)['vertices']
    planner._low_object_vertices=(side,result)
    if active:planner._request_gripper_vertices=result
    return result


def orientation(planner,source,tcp,center,product,yaw,adjust_height=False):
    tool=Rotation.from_quat(planner.cfg['T_wrist_tool_quat_xyzw']).as_matrix()
    _,R0=planner.pose(source,tcp);u=np.asarray(planner.cfg['up_torso'])
    short=enabled(product,planner.cfg);angles=planner.cfg.get('low_object_pitch_candidates_deg',[35.,45.,55.,65.,75.]) if short else [0.]
    if not angles or any(not np.isfinite(a) or not 0<float(a)<80 for a in angles) and short:
        raise ValueError('低物体俯角候选应在0～80度之间')
    vertices=gripper_vertices(planner)-np.asarray(tcp) if short else None
    surface=product.get('low_object_support') if short else None
    if short and surface is None:surface=support(product,center,u)['low_object_support']
    margin=float(planner.cfg.get('low_object_table_clearance_m',.003))
    for pitch in angles:
        B=basis(u,yaw,pitch)
        candidates=[B@tool.T,B@np.diag([1.,-1.,-1.])@tool.T]
        candidates.sort(key=lambda R:Rotation.from_matrix(R0.T@R).magnitude())
        for R in candidates:
            clearance=None if not short else float(np.min((vertices@R.T+center)@np.array(surface['normal']))-surface['offset'])
            shift=0.
            if short and adjust_height:
                needed=max(0.,margin+.003-clearance)/max(.1,float(u@np.asarray(surface['normal'])))
                # 只在已观测物体内部上移夹取中心，保留顶面以下2mm；物体整体模型不缩小。
                if needed<=max(0.,float(product['height_m'])/2-.002):shift=needed
            if not short or clearance+shift*float(u@np.asarray(surface['normal']))>=margin+.003-1e-9:
                return R,B[:,0],float(pitch),None if clearance is None else clearance+shift*float(u@np.asarray(surface['normal'])),shift
    raise ValueError('低矮物体的抓点离桌面过近，倾斜后夹爪仍可能触桌；请向上微调抓点或调整物体位置')


def check_clearance(planner,q,tcp,product):
    if not getattr(planner,'_low_object_grasp_active',False):return
    if not enabled(product,planner.cfg) or 'low_object_support' not in product:return
    surface=product['low_object_support'];up=np.array(surface['normal']);c=np.array(surface['center'])
    # pose()返回本次规划起点的躯干系；原厂腰臂协同时也保持这个固定参考。
    p,R=planner.pose(q,tcp);points=(gripper_vertices(planner)-np.asarray(tcp))@R.T+p
    delta=points-c;flat=delta-(delta@up)[:,None]*up
    # 仅约束已观察物体附近局部桌面，折叠在机器人侧的手臂不按无限平面阻挡。
    # 凸包顶点可能都在局部圆外，但手指面仍跨过桌面区域。
    # 用投影包围矩形判定相交；相交时检查整个夹爪，避免遗漏面内部。
    f,s,_=horizontal_basis(up);xy=flat@np.column_stack((f,s))
    nearest=np.maximum(xy.min(axis=0),np.minimum(0.,xy.max(axis=0)))
    if np.linalg.norm(nearest)<float(surface['radius_m']):
        clearance=float(np.min(points@up)-surface['offset'])
        if clearance<float(planner.cfg.get('low_object_table_clearance_m',.003)):
            from .planning import CollisionFailure
            raise CollisionFailure(f'低物体抓取夹爪接近局部桌面（模型间隙{clearance*1000:.1f}mm）')
