"""Small RGB-D plane/clear-patch detector; no neural network or motor access."""
import cv2
from .point_cloud_filter import group_cells
import numpy as np


def retryable_surface_error(error,attempt):
    """A bounded fresh look, never reusing a rejected plan or moving hardware."""
    return attempt<2 and any(text in str(error) for text in (
        '桌面有效深度区域不足','实际底面仍没有足够','可支撑区域被物品占用','放置轨迹接近桌面已有物品'))


def retryable_placement_error(diagnostics,attempt):
    """Reobserve clutter-rejected geometry; no motion or guard relaxation."""
    return attempt<2 and any('放置轨迹接近桌面已有物品' in d.get('error','')
                             for d in diagnostics)


def fuse_stationary_depth(depths,scale):
    """Median of five fresh aligned frames; missing/moving pixels stay unknown."""
    if len(depths)!=5 or len({np.asarray(v).shape for v in depths})!=1:
        raise ValueError('需要五帧同尺寸的新鲜桌面深度')
    values=np.stack(depths).astype(np.float32)
    valid=np.isfinite(values)&(values*scale>.25)&(values*scale<1.8)
    keep=valid.sum(axis=0)>=3
    selected=values[:,keep];selected[~valid[:,keep]]=np.nan
    med=np.nanmedian(selected,axis=0)
    spread=(np.nanmax(selected,axis=0)-np.nanmin(selected,axis=0))*scale
    out=np.zeros(values.shape[1:],np.float32);out[keep]=np.where(spread<=.04,med,0.)
    if np.count_nonzero(out)<300:raise ValueError('桌面深度持续变化或遮挡，请停稳并保持桌面不动')
    if np.max(out)>65535:raise ValueError('融合深度超出Z16范围')
    return np.rint(out).astype(np.uint16)


def surface_obstacles(points,coef):
    """Reject isolated near-plane depth speckles, not raised objects.

    A 15 mm cell must have repeated evidence for an 18--40 mm protrusion;
    returns over 40 mm remain obstacles regardless of the support fraction.
    """
    p=np.asarray(points);above=p[:,2]-np.c_[p[:,:2],np.ones(len(p))]@coef
    cells=np.floor(p[:,:2]/.015).astype(int)
    _,ids=group_cells(cells)
    near=np.abs(above)<.04;raised=(above>.018)&(above<.65)
    total=np.bincount(ids,weights=near);hits=np.bincount(ids,weights=raised&near,minlength=len(total))
    repeated=(hits>=3)&(hits>=total*.2)
    return p[raised&((above>=.04)|repeated[ids])]


def surfaces(points,settings,radius,diagnostics=None,support_points=None,support_outline=None,support_offset=(0.,0.)):
    diagnostics=diagnostics if diagnostics is not None else {}
    p=np.asarray(points,float);p=p[np.isfinite(p).all(axis=1)]
    ref=settings['table_height_m'];band=settings['height_search_m']
    semantic=support_points is not None
    if semantic:
        seeds=np.asarray(support_points,float).reshape(-1,3)
        seeds=seeds[np.isfinite(seeds).all(axis=1)]
        # The selected table mask supplies the search region, not a guessed
        # height. Keep broad physical bounds to exclude floor/ceiling planes.
        near=seeds[(seeds[:,2]>.35)&(seeds[:,2]<1.25)]
    else:near=p[np.abs(p[:,2]-ref)<band]
    if len(near)<300:raise ValueError('选中桌面有效深度不足，请调整观察位置' if semantic else '参考高度附近没有足够桌面深度，请调整位置或桌高参数')
    # Bounded RANSAC rejects bottles, people and vertical panels in the height band.
    rng=np.random.default_rng(300)
    # Uniform sampling avoids aliasing the ordered camera scan lines.
    sample=near[rng.choice(len(near),min(len(near),6000),replace=False)]
    best=None;score=0
    sample_design=np.c_[sample[:,:2],np.ones(len(sample))]
    for _ in range(100):
        trip=sample[rng.choice(len(sample),3,replace=False)]
        A=np.column_stack([trip[:,:2],np.ones(3)])
        if abs(np.linalg.det(A))<1e-5:continue
        coef=np.linalg.solve(A,trip[:,2])
        if np.linalg.norm(coef[:2])>np.tan(np.deg2rad(8)):continue
        residual=np.abs(sample[:,2]-sample_design@coef)
        count=int((residual<.012).sum())
        if count>score:best=coef;score=count
    if best is None or score<250:raise ValueError('没有稳定的近水平桌面')
    coef=best
    for _ in range(2):
        keep=np.abs(near[:,2]-np.c_[near[:,:2],np.ones(len(near))]@coef)<.012
        plane=near[keep];coef=np.linalg.lstsq(np.c_[plane[:,:2],np.ones(len(plane))],plane[:,2],rcond=None)[0]
    diagnostics.update(plane_points=len(plane),coef=coef.tolist(),minimum=plane.min(axis=0).tolist(),maximum=plane.max(axis=0).tolist())
    if np.linalg.norm(coef[:2])>np.tan(np.deg2rad(8)):raise ValueError('桌面倾斜过大')
    # Grid covers observed support only. Missing space is not assumed to be table.
    res=.005;origin=plane[:,:2].min(axis=0)-res*2
    size=np.ceil((plane[:,:2].max(axis=0)-origin)/res).astype(int)+3
    if np.any(size>900):raise ValueError('桌面候选范围过大')
    grid=np.zeros((size[1],size[0]),np.uint8)
    xy=np.floor((plane[:,:2]-origin)/res).astype(int);grid[xy[:,1],xy[:,0]]=1
    # Reconnect only small depth holes (20 mm kernel span); never use a hull
    # or extend the plane across the large unseen tabletop interior.
    if not semantic:
        grid=cv2.morphologyEx(grid,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8),
                              borderType=cv2.BORDER_CONSTANT,borderValue=0)
    obstacle=surface_obstacles(p,coef)
    blocked=np.zeros_like(grid)
    ij=np.floor((obstacle[:,:2]-origin)/res).astype(int)
    valid=(ij[:,0]>=0)&(ij[:,0]<size[0])&(ij[:,1]>=0)&(ij[:,1]<size[1]);ij=ij[valid]
    blocked[ij[:,1],ij[:,0]]=1
    # Prefer a broad supported patch, then reduce ONLY its extra edge margin.
    # A finer grid avoids treating a 15 mm cell as exact point geometry.
    # Bottle radius and measured obstacle footprint are never removed.
    preferred=float(settings['edge_margin_m'])
    minimum=float(settings.get('minimum_edge_margin_m',.01))
    if not np.isfinite(minimum) or not .01<=minimum<=preferred:
        raise ValueError('最小桌边余量无效')
    margins=sorted(set([preferred,minimum]+[v for v in (.05,.03,.02,.015) if minimum<=v<=preferred]),reverse=True)
    # Keep all locally safe semantic candidates. A far-away large-margin patch
    # must not hide a near reachable patch meeting the validated minimum.
    # Support clearance is a preference, scored jointly with reachability.
    if semantic:margins=[minimum]
    from scipy.ndimage import distance_transform_edt
    # Metric distances avoid rounding both radii up by a whole grid cell.
    # Half-cell diagonal accounts for occupancy location within the cell.
    cell_uncertainty=res/np.sqrt(2)
    edge_distance=distance_transform_edt(grid)*res-cell_uncertainty
    occupied=(distance_transform_edt(1-blocked)*res-cell_uncertainty<radius+.025) if support_outline is None and blocked.any() else np.zeros_like(grid,bool)
    clearance=res
    for extra in (margins if support_outline is None else []):
        clearance=float(radius+extra)
        safe=(edge_distance>=clearance).astype(np.uint8)
        diagnostics.update(observed_cells=int(grid.sum()),support_candidates=int(safe.sum()),edge_extra_m=extra,
                           grid_resolution_m=res,required_support_radius_m=clearance,
                           max_support_radius_m=float(edge_distance.max()))
        safe[occupied]=0
        if safe.any():break
    if support_outline is not None:
        from .placement_support import support_candidates
        # Occupied and unobserved cells remain unavailable. Fit the measured
        # footprint at possible headings instead of requiring its outer circle.
        free=grid.copy();free[blocked!=0]=0
        safe=support_candidates(free,support_outline,res,support_offset)
        diagnostics.update(support_policy='actual_oriented_footprint',support_candidates=int(safe.sum()))
    diagnostics.update(obstacle_points=len(obstacle),blocked_cells=int(blocked.sum()),final_candidates=int(safe.sum()),
                       semantic_support=semantic,unknown_is_free=False)
    rows,cols=np.where(safe)
    if len(rows)==0:
        if support_outline is not None:raise ValueError('已识别桌面，但物品实际底面仍没有足够的已观测支撑；请补充桌面视野')
        reason=('桌面有效深度区域不足' if diagnostics['support_candidates']==0 else '可支撑区域被物品占用')
        raise ValueError(f"{reason}：需要{clearance*100:.1f}cm支撑半径，"
                         f"当前最大{edge_distance.max()*100:.1f}cm；请调整观察位置或前台航点")
    candidates=np.c_[origin[0]+(cols+.5)*res,origin[1]+(rows+.5)*res,np.ones(len(rows))]
    candidates[:,2]=np.c_[candidates[:,:2],np.ones(len(rows))]@coef
    # Support clearance is a preference, never an excuse to discard nearby
    # reachable patches. Return high-clearance candidates first for tie breaks.
    if semantic:candidates=candidates[np.argsort(-edge_distance[rows,cols],kind='stable')]
    # Bounding prism is conservative: includes holes within the observed table.
    table=dict(coef=coef.tolist(),minimum=plane.min(axis=0).tolist(),
               maximum=plane.max(axis=0).tolist(),points=len(plane),margin_m=clearance)
    if support_outline is not None:
        from .placement_support import pack_support
        table['footprint_support']=pack_support(free,origin,res)
    return candidates,table


def stable_tables(results):
    # Candidate must have independent support in every fresh frame.
    from scipy.spatial import cKDTree
    points=results[-1][0]
    for candidates,table in results[:-1]:
        same_height=np.abs(np.c_[points[:,:2],np.ones(len(points))]@table['coef']-points[:,2])<.010
        points=points[(cKDTree(candidates).query(points)[0]<.025)&same_height]
    if not len(points):raise ValueError('多帧没有一致的空闲放置区域')
    table=dict(results[-1][1]);coef=np.median([r[1]['coef'] for r in results],axis=0)
    points=points.copy();points[:,2]=np.c_[points[:,:2],np.ones(len(points))]@coef
    table['coef']=coef.tolist();table['margin_m']=min(r[1]['margin_m'] for r in results)
    return points,table


def obstacle_boxes(points,table,frames=None,camera_origin=None):
    """Observed obstacles above table; unknown crowded scenes are rejected."""
    # Validate spatial clusters per observation, not on accumulated points.
    # Keep the union of boxes, without bridging gaps between observations.
    # Even an obstacle present only in the newest frame remains protected.
    if frames is not None:
        from scipy.spatial import cKDTree
        latest=np.asarray(frames[-1]);coef=np.asarray(table['coef'])
        latest_height=latest[:,2]-np.c_[latest[:,:2],np.ones(len(latest))]@coef
        clear=latest[np.abs(latest_height)<.012,:2]
        tree=cKDTree(clear) if len(clear) else None
        rays=None
        if camera_origin is not None:
            origin=np.asarray(camera_origin,float)
            if origin.shape!=(3,) or not np.isfinite(origin).all():raise ValueError('相机位置无效')
            vectors=latest-origin;ranges=np.linalg.norm(vectors,axis=1)
            good=np.isfinite(vectors).all(axis=1)&(ranges>.05)
            ranges=ranges[good]
            if len(ranges)>=3:rays=cKDTree(vectors[good]/ranges[:,None])
        boxes=[];columns={}
        for index,frame in enumerate(frames):
            # Discard OLD small surface protrusions only where the NEWEST
            # observation positively measures the tabletop at that position.
            # Unknown/occluded space is never treated as clear; tall objects
            # and all newest-frame obstacles are retained.
            if index<len(frames)-1 and rays is not None:
                # A later, farther measurement on the SAME camera ray is
                # positive free-space evidence, including above-table space.
                # Missing rays and nearer occluders never clear old obstacles.
                h=frame[:,2]-np.c_[frame[:,:2],np.ones(len(frame))]@coef
                raised=np.flatnonzero((h>.018)&(h<.65))
                vectors=frame[raised]-origin;distance=np.linalg.norm(vectors,axis=1)
                angular,indices=rays.query(vectors/np.maximum(distance[:,None],1e-9),k=3)
                clear_ray=np.all(angular<.0025,axis=1)&np.all(ranges[indices]>distance[:,None]+.012,axis=1)
                keep=np.ones(len(frame),bool);keep[raised[clear_ray]]=False;frame=frame[keep]
            elif index<len(frames)-1 and tree is not None:
                h=frame[:,2]-np.c_[frame[:,:2],np.ones(len(frame))]@coef
                contradicted=(h>.018)&(h<.04)&(tree.query(frame[:,:2])[0]<.008)
                frame=frame[~contradicted]
            for box in obstacle_boxes(frame,table):
                lo=np.array(box['minimum']);hi=np.array(box['maximum'])
                if 'cell' in box:
                    key=tuple(box['cell'])
                    if key in columns:
                        old=columns[key];old['minimum']=np.minimum(old['minimum'],lo).tolist();old['maximum']=np.maximum(old['maximum'],hi).tolist()
                    else:columns[key]=dict(box)
                    continue
                if any(np.all(lo>=b['minimum']) and np.all(hi<=b['maximum']) for b in boxes):continue
                boxes=[b for b in boxes if not (np.all(lo<=b['minimum']) and np.all(hi>=b['maximum']))]
                boxes.append(box)
        if columns:
            if len(columns)>800:raise ValueError('桌面障碍范围过大，请调整观察位置')
            return list(columns.values())
        if len(boxes)>60:raise ValueError('桌面环境过于拥挤，请清空放置和机械臂通道')
        return boxes
    p=surface_obstacles(points,table['coef'])
    a,b,c=table['coef'];lo=np.array(table['minimum']);hi=np.array(table['maximum'])
    p=p[np.all(p[:,:2]>lo[:2]-.05,axis=1)&np.all(p[:,:2]<hi[:2]+.05,axis=1)]
    if table.get('support_method')=='observed_semantic_plane':
        # Local occupied columns preserve empty gaps in concave/connected
        # clutter. Never replace the entire connected component by one AABB.
        cells=np.floor(p[:,:2]/.05).astype(int)
        unique,ids=group_cells(cells)
        if len(unique)>800:raise ValueError('桌面障碍范围过大，请调整观察位置')
        boxes=[]
        if len(unique):
            # Group once, instead of scanning every point for every column.
            order=np.argsort(ids,kind='stable');grouped=p[order]
            starts=np.r_[0,np.flatnonzero(np.diff(ids[order]))+1]
            lower=np.minimum.reduceat(grouped,starts,axis=0)-.010
            upper=np.maximum.reduceat(grouped,starts,axis=0)+.010
            ground=np.minimum.reduceat(grouped[:,0]*a+grouped[:,1]*b+c,starts)
            # Only extrude returns to the table inside its observed bounds.
            # Outside returns retain their full measured 3-D bounds and
            # padding, without a pillar down to an extrapolated plane.
            # Border overlap (including padding) remains conservative.
            inside=np.all(upper[:,:2]>=lo[:2],axis=1)&np.all(lower[:,:2]<=hi[:2],axis=1)
            lower[inside,2]=np.minimum(lower[inside,2],ground[inside])
            boxes=[dict(minimum=lo.tolist(),maximum=hi.tolist(),cell=cell.tolist(),padding_m=.010)
                   for cell,lo,hi in zip(unique,lower,upper)]
        return boxes
    if len(p)<15:return []
    origin=p[:,:2].min(axis=0)-.02;xy=np.floor((p[:,:2]-origin)/.02).astype(int);size=xy.max(axis=0)+2
    grid=np.zeros((size[1],size[0]),np.uint8);grid[xy[:,1],xy[:,0]]=1
    _,labels=cv2.connectedComponents(cv2.dilate(grid,np.ones((3,3),np.uint8)))
    ids=labels[xy[:,1],xy[:,0]];boxes=[]
    for idx in np.unique(ids):
        cluster=p[ids==idx]
        if len(cluster)<15:continue
        lower=cluster.min(axis=0)-.010;upper=cluster.max(axis=0)+.010
        lower[2]=min(lower[2],float(np.min(cluster[:,0]*a+cluster[:,1]*b+c)))
        boxes.append({'minimum':lower.tolist(),'maximum':upper.tolist(),'padding_m':.010})
    if len(boxes)>12:raise ValueError('桌面环境过于拥挤，请清空放置和机械臂通道')
    return boxes
