"""Measured solid-object extent; gripper aperture is directional, not diagonal."""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .bottle import DepthQualityError,horizontal_basis


def coherent_component(points):
    """Remove detached/sparse depth tails without inventing hidden surfaces."""
    p=np.asarray(points,float)
    if len(p)<80:raise DepthQualityError('物体有效深度点不足')
    distances,indices=cKDTree(p).query(p,k=12)
    dense=distances[:,-1]<.008
    edges=(distances<.008)&dense[:,None]&dense[indices]
    rows=np.broadcast_to(np.arange(len(p))[:,None],indices.shape)[edges];cols=indices[edges]
    graph=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(p),len(p))).tocsr()
    _,labels=connected_components(graph,directed=False);counts=np.bincount(labels)
    order=np.argsort(counts)[::-1];keep=labels==order[0]
    if keep.sum()<max(80,len(p)*.70) or (len(order)>1 and counts[order[1]]>len(p)*.15):
        raise DepthQualityError('物体深度存在多个分离实体或大量噪声，请重新分割目标')
    return keep


def oriented_bounds(points,up):
    u,v,n=horizontal_basis(up);plane=np.column_stack([u,v]);xy=points@plane
    lo,hi=np.percentile(xy,[1,99],axis=0);core=xy[((xy>=lo)&(xy<=hi)).all(axis=1)]
    if len(core)<40:raise DepthQualityError('物体水平轮廓深度不足')
    rect=cv2.minAreaRect(core.astype(np.float32));a=np.deg2rad(rect[2])
    axis=plane@np.array([np.cos(a),np.sin(a)])
    other=np.cross(n,axis)
    if rect[1][0]>rect[1][1]:axis=other
    # Canonical sign; rectangle axes otherwise flip by 180 degrees per frame.
    if axis@u<0:axis=-axis
    basis=np.column_stack([axis,np.cross(n,axis),n]);local=points@basis
    low,high=np.percentile(local,[1,99],axis=0)
    return local,low,high,basis


def footprint_vertices(points,up,center):
    u,v,n=horizontal_basis(up);plane=np.column_stack([u,v]);xy=points@plane
    low,high=np.percentile(xy,[1,99],axis=0)
    core=xy[((xy>=low)&(xy<=high)).all(axis=1)]
    hull=cv2.convexHull(core.astype(np.float32))[:,0].astype(float)
    return (hull-center@plane)@plane.T


def bbox(product):
    axes=np.asarray(product['bbox_axes_torso'],float);extent=np.asarray(product['bbox_extents_m'],float)
    if (axes.shape!=(3,3) or extent.shape!=(3,) or not np.isfinite(axes).all() or
        not np.isfinite(extent).all() or np.any(extent<=0) or np.any(extent>.45) or
        not np.allclose(axes.T@axes,np.eye(3),atol=1e-5)):
        raise ValueError('物体方向包络数据无效')
    return axes,extent


def projected_width(product,closing_axis):
    axes,extent=bbox(product);direction=np.asarray(closing_axis,float)
    norm=np.linalg.norm(direction)
    if not np.isfinite(norm) or norm<1e-8:raise ValueError('夹爪闭合方向无效')
    return float(np.abs(axes.T@(direction/norm))@extent)


def directional_yaws(product,cfg):
    """Prefer forward; cheaply skip jaws that cannot fit, before running IK."""
    axes,_=bbox(product);base=float(cfg['approach_yaw_deg'])
    # The long axis is the approach axis when jaws close across the short side.
    up=np.asarray(cfg['up_torso'],float);long_axis=axes[:,1]
    # Invert horizontal_basis's projection, including torso -Z-up frames.
    raw=long_axis-up*(long_axis[2]/up[2]) if abs(up[2])>1e-6 else long_axis
    ideal=np.rad2deg(np.arctan2(raw[1],raw[0]))
    ideal=(ideal-base+90)%180-90
    offsets=sorted(set([0.,float(ideal),*map(float,range(-90,91,5))]),key=lambda v:(abs(v),v))
    for offset in offsets:
        _,side,_=horizontal_basis(cfg['up_torso'],np.deg2rad(base+offset))
        if projected_width(product,side)+.010<=cfg['gripper_max_width_m']:yield offset
