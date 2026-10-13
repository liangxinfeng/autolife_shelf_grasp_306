"""Observed support under the actual footprint, not its enclosing circle."""
import base64
import itertools
import cv2
import numpy as np


def footprint_outline(product,rotation):
    R=np.asarray(rotation,float)
    if 'bbox_axes_torso' in product:
        from .object_extent import bbox
        axes,extent=bbox(product)
        corners=np.array(list(itertools.product((-1,1),repeat=3)))*extent/2
        xy=(corners@(R@axes).T)[:,:2]
    else:
        from .demo_config import footprint_radius
        # Old products without measured axes have no reliable heading; retain
        # their complete conservative footprint rather than inventing one.
        a=np.linspace(0,2*np.pi,48,endpoint=False)
        xy=np.c_[np.cos(a),np.sin(a)]*footprint_radius(product)
    return cv2.convexHull(np.asarray(xy,np.float32)).reshape(-1,2)


def footprint_kernel(outline,res):
    # Cell half-diagonal plus one grid cell of clearance. No large circular
    # edge-margin or direction-independent calibration padding.
    poly=np.asarray(outline,float)
    if poly.ndim!=2 or poly.shape[1]!=2 or len(poly)<3 or not np.isfinite(poly).all():
        raise ValueError('物品底面轮廓无效')
    half=int(np.ceil(np.max(np.abs(poly))/res))+3
    if half>100:raise ValueError('物品底面超出可验证范围')
    kernel=np.zeros((half*2+1,half*2+1),np.uint8)
    cv2.fillConvexPoly(kernel,np.rint(poly/res+half).astype(np.int32),1)
    return cv2.dilate(kernel,np.ones((3,3),np.uint8))


def support_candidates(grid,outline,res,offset=(0.,0.)):
    safe=np.zeros_like(grid)
    for yaw in range(-40,41,5):
        a=np.deg2rad(yaw);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        kernel=footprint_kernel(np.asarray(outline)@R.T+offset,res)
        safe|=cv2.erode(grid,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
    return safe


def pack_support(grid,origin,res):
    return dict(shape=list(grid.shape),origin=np.asarray(origin).tolist(),resolution_m=float(res),
                observed=base64.b64encode(np.packbits(grid.ravel())).decode('ascii'))


def unpack_support(data):
    shape=tuple(data['shape']);res=float(data['resolution_m']);origin=np.asarray(data['origin'],float)
    if len(shape)!=2 or not all(isinstance(v,int) and 0<v<=900 for v in shape) or res!=.005 or origin.shape!=(2,) or not np.isfinite(origin).all():
        raise ValueError('桌面支撑网格无效')
    bits=np.unpackbits(np.frombuffer(base64.b64decode(data['observed'],validate=True),dtype=np.uint8))
    if len(bits)<np.prod(shape):raise ValueError('桌面支撑网格不完整')
    return bits[:np.prod(shape)].reshape(shape),origin,res


def supported(grid,origin,res,center,outline):
    center=np.asarray(center,float)
    cell=np.floor((center[:2]-origin)/res).astype(int)
    # Account for the exact centre within its cell, not a rounded TCP.
    shift=center[:2]-(origin+(cell+.5)*res)
    kernel=footprint_kernel(np.asarray(outline)+shift,res)
    rows,cols=np.where(kernel);rows=rows-kernel.shape[0]//2+cell[1];cols=cols-kernel.shape[1]//2+cell[0]
    if np.any(rows<0) or np.any(cols<0) or np.any(rows>=grid.shape[0]) or np.any(cols>=grid.shape[1]):return False
    return bool(np.all(grid[rows,cols]))
