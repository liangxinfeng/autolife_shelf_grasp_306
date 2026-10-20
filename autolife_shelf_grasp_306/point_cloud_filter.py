"""Exact body exclusion with a conservative bounding-box broad phase."""
import numpy as np


def outside_boxes(world,placements,boxes,padding=.015):
    world=np.asarray(world)
    if len(placements)!=len(boxes):raise ValueError('机器人几何包络数量不一致')
    keep=np.isfinite(world).all(axis=1)
    for transform,(low,high) in zip(placements,boxes):
        low=np.asarray(low);high=np.asarray(high)
        rotation=transform.rotation;translation=transform.translation
        center=rotation@((low+high)*.5)+translation
        extent=np.abs(rotation)@((high-low)*.5+padding)+1e-12
        lower=center-extent;upper=center+extent
        nearby=keep.copy()
        for axis in range(3):
            nearby&=(world[:,axis]>=lower[axis])&(world[:,axis]<=upper[axis])
        indices=np.flatnonzero(nearby)
        if not len(indices):continue
        local=(world[indices]-translation)@rotation
        delta=local-np.clip(local,low,high)
        # The exact old Euclidean test remains; broad phase rejects no nearby
        # points and never substitutes the enclosing AABB for the mesh box.
        keep[indices]=np.linalg.norm(delta,axis=1)>padding
    return keep


def group_cells(cells):
    """Same lexicographic groups as np.unique(axis=0), with scalar keys."""
    cells=np.asarray(cells,dtype=np.int64)
    if not len(cells):return np.empty((0,2),np.int64),np.empty(0,np.int64)
    low=cells.min(axis=0);high=cells.max(axis=0)
    width=int(high[0])-int(low[0])+1;height=int(high[1])-int(low[1])+1
    if width*height>np.iinfo(np.int64).max:
        return np.unique(cells,axis=0,return_inverse=True)
    keys=(cells[:,0]-low[0])*height+(cells[:,1]-low[1])
    unique,ids=np.unique(keys,return_inverse=True)
    return np.column_stack([unique//height+low[0],unique%height+low[1]]),ids
