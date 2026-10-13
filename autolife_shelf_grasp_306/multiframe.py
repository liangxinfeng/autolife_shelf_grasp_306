"""Bounded majority fusion: tolerate isolated depth tails, not missing geometry."""
import math
import numpy as np


def consensus_indices(centers, dimensions=None):
    points=np.asarray(centers,float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<5:
        raise ValueError('有效深度帧不足，请重试')
    if not np.isfinite(points).all():raise ValueError('物体中心含无效深度')
    keep=np.linalg.norm(points-np.median(points,axis=0),axis=1)<=.010
    if dimensions is not None:
        sizes=np.asarray(dimensions,float)
        if sizes.shape!=points.shape or not np.isfinite(sizes).all() or np.any(sizes<=0):
            raise ValueError('物体尺寸数据无效')
        median=np.median(sizes,axis=0)
        # Compare each observation to the consensus, not two noisy extremes.
        # Width/depth have already been projected into the SAME physical axes.
        # Extra 1mm covers quantization rather than interpreting it as shape.
        keep&=np.all(np.abs(sizes-median)<=np.maximum(.008,median*.15)+.001,axis=1)
    minimum=max(5,math.ceil(len(points)*.70))
    if keep.sum()<minimum:
        raise ValueError('多帧物体尺寸不稳定或中心波动超过10mm，尚无至少5帧一致多数，请重试')
    center=np.median(points[keep],axis=0)
    keep&=np.linalg.norm(points-center,axis=1)<=.010
    if keep.sum()<minimum:raise ValueError('多帧物体中心仍在变化，请保持物体和机器人不动后重试')
    return np.flatnonzero(keep).tolist()
