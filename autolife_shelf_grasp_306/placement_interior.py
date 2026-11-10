"""放置落点的桌内偏好；只排序，不代替实际支撑、可达性和碰撞检查。"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from .placement_support import unpack_support


class InteriorPreference:
    def __init__(self, table, radius, preferred=.08):
        self.radius=float(radius)
        self.preferred=max(0.,float(preferred))
        self.low=np.asarray(table['minimum'],float)[:2]
        self.high=np.asarray(table['maximum'],float)[:2]
        self.distance=None
        if 'footprint_support' in table:
            grid,self.origin,self.resolution=unpack_support(table['footprint_support'])
            # 外围也按未观测区域处理；孔洞和桌边都不能凭矩形外框填补。
            self.distance=distance_transform_edt(np.pad(grid.astype(bool),1))*self.resolution
            self.distance=self.distance[1:-1,1:-1]

    def margins(self, positions, ground):
        """返回物品外缘到已确认桌面边界的保守余量（米），支持腰部倾斜坐标。"""
        points=np.asarray(positions,float).reshape(-1,3)@ground[:3,:3].T+ground[:3,3]
        distance=np.min(np.minimum(points[:,:2]-self.low,self.high-points[:,:2]),axis=1)
        if self.distance is not None:
            indices=np.floor((points[:,:2]-self.origin)/self.resolution).astype(int)
            valid=(indices[:,0]>=0)&(indices[:,1]>=0)&(indices[:,0]<self.distance.shape[1])&(indices[:,1]<self.distance.shape[0])
            measured=np.zeros(len(points))
            measured[valid]=self.distance[indices[valid,1],indices[valid,0]]
            # 保守扣除一个栅格对角线，避免把单元中心距离当作精确物理边界。
            distance=np.minimum(distance,measured-np.sqrt(2)*self.resolution)
        return distance-self.radius

    def shortfalls(self, margins):
        # 达到舒适余量后不继续追求桌心，避免无谓伸远和增加规划耗时。
        return np.maximum(0.,self.preferred-np.asarray(margins,float))


def interior_rank(choice):
    """优先缩小距边不足，毫米级差异不压过运动代价；缺数据保持原排序。"""
    return (round(float(choice.get('edge_shortfall_m',0.))/.005),choice['score'])
