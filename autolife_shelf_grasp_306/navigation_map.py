"""Map coordinate math shared by drawing and manual localization."""
import math


def world_to_pixel(meta,x,y):
    dx=x-meta['x'];dy=y-meta['y'];a=meta['yaw'];r=meta['resolution']
    return ((math.cos(a)*dx+math.sin(a)*dy)/r,
            meta['height']-(-math.sin(a)*dx+math.cos(a)*dy)/r)


def pixel_to_world(meta,x,y):
    dx=x*meta['resolution'];dy=(meta['height']-y)*meta['resolution'];a=meta['yaw']
    return (meta['x']+math.cos(a)*dx-math.sin(a)*dy,
            meta['y']+math.sin(a)*dx+math.cos(a)*dy)


def manual_pose(meta,start,end):
    if not meta or not all(math.isfinite(float(v)) for v in (*start,*end)):
        raise ValueError('地图或定位坐标无效')
    if not 0<=start[0]<meta['width'] or not 0<=start[1]<meta['height']:
        raise ValueError('请在地图范围内选择机器人位置')
    x,y=pixel_to_world(meta,*start);ex,ey=pixel_to_world(meta,*end)
    if math.hypot(ex-x,ey-y)<.05:raise ValueError('请从机器人位置拖出朝向箭头，再松开鼠标')
    return x,y,math.atan2(ey-y,ex-x)
