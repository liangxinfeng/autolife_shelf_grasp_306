"""Persistent demonstration parameters, snapshotted once per run."""
import math
from .core import atomic_json

DEFAULTS=dict(table_height_m=.74,height_search_m=.10,edge_margin_m=.08,minimum_edge_margin_m=.01,place_clearance_m=.004,
              approach_height_m=.10,release_retreat_m=.12,place_speed_m_s=.03)
LIMITS=dict(table_height_m=(.4,1.1),height_search_m=(.03,.15),
            edge_margin_m=(.06,.20),minimum_edge_margin_m=(.01,.06),place_clearance_m=(.002,.008),
            approach_height_m=(.07,.18),release_retreat_m=(.10,.20),place_speed_m_s=(.01,.04))


def validate(values):
    cfg={**DEFAULTS,**values}
    for key,(lo,hi) in LIMITS.items():
        value=float(cfg[key])
        if not math.isfinite(value) or not lo<=value<=hi:raise ValueError('演示参数超出范围：'+key)
        cfg[key]=value
    return {k:cfg[k] for k in DEFAULTS}


def load(root):
    import json
    path=root/'data/demo_settings.json'
    return validate(json.loads(path.read_text()) if path.exists() else {})


def save(root,values):atomic_json(root/'data/demo_settings.json',validate(values))


def require_drink(product):
    if product.get('geometry','cylinder')!='cylinder' or not .08<=float(product['height_m'])<=.4:
        raise ValueError('演示放置仅支持已录入尺寸的饮料瓶/罐（高度8～40cm）')


def require_placeable(product):
    shape=product.get('geometry','cylinder')
    w=float(product['diameter_m']);h=float(product['height_m']);d=float(product.get('depth_m',w))
    directional='bbox_axes_torso' in product
    if directional:
        from .object_extent import bbox
        bbox(product)
    if not all(math.isfinite(v) for v in (w,h,d)) or not (.015<=w<=(.35 if directional else .09) and .025<=h<=.4):
        raise ValueError('物品几何尺寸不足以安全规划放置')
    if shape not in ('solid','cylinder') or (shape=='solid' and not .005<=d<=(.35 if directional else .15)):
        raise ValueError('当前自动放置支持瓶罐及尺寸可靠的硬质物品，暂不支持柔性袋体')
    if product.get('temporary') and (product.get('dimension_source')!='multi_frame_rgbd' or product.get('dimension_frames',0)<5):
        raise ValueError('未录入物品缺少本轮多帧尺寸记录，不能自动放置')


def footprint_radius(product):
    if 'bbox_axes_torso' in product:
        from .object_extent import bbox
        return float(math.hypot(*bbox(product)[1][:2])/2)
    width=float(product['diameter_m'])
    return (math.hypot(width,float(product['depth_m']))/2
            if product.get('geometry')=='solid' else width/2)


def carried_height(product):
    if 'bbox_axes_torso' in product:
        from .object_extent import bbox
        return max(float(product['height_m']),float(bbox(product)[1][2]))
    return float(product['height_m'])


def wrap(angle):return math.atan2(math.sin(angle),math.cos(angle))
