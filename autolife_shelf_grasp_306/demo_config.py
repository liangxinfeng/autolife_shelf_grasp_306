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


def wrap(angle):return math.atan2(math.sin(angle),math.cos(angle))
