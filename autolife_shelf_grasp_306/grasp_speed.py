"""抓取与原厂腰臂放置共用速度：不改变复位、示教或夹爪速度。"""
import math
import json


def checked_percent(value):
    value=float(value)
    if not math.isfinite(value) or not 20 <= value <= 100:
        raise ValueError('抓取运动速度需在20%～100%之间')
    return value


def read_percent(root):
    path=root/'data/grasp_speed.json'
    if not path.exists():return 100.
    return checked_percent(json.loads(path.read_text('utf-8'))['percent'])
