"""Small, testable adapter policy; no chassis commands or replacement planner."""
import json, math
from pathlib import Path
import yaml

SOURCE=Path('/home/ubuntu/ros2_ws/src/autolife_robot_control_center')
NAV=SOURCE/'control_center_navigation'
ACTIVE=NAV/'config/active_map.yaml'
SUCCESS={'succeeded','short_distance_succeeded'}
FAILED={'failed','rejected','canceled','cancelled','paused','undock_failed'}
IDLE=SUCCESS|FAILED|{'idle','waiting','sequence_succeeded'}


def active_map():
    data=yaml.safe_load(ACTIVE.read_text()) or {}
    path=Path(data['active_map'])
    return str((path if path.is_absolute() else NAV/path).resolve())


def map_catalog():
    found={}
    for directory in (SOURCE/'multi_lidar_slam/maps',NAV/'maps'):
        for p in directory.glob('*.yaml'):
            try:
                d=yaml.safe_load(p.read_text());im=Path(d['image'])
                if (im if im.is_absolute() else p.parent/im).is_file():found[str(p.resolve())]=p.stem
            except (ValueError,KeyError,TypeError,OSError):continue
    return [{'path':k,'name':v} for k,v in sorted(found.items())]


def write_active(path,waypoints):
    from .core import atomic_json
    # JSON is valid YAML, and atomic_json avoids partially written config.
    atomic_json(ACTIVE,dict(version=1,active_map=str(path),waypoints_file=str(waypoints)))


def waypoint_poses(names=None):
    try:
        data=yaml.safe_load(ACTIVE.read_text());p=Path(data['waypoints_file'])
        p=p if p.is_absolute() else NAV/p
        index=json.loads((p.parent/'maps_index.json').read_text())
        result={}
        for entry in index['maps'].values():
            for w in entry.get('prepared_positions',[]):
                if names is None or w['name'] in names:
                    x,y,a=w['position'][0]
                    result[w['name']]=[float(x),float(y),math.radians(float(a))]
        return result
    except (ValueError,KeyError,TypeError,OSError):return {}


def select_product(items,product_id):
    matches=[v for v in items if v.get('product_id')==product_id and v.get('matched')]
    if not matches:raise ValueError('货架上未可靠识别到所选商品，未开始抓取')
    # Several bottles of the same SKU: use the strongest match, not another SKU.
    return max(matches,key=lambda v:float(v.get('score',0)))['id']


class GoalProgress:
    def __init__(self,name,sent_stamp):
        self.name=name;self.sent_stamp=sent_stamp;self.seen=False;self.state='sending'

    def update(self,status):
        if float(status.get('stamp',0))<=self.sent_stamp:return None
        name=status.get('active_waypoint')
        if name and name!=self.name:
            raise ValueError('导航目标被其他操作更改，演示停止')
        if name!=self.name:return None
        state=status.get('state','')
        if state in SUCCESS:
            if not self.seen:return None  # never consume a latched old arrival
            self.state=state;return 'arrived'
        if state in FAILED:
            raise ValueError(status.get('message') or '导航未成功，未继续抓放')
        self.seen=True;self.state=state
        return 'running'


class LocalizationGate:
    """Accept a new localization from either UI, never an old TF alone."""
    def __init__(self):
        self.pending=False;self.requested=0.;self.initial_stamp=0.;self.sample=None

    def request(self,stamp):
        self.pending=True;self.requested=stamp;self.initial_stamp=0.;self.sample=None

    def initial(self,stamp):
        if self.pending and stamp>=self.requested:self.initial_stamp=stamp

    def amcl(self,stamp,pose,cov):self.sample=(stamp,pose,cov)

    def navigation(self,data):
        if self.pending and data.get('state') in SUCCESS and float(data.get('stamp',0))>self.requested:
            self.pending=False  # control-center's checked arrival after this request

    def check(self,pose,now):
        if not self.pending:return True
        if pose is None or self.sample is None or not self.initial_stamp:return False
        stamp,p,cov=self.sample
        values=[stamp,*p,*cov,*pose]
        if not all(math.isfinite(v) for v in values):return False
        if stamp<self.initial_stamp or not -.2<=now-stamp<=2.:return False
        if any(v<0 or v>.25 for v in cov):return False
        if math.hypot(p[0]-pose[0],p[1]-pose[1])>.15:return False
        if abs(math.atan2(math.sin(p[2]-pose[2]),math.cos(p[2]-pose[2])))>.15:return False
        self.pending=False;return True
