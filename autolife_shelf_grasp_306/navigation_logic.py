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


def navigation_summary(data,inflight=False):
    if data.get('map_fault'):return '地图切换未完成，请重新加载地图'
    if data.get('startup_error'):return data['startup_error']
    if data.get('localization_pending'):
        return '等待本轮定位确认；可在放大地图中查看或手动重定位'
    nav=data.get('navigation',{})
    if inflight or nav.get('state') in ('failed','rejected'):
        text=nav.get('message','')
        if text:
            if '占用' in text or '目标可达' in text:text+='；请放大地图核对定位与桌柜距离，未更改原航点'
            return text
    if data.get('ready') and data.get('pose'):return '导航已连接 · 定位正常'
    if data.get('localization_ready'):
        state=data.get('localization',{}).get('state')
        return {'matching':'正在搜索全地图位置…','failed':'全局定位未成功，请重试或手动定位',
                'waiting_scans':'等待双激光数据…','waiting':'等待地图和激光数据…',
                'canceled':'自动搜索已停止，请手动定位'}.get(state,data.get('pose_reason') or '等待定位确认')
    return '导航后台启动中…' if data.get('starting') else '点击“启动导航”，无需另开控制中心'


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
        document=yaml.safe_load(p.read_text()) or {}
        declared=Path(document['map_yaml'])
        declared=declared if declared.is_absolute() else p.parent/declared
        current=Path(data['active_map'])
        current=current if current.is_absolute() else NAV/current
        if declared.resolve()!=current.resolve():return {}
        result={}
        for name, w in document.get('waypoints',{}).items():
            if names is not None and name not in names:continue
            if w.get('frame_id','map')!='map':continue
            values=[float(w[k]) for k in ('x','y','yaw')]
            if all(math.isfinite(v) for v in values):result[name]=values
        return result
    except (ValueError,KeyError,TypeError,OSError):return {}


def waypoint_names(snapshot):
    """Editing is allowed offline; starting still requires the live service."""
    if snapshot.get('map_fault'):return []
    names=snapshot.get('names') or snapshot.get('library') or waypoint_poses()
    return list(dict.fromkeys(n for n in names if isinstance(n,str) and n.strip()))


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
        self.requested_pose=None;self.error=''

    def request(self,stamp):
        self.pending=True;self.requested=stamp;self.initial_stamp=0.;self.sample=None
        self.requested_pose=None;self.error=''

    def fail(self,reason):
        self.pending=False;self.requested_pose=None;self.sample=None;self.initial_stamp=0.
        self.error=reason

    def initial(self,stamp,pose=None):
        # The global matcher repeats the same initial pose three times. Keep
        # the first publication as the request boundary; the final repeat can
        # otherwise be newer than AMCL's only stationary update forever.
        if self.pending and stamp>=self.requested and not self.initial_stamp:
            self.initial_stamp=stamp
            self.requested_pose=list(pose) if pose is not None else None

    def amcl(self,stamp,pose,cov):self.sample=(stamp,pose,cov)

    def navigation(self,data):
        if self.pending and data.get('state') in SUCCESS and float(data.get('stamp',0))>self.requested:
            self.pending=False  # control-center's checked arrival after this request

    def check(self,pose,now):
        if self.error:return False
        if not self.pending:return True
        if pose is None or self.sample is None or not self.initial_stamp:return False
        stamp,p,cov=self.sample
        values=[stamp,*p,*cov,*pose]
        if not all(math.isfinite(v) for v in values):return False
        # AMCL publishes on motion/filter updates, not a stationary heartbeat.
        # A sample belonging to THIS initialpose remains valid when stationary;
        # the caller separately requires a fresh TF, and we check agreement below.
        if stamp<self.initial_stamp or stamp>now+.2:return False
        if any(v<0 or v>.25 for v in cov):return False
        if math.hypot(p[0]-pose[0],p[1]-pose[1])>.15:return False
        if abs(math.atan2(math.sin(p[2]-pose[2]),math.cos(p[2]-pose[2])))>.15:return False
        self.pending=False;return True


class NavigationPause:
    """Serialize stop acknowledgement and resume; transient settling is not failure."""
    def __init__(self):
        self.confirmed=False;self.future=None;self.since=None;self.retry_at=0.
        self.operation=None;self.started=0.
        self.last_resume_stamp=0.

    def owns_paused_status(self,wanted,stamp):
        return bool(wanted or self.confirmed or self.future is not None or
                    (self.last_resume_stamp and stamp<=self.last_resume_stamp))

    def tick(self,wanted,now,call):
        if self.future is not None:
            if not self.future.done():
                if now-self.started>5:raise ValueError('导航暂停/继续服务响应超时')
                return None
            reply=self.future.result();self.future=None
            if not reply.success:
                transient=(self.operation=='stop_ready' or
                           '导航请求尚在确认' in reply.message or
                           reply.message.startswith('等待原导航动作停止') or
                           reply.message.startswith('等待新鲜底盘反馈'))
                if not transient:raise ValueError(reply.message)
                self.retry_at=now+.3
            elif self.operation=='stop_ready':
                if not wanted:
                    self.operation='resume';self.started=now
                    self.future=call('resume');return None
            else:
                self.confirmed=self.operation=='pause';self.since=None
                return self.confirmed
        if wanted==self.confirmed:
            self.since=None;return None
        if self.since is None:self.since=now
        if now-self.since>20:raise ValueError('导航暂停/继续等待超时，请检查停止状态和里程计')
        if now<self.retry_at:return None
        self.operation='pause' if wanted else 'stop_ready';self.started=now
        self.future=call(self.operation)
