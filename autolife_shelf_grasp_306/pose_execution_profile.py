"""已保存姿态与抓取页复位的唯一速度设置和轨迹入口；无硬件操作。"""
import json
from pathlib import Path
from .core import atomic_json
from .teaching_motion import finite,compile_action

def settings_path(root):return Path(root)/'data/fixed_actions/manual_pose_settings.json'

def read_speed(root):
    path=settings_path(root)
    value=json.loads(path.read_text('utf-8'))['speed_percent'] if path.exists() else 40.
    return finite(value,25.,100.,'姿态 / 复位速度')

def save_speed(root,value):
    value=finite(value,25.,100.,'姿态 / 复位速度')
    path=settings_path(root);path.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(path,dict(version=1,speed_percent=value))

def pose_record(target,source,speed=40.,name='姿态'):
    from .pose_workspace_data import frame,compose
    speed=finite(speed,25.,100.,'姿态 / 复位速度')
    return compile_action(compose(name,[frame(name,target)],speed),source)
