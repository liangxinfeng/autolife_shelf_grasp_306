"""姿态快照独立存储，动作串复用原动作库；只处理数据，不连接硬件。"""
import json,math,os,uuid,fcntl
from pathlib import Path
from copy import deepcopy
from .teaching_motion import checked_joints,normalize_draft,compile_action
from control_center_joint_control.action_library import validate_action_name
from control_center_joint_control.joint_definitions import JOINT_LIMITS,JOINT_TO_GROUP


def checked_targets(values):
    if not isinstance(values,dict) or not values:raise ValueError('请选择需要调节的关节')
    result={}
    for n,v in values.items():
        if n not in JOINT_LIMITS:raise ValueError('未知关节：'+n)
        v=float(v);lo,hi=JOINT_LIMITS[n]
        if not math.isfinite(v) or not lo<=v<=hi:raise ValueError('目标超出本机限位：'+n)
        result[n]=v
    return result


def operator_commands(values,current,speed=.4,profile=''):
    """只编辑指定组；组内未选轴从最新反馈补齐，不使用旧预览。"""
    values=checked_targets(values)
    if not math.isfinite(speed) or not .1<=speed<=1.:raise ValueError('调节速度超出10～100%')
    from control_center_joint_control.joint_definitions import group_joint_names
    commands=[]
    for group in dict.fromkeys(JOINT_TO_GROUP[n] for n in values):
        selected={n:v for n,v in values.items() if JOINT_TO_GROUP[n]==group}
        targets={n:float(current[n]) for n in group_joint_names(group)};targets.update(selected)
        cmd=dict(group=group,targets_deg=targets,speed_scale=speed)
        # 单轴接口只收选中轴；未选轴由底层在接收时用最新反馈补齐。
        if len(selected)==1:cmd.update(single_joint=next(iter(selected)),targets_deg=dict(selected))
        elif profile and group=='leg_waist':cmd['motion_profile']=profile
        commands.append(cmd)
    return commands


def frame(label,values,speed=100.):
    return dict(label=label,joints_deg=checked_joints(values),hold_sec=0.,
                speed_percent=float(speed),minimum_sec=.6,ghost=True)


class PoseStore:
    def __init__(self,directory):self.directory=Path(directory);self.path=self.directory/'manual_poses.json'
    def read(self):
        if not self.path.exists():return {}
        doc=json.loads(self.path.read_text('utf8'))
        if doc.get('version')!=1 or not isinstance(doc.get('poses'),dict):raise ValueError('姿态库格式不正确')
        return {validate_action_name(n):checked_joints(v) for n,v in doc['poses'].items()}
    def _edit(self,name,values=None,overwrite=False,delete=False):
        name=validate_action_name(name);self.directory.mkdir(parents=True,exist_ok=True)
        with (self.directory/'.manual-poses.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);records=self.read()
            if delete:
                if name not in records:raise ValueError('姿态已不存在，请刷新')
                del records[name]
            else:
                if name in records and not overwrite:raise ValueError('姿态名称已存在，请使用新名称或确认覆盖')
                records[name]=checked_joints(values)
            tmp=self.path.with_name(self.path.name+'.'+uuid.uuid4().hex+'.tmp')
            try:
                with tmp.open('w',encoding='utf8') as f:
                    json.dump(dict(version=1,poses=records),f,ensure_ascii=False,allow_nan=False,indent=2);f.flush();os.fsync(f.fileno())
                os.replace(tmp,self.path)
            finally:tmp.unlink(missing_ok=True)
    def save(self,name,values,overwrite=False):self._edit(name,values,overwrite)
    def delete(self,name):self._edit(name,delete=True)


def compose(name,rows,speed):
    # 存姿态快照而非可变名称引用：修改/删除姿态库不会悄悄改变已保存动作。
    result=normalize_draft(dict(name=name,frames=[{**deepcopy(r),'speed_percent':r.get('speed_percent',100.)*speed/100.} for r in rows]))
    compile_action(result)
    return result
