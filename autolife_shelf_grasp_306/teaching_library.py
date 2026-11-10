"""One atomic YAML document: compatible actions plus optional editor metadata."""
import hashlib,json,os,uuid
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import yaml
import fcntl
from functools import wraps
from control_center_joint_control.action_library import validate_action_name,normalize_action_record
from .teaching_motion import normalize_draft,compile_action
from .task_flow import full_record


def library_lock(method):
    @wraps(method)
    def locked(self,*args,**kwargs):
        self.directory.mkdir(parents=True,exist_ok=True)
        with (self.directory/'.teaching-library.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            return method(self,*args,**kwargs)
    return locked


def normalized_record(record):
    return full_record(record) if 'trajectory' in record or 'visual_bindings' in record else normalize_action_record(record)


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class TeachingLibrary:
    def __init__(self,directory):
        self.directory=Path(directory);self.path=self.directory/'saved_joint_actions.yaml'
        self.revision=None

    def read(self):
        raw=self.path.read_bytes() if self.path.exists() else b''
        self.revision=hashlib.sha256(raw).hexdigest()
        doc=yaml.safe_load(raw) or {'version':1,'actions':{}}
        if not isinstance(doc,dict) or not isinstance(doc.get('actions',{}),dict):raise ValueError('动作库格式不正确')
        return doc

    def records(self):
        doc=self.read();result={}
        for name,record in doc.get('actions',{}).items():
            try:result[validate_action_name(name)]=normalized_record(record)
            except (ValueError,TypeError,KeyError):continue
        return result

    def load(self,name):
        doc=self.read();record=normalized_record(doc['actions'][name])
        meta=doc.get('teaching_editor',{}).get(name,{})
        if meta.get('record_digest')==digest(record):
            draft=deepcopy(meta['draft']);draft['name']=name;return normalize_draft(draft)
        if record.get('visual_bindings'):raise ValueError('动作参照编辑信息不一致，请恢复原动作库；不会静默丢弃纠偏信息')
        if 'trajectory' in record:raise ValueError('连续轨迹的编辑信息不一致，请恢复录制数据，不能退化为少量姿态执行')
        # Legacy and independently edited actions retain their current poses.
        frames=[dict(label=f'姿态 {i+1}',joints_deg=f['joints_deg'],hold_sec=f['hold_sec'],
                     minimum_sec=min(30.,f['transition_sec']/(record['speed_percent']/100.)),
                     speed_percent=100.,ghost=True) for i,f in enumerate(record['keyframes'])]
        return normalize_draft(dict(name=name,frames=frames))

    @library_lock
    def save(self,draft,expected_revision=None):
        draft=normalize_draft(draft);name=validate_action_name(draft['name']);record=compile_action(draft)
        record['saved_at']=datetime.now().astimezone().isoformat(timespec='seconds')
        doc=self.read()
        if expected_revision is not None and expected_revision!=self.revision:
            raise ValueError('动作库已被其他窗口修改，请重新载入后再保存；当前草稿保留')
        doc.setdefault('actions',{})[name]=record
        doc.setdefault('teaching_editor',{})[name]=dict(record_digest=digest(record),draft=draft,curve='quintic_rest_to_rest_v1')
        self._write(doc);return record

    @library_lock
    def delete(self,name,expected_revision=None):
        doc=self.read()
        if expected_revision is not None and expected_revision!=self.revision:raise ValueError('动作库已更新，请刷新后删除')
        doc['actions'].pop(name,None);doc.get('teaching_editor',{}).pop(name,None);self._write(doc)

    def _write(self,doc):
        self.directory.mkdir(parents=True,exist_ok=True)
        tmp=self.path.with_name(self.path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with tmp.open('w',encoding='utf8') as f:
                yaml.safe_dump(doc,f,allow_unicode=True,sort_keys=False);f.flush();os.fsync(f.fileno())
            os.replace(tmp,self.path)
        finally:
            if tmp.exists():tmp.unlink()
        self.read()
