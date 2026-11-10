"""物体参照与动作绑定。无ROS、无运动接口；原示教数据永不被纠偏覆盖。"""
import hashlib,json,os,re,uuid
from pathlib import Path
from copy import deepcopy
import numpy as np


def matrix(value):
    t=np.asarray(value,float)
    if (t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1])
        or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-6)
        or abs(np.linalg.det(t[:3,:3])-1)>1e-6):raise ValueError('参照坐标变换无效')
    return t


def reference_id(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('物体参照编号无效')
    return value


def binding(value):
    if not isinstance(value,dict):raise ValueError('姿态参照格式无效')
    side=value.get('side');mode=value.get('mode','position')
    if side not in ('left','right') or mode not in ('position','rigid'):raise ValueError('参照手臂或纠偏方式无效')
    return dict(reference=reference_id(value['reference']),side=side,mode=mode,
                taught_object_pose=matrix(value['taught_object_pose']).tolist(),
                relative_tcp=matrix(value['relative_tcp']).tolist())


def corrected_tcp(ref_pose,current_pose,relative,mode):
    old=matrix(ref_pose);new=matrix(current_pose);local=matrix(relative)
    if mode=='rigid':return new@local
    if mode!='position':raise ValueError('未知纠偏方式')
    result=old@local;result[:3,3]+=new[:3,3]-old[:3,3]
    return result


def record_bindings(record):
    raw=record.get('visual_bindings',{})
    if not isinstance(raw,dict):raise ValueError('动作参照映射无效')
    if raw and 'trajectory' in record:raise ValueError('连续录制保持原时序；请在姿态与动作中建立参照姿态串')
    result={}
    for key,value in raw.items():
        index=int(key)
        if str(index)!=str(key) or not 0<=index<len(record['keyframes']):raise ValueError('动作参照帧索引无效')
        result[str(index)]=binding(value)
    return result


class ReferenceStore:
    """内容哈希目录：动作引用快照，不受同名重新拍摄影响。"""
    def __init__(self,directory):self.directory=Path(directory)/'visual_references'
    def save(self,meta,arrays):
        self.directory.mkdir(parents=True,exist_ok=True)
        raw=json.dumps(meta,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()
        tmp=self.directory/('.'+uuid.uuid4().hex+'.npz')
        try:
            with tmp.open('wb') as stream:
                np.savez_compressed(stream,metadata=np.frombuffer(raw,dtype=np.uint8),**arrays)
                stream.flush();os.fsync(stream.fileno())
            ident=hashlib.sha256(tmp.read_bytes()).hexdigest();os.replace(tmp,self.directory/(ident+'.npz'))
        finally:tmp.unlink(missing_ok=True)
        return ident
    def load(self,ident):
        path=self.directory/(reference_id(ident)+'.npz')
        if not path.is_file():raise ValueError('动作关联的物体参照文件缺失，请重新记录参照')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=ident:raise ValueError('物体参照文件已改变，请重新记录')
        import io
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:
            meta=json.loads(z['metadata'].tobytes());arrays={k:z[k].copy() for k in z.files if k!='metadata'}
        matrix(meta['object_pose']);return meta,arrays
    def list(self):
        result=[];self.errors=[]
        removed=set(self._index().get('removed',[]))
        for path in sorted(self.directory.glob('*.npz'),key=lambda p:p.stat().st_mtime,reverse=True):
            try:
                meta,_=self.load(path.stem)
                if path.stem in removed or meta.get('parent_reference'):continue
                result.append(dict(id=path.stem,name=meta['name']))
            except (ValueError,OSError,KeyError,TypeError) as exc:
                # 单个参照损坏不能阻断普通抓取/旧动作；真正调用它仍由load明确报错。
                self.errors.append(path.name+': '+str(exc))
        return result
    def _index(self):
        p=self.directory/'index.json'
        return json.loads(p.read_text('utf-8')) if p.exists() else {'removed':[],'views':{}}
    def _write_index(self,value):
        from .core import atomic_json
        self.directory.mkdir(parents=True,exist_ok=True);atomic_json(self.directory/'index.json',value)
    def remove(self,ident):
        """列表删除不破坏已保存姿态的不可变参照，旧动作仍能解析该快照。"""
        self.load(ident);doc=self._index();doc['removed']=sorted(set(doc.get('removed',[]))|{ident});self._write_index(doc)
    def add_view(self,ident,meta,arrays):
        self.load(ident)
        doc=self._index();views=doc.setdefault('views',{}).setdefault(ident,[])
        if len(views)>=24:raise ValueError('该物体已有24个补录视角；当前编号与已有动作仍可使用，请避免重复拍摄')
        child=self.save(dict(meta,parent_reference=ident),arrays)
        if child not in views:views.append(child)
        self._write_index(doc);return child
    def view_ids(self,ident):
        """显示/识别共同使用这份有效视角列表；初始帧永远排第一。"""
        self.load(ident)
        return [ident]+list(dict.fromkeys(self._index().get('views',{}).get(ident,[])))
    def remove_view(self,ident,child):
        """移出匹配库；不可变快照保留，避免删除历史证据或原动作引用。"""
        self.load(ident);reference_id(child)
        if child==ident:raise ValueError('初始帧是动作的位姿基准，不能单独删除；可从主界面删除整条物体记忆')
        doc=self._index();views=doc.get('views',{}).get(ident,[])
        if child not in views:raise ValueError('该视角不属于当前物体，或已删除，请刷新列表')
        doc['views'][ident]=[v for v in views if v!=child]
        retired=doc.setdefault('removed_views',{}).setdefault(ident,[])
        if child not in retired:retired.append(child)
        self._write_index(doc)

    def views(self,ident):
        """补录仍归原编号；metric_registered标识是否已关联最初几何，旧绑定不改写。"""
        result=[self.load(ident)]
        for child in self._index().get('views',{}).get(ident,[]):
            meta,arrays=self.load(child)
            if meta.get('parent_reference')!=ident:raise ValueError('补录视角与原参照不一致')
            result.append((meta,arrays))
        return result


def make_binding(store,ident,side,mode,q,geometry):
    """用保存的初始物体位姿计算相对TCP；不推测物体此刻在哪，也不改原参照。"""
    from .teaching_motion import checked_joints
    q=checked_joints(q)
    if side not in ('left','right') or mode not in ('position','rigid'):raise ValueError('参照手臂或纠偏方式无效')
    meta,_=store.load(ident)
    if meta.get('parent_reference'):raise ValueError('请选择物体编号，补录图片不是独立位姿基准')
    if meta['model_sha256']!=hashlib.sha256(geometry.path.read_bytes()).hexdigest():raise ValueError('本机模型已变化，请重新记录物体参照')
    tcp=geometry.frames(q)['Link_'+side.title()+'_Wrist_Lower_to_Gripper']@geometry.tool_transform(side)
    return binding(dict(reference=ident,side=side,mode=mode,taught_object_pose=meta['object_pose'],relative_tcp=(np.linalg.inv(matrix(meta['object_pose']))@tcp).tolist()))


def clone_without_bindings(record):
    result=deepcopy(record);result.pop('visual_bindings',None);return result
