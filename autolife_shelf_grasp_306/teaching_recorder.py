"""Passive recording state machine. All commands below only capture or save data."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import threading
import time
import uuid
from .teaching_motion import checked_joints
from .teaching_track import NAMES, make_draft


class Recorder:
    def __init__(self, library, feedback, clock=time.monotonic):
        self.library=library;self.feedback=feedback;self.clock=clock;self.lock=threading.RLock()
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='teaching-save')
        self.enabled=False;self.mode='poses';self.phase='idle';self.session=uuid.uuid4().hex
        self.service_id=uuid.uuid4().hex
        self.name='';self.draft=None;self.message='请在动作编排器中开启手柄示教'
        self.revision=0;self.saved_name='';self.error='';self.samples=[];self.markers=[]
        self.heartbeat=0.;self.vr_ready=False;self.vr_continuity=False;self.resetting=False
        self.events=[];self.reset_request=None;self.journal=None;self.journal_path=None
        self.started=0.;self.deadline=0.;self.last_sample=0.;self.last_sync=0.;self.cache=OrderedDict()
        self.pending_dir=library.directory/'recordings_incomplete';self.pending_dir.mkdir(parents=True,exist_ok=True)

    def status(self):
        with self.lock:
            return dict(success=True,available=True,enabled=self.enabled,mode=self.mode,phase=self.phase,
                session=self.session,service_id=self.service_id,name=self.name,count=len(self.draft['frames']) if self.draft else 0,
                samples=len(self.samples),elapsed=round(self.samples[-1][0],2) if self.samples else 0.,
                countdown=max(0.,round(self.deadline-self.clock(),1)) if self.phase=='countdown' else 0.,
                revision=self.revision,saved_name=self.saved_name,message=self.message,error=self.error,resetting=self.resetting,
                feedback_fresh=self.feedback.fresh(),vr_ready=self.vr_ready and self.clock()-self.heartbeat<1.2,
                recoverable=[p.name for p in sorted(self.pending_dir.glob('*.jsonl'))][:20])

    def _new(self):
        self.name=('轨迹' if self.mode=='trajectory' else '示教')+'_'+datetime.now().strftime('%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4]
        self.draft=dict(name=self.name,frames=[]);self.samples=[];self.markers=[];self.events=[];self.reset_request=None;self.saved_name=''
        self.phase='ready';self.error='';self.message='新动作已准备好';self.revision+=1

    def command(self, body, *, desktop=False):
        with self.lock:
            op=body.get('op');now=self.clock()
            if op=='heartbeat':
                if body.get('session')==self.session:
                    self.heartbeat=now;self.vr_ready=bool(body.get('ready'))
                    self.vr_continuity=bool(body.get('recording_ready',self.vr_ready))
                    resetting=bool(body.get('resetting'))
                    if self.phase=='recording' and self.samples:
                        if body.get('reset_held') and self.reset_request is None:
                            age=max(0.,min(3.,float(body.get('reset_hold_age',0.))))
                            target=self.samples[-1][0]-age
                            self.reset_request=next((i for i,s in enumerate(self.samples) if s[0]>=target),len(self.samples)-1)
                        if resetting and not self.resetting:
                            self.events.append(dict(kind='reset',pressed=self.reset_request if self.reset_request is not None else len(self.samples)-1,
                                                    start=len(self.samples)-1,end=None))
                            self.message='正在录制：A+X复位轨迹已自动接入';self.revision+=1
                        if self.resetting and not resetting and self.events and self.events[-1]['end'] is None:
                            self.events[-1]['end']=len(self.samples)-1
                            self.message='复位轨迹已接入，录制继续';self.revision+=1
                            if self.journal:self.journal.write(json.dumps(dict(events=self.events))+'\n')
                        if not body.get('reset_held') and not resetting:self.reset_request=None
                    self.resetting=resetting
                return self.status()
            request_id=str(body.get('request_id',''))
            if not request_id or len(request_id)>80:raise ValueError('缺少请求编号')
            fingerprint=json.dumps(body,sort_keys=True)
            if request_id in self.cache:
                previous,response=self.cache[request_id]
                if previous!=fingerprint:raise ValueError('请求编号已被使用')
                return deepcopy(response)
            if op=='enable':
                if not desktop:raise ValueError('请先在机器人界面开启手柄示教')
                if self.phase in ('recording','countdown','processing'):raise ValueError('请先结束当前录制')
                mode=body.get('mode','poses')
                if mode not in ('poses','trajectory'):raise ValueError('示教模式无效')
                self.mode=mode;self.enabled=True;self.session=uuid.uuid4().hex;self._new()
            else:
                if body.get('session')!=self.session:raise ValueError('示教会话已更换，请等待状态刷新后重试')
                if op not in ('recover','delete_partial') and not self.enabled:raise ValueError('手柄示教尚未开启')
                if op=='finish':
                    if self.phase=='recording':self._stop()
                    elif self.phase=='countdown':self.phase='ready'
                    self.enabled=False;self.message='正在保存，完成后可编辑与预演' if self.phase=='processing' else '示教已结束，可在界面编辑和预演';self.revision+=1
                elif op in ('new','mode'):
                    if self.phase in ('recording','countdown','processing'):raise ValueError('请先结束当前录制')
                    if op=='mode':
                        if body.get('mode') not in ('poses','trajectory'):raise ValueError('示教模式无效')
                        self.mode=body['mode']
                    self._new()
                elif op=='capture':
                    if self.mode=='trajectory':
                        if self.phase!='recording':raise ValueError('连续模式请长按右摇杆开始录制')
                        if len(self.markers)>=60:raise ValueError('本次标记已达60个')
                        if self.samples:self.markers.append(len(self.samples)-1)
                        self.message=f'已添加轨迹标记 {len(self.markers)}';self.revision+=1
                    else:self._capture()
                elif op in ('toggle_record','start','stop'):
                    if self.mode!='trajectory':raise ValueError('逐点模式请双击右摇杆记录姿态')
                    if self.phase=='recording' and op!='start':self._stop()
                    elif self.phase=='countdown' and op!='start':
                        self.phase='ready';self.message='已取消开始录制';self.revision+=1
                    elif op!='stop':self._countdown()
                elif op in ('recover','delete_partial'):
                    if not desktop:raise ValueError('请在机器人界面处理未完成录制')
                    self._recover(body.get('file',''),op=='delete_partial')
                else:raise ValueError('不支持的示教操作')
            response=self.status();self.cache[request_id]=(fingerprint,deepcopy(response))
            while len(self.cache)>256:self.cache.popitem(last=False)
            return response

    def _current(self):
        if not self.feedback.fresh():raise ValueError('关节反馈不新鲜，未记录')
        return checked_joints(self.feedback.values)

    def _capture(self):
        if self.phase in ('processing','recording'):raise ValueError('当前正在处理轨迹')
        values=self._current()
        if self.draft is None or self.phase=='saved':self._new()
        if len(self.draft['frames'])>=100:raise ValueError('当前动作已有100个姿态，请新建动作')
        previous=deepcopy(self.draft)
        self.draft['frames'].append(dict(label=f'姿态 {len(self.draft["frames"])+1}',joints_deg=values,
            hold_sec=0.,speed_percent=100.,minimum_sec=.6,ghost=True))
        try:self.library.save(self.draft)
        except Exception:self.draft=previous;raise
        self.saved_name=self.name;self.message=f'已保存姿态 {len(self.draft["frames"])}';self.phase='ready';self.revision+=1

    def _countdown(self):
        if self.phase=='processing':raise ValueError('轨迹正在整理，请稍候')
        if not self.vr_ready or self.clock()-self.heartbeat>1.2:raise ValueError('请进入VR并正常使能后开始录制')
        self._current()
        if self.phase in ('saved','interrupted') or self.samples:self._new()
        self.phase='countdown';self.deadline=self.clock()+2.;self.message='2秒后开始；再次长按可取消';self.revision+=1

    def _open(self):
        self.started=self.clock();self.samples=[];self.markers=[];self.events=[];self.reset_request=None;self.last_sample=0.
        self.journal_path=self.pending_dir/(uuid.uuid4().hex+'.jsonl')
        self.journal=self.journal_path.open('x',encoding='utf8')
        self.journal.write(json.dumps(dict(version=1,name=self.name,joints=NAMES),ensure_ascii=False)+'\n')
        self.journal.flush();os.fsync(self.journal.fileno());self.last_sync=self.started
        self.phase='recording';self.message='正在录制；长按右摇杆结束，双击添加标记';self.revision+=1

    def tick(self):
        with self.lock:
            now=self.clock()
            if self.phase in ('countdown','recording'):
                continuity=self.vr_continuity if self.phase=='recording' else self.vr_ready
                if not self.feedback.fresh() or now-self.heartbeat>1.2 or not continuity:
                    self.interrupt('关节反馈或VR使能状态中断；已保留未完成录制');return
            if self.phase=='countdown' and now>=self.deadline:self._open()
            if self.phase!='recording':return
            if self.feedback.stamp<=self.last_sample:return
            try:values=self._current()
            except ValueError as exc:self.interrupt(str(exc));return
            stamp=self.feedback.stamp
            if self.last_sample and stamp-self.last_sample>.25:
                self.interrupt('关节采样出现超过0.25秒的间隔，已保留未完成录制');return
            if not self.samples:self.started=stamp
            sample=(stamp-self.started,[values[n] for n in NAMES])
            self.samples.append(sample);self.last_sample=stamp
            self.journal.write(json.dumps(sample,separators=(',',':'))+'\n')
            if now-self.last_sync>=1.:
                self.journal.flush();os.fsync(self.journal.fileno());self.last_sync=now
            if sample[0]>=299.8 or len(self.samples)>=15000:
                self._stop();self.message='达到5分钟上限，正在保存当前轨迹'

    def _close_journal(self):
        if self.journal:
            self.journal.write(json.dumps(dict(events=self.events,markers=self.markers))+'\n')
            self.journal.flush();os.fsync(self.journal.fileno());self.journal.close();self.journal=None

    def interrupt(self, reason):
        with self.lock:
            if self.phase not in ('countdown','recording'):return
            self._close_journal();self.phase='interrupted';self.message=reason;self.error=reason;self.revision+=1

    def _stop(self):
        self._close_journal()
        if len(self.samples)<3 or self.samples[-1][0]<.2:
            self.phase='interrupted';self.message='录制不足0.2秒，未生成动作';self.revision+=1;return
        self.phase='processing';self.message='正在整理连续轨迹与夹爪交互…';self.revision+=1
        self.pool.submit(self._process,deepcopy(self.samples),list(self.markers),deepcopy(self.events),self.name,self.journal_path)

    def _process(self,samples,markers,events,name,path):
        try:
            draft=make_draft(self.library.directory,name,[s[0] for s in samples],[s[1] for s in samples],markers,events)
            from .teaching_motion import compile_action
            from .teaching_trajectory import plan
            record=compile_action(draft)
            prepared,segments,phases=plan(record,draft['frames'][0]['joints_deg'],self.library.directory)
            self.library.save(draft)
            if path:path.unlink(missing_ok=True)
            with self.lock:
                self.draft=draft;self.saved_name=name;self.phase='saved';self.error=''
                self.message=f'已保存 {name}；请先模型预演。'+draft['notes'];self.revision+=1
        except Exception as exc:
            with self.lock:
                self.phase='interrupted';self.error=str(exc);self.message='未完成轨迹已保留：'+str(exc);self.revision+=1

    def _recover(self,filename,delete=False):
        if self.phase in ('countdown','recording','processing'):raise ValueError('请先结束当前录制')
        path=self.pending_dir/str(filename)
        if path.parent.resolve()!=self.pending_dir.resolve() or not path.name.endswith('.jsonl'):raise ValueError('录制文件无效')
        if delete:path.unlink(missing_ok=True);self.revision+=1;return
        if path.stat().st_size>20*1024*1024:raise ValueError('未完成文件过大')
        lines=path.read_text('utf8').splitlines();header=json.loads(lines[0]);samples=[];events=[];markers=[]
        if header.get('joints')!=NAMES:raise ValueError('未完成录制的关节顺序不匹配')
        for i,line in enumerate(lines[1:]):
            try:sample=json.loads(line)
            except ValueError:
                if i==len(lines)-2:break
                raise ValueError('未完成录制中段损坏')
            if isinstance(sample,list):samples.append(sample)
            elif isinstance(sample,dict):events=sample.get('events',events);markers=sample.get('markers',markers)
        self.mode='trajectory';self._new();self.samples=samples;self.events=events;self.markers=markers;self.journal_path=path;self._stop()

    def close(self):
        self.interrupt('界面已关闭，录制中断；可在下次打开时恢复')
        self.pool.shutdown(wait=True)
