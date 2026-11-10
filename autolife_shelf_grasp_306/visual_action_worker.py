"""在现有只读视觉进程内复用相机/分割与IK；从不使能或发布运动。"""
import hashlib,time,uuid
import numpy as np
import cv2
from .visual_reference import ReferenceStore,matrix,record_bindings,corrected_tcp,clone_without_bindings
from .visual_registration import features,register,register_views,validate_surface
from .pose_geometry import ModelGeometry


class VisualActions:
    def __init__(self,root,run,cfg,robot,snapshot,transform,fk,perception,products):
        self.root=root;self.run=run;self.cfg=cfg;self.robot=robot;self.snapshot=snapshot
        self.transform=transform;self.fk=fk;self.perception=perception;self.products=products
        self.geometry=ModelGeometry(cfg['urdf_path']);self.store=ReferenceStore(root/'data/fixed_actions')
        self.frozen=None;self.pending_view=None;self.solvers={};self.reference_cache={}
    def cancelled(self,cmd):
        if (self.run/('visual_cancel_'+cmd['visual_id'])).exists():raise ValueError('已取消物体参照操作，未执行运动')
    def sample(self):
        frame=self.snapshot(depth=True);q,T=self.transform(frame)
        from .planning import TORSO
        return frame,q,self.geometry.frames(q)[TORSO]@T
    def model_hash(self):return hashlib.sha256(self.geometry.path.read_bytes()).hexdigest()
    def mount(self,q,T):
        return np.linalg.inv(self.geometry.frames(q)['Link_Camera_Head_Forehead'])@T
    def check_mount(self,meta,q,T):
        if not np.allclose(matrix(meta['camera_mount']),self.mount(q,T),atol=1e-5):
            raise ValueError('相机外参来源已改变，请重新记录物体参照，不能套用旧相对关系')
    def stable(self,source,base):
        from .demo_protocol import check_base_anchor
        from control_center_joint_control.joint_definitions import BODY_JOINTS
        check_base_anchor(self.robot.base_sample(),base)
        q=self.robot.sample(stationary=True)
        if max(abs(q[n]-source[n]) for n in BODY_JOINTS)>.5:raise ValueError('拍摄或计算期间机器人姿态改变，请停稳后重新定位参照')
    def observe(self,ident,frame,T,current=None,position_only=False):
        from .visual_object_memory import ObjectAppearanceMemory
        self.last_observed_mask=None
        try:
            # 先认整物体；外观识别不依赖可用SIFT点和深度点恰好重合。
            if hasattr(self.perception,'embed'):
                if not hasattr(self,'appearance_memory'):
                    self.appearance_memory=ObjectAppearanceMemory(self.perception,self.cfg)
                raw=self.store.views(ident)
                found=self.appearance_memory.recognize(self.store,ident,raw,frame)
                if found is not None:
                    item,recognition=found
                    try:return self.locate_mask(raw,raw,frame,T,item,position_only,recognition)
                    except ValueError:
                        # 整物体检索已确定身份，只在该物体框内细化SAM轮廓。
                        h,w=frame['image'].shape[:2]
                        box=np.asarray(item['box'],float)/[w,h,w,h]
                        return self.observe_box(raw,raw,frame,T,box,position_only,recognition)
            # 非通用类别仍保留纹理定位，不用外观低分直接否定旧的可靠参照。
            try:return self.observe_texture(ident,frame,T,current,position_only)
            except ValueError as exc:
                if hasattr(self.perception,'embed'):
                    raise ValueError('当前画面未找到可可靠定位的已记忆物体；请确认目标在画面内、未被遮挡后重新预览，已有记录保留，无需删除重录。') from exc
                raise
        except ValueError as exc:
            self.save_observation_failure(ident,frame,T,str(exc))
            raise

    def observe_texture(self,ident,frame,T,current=None,position_only=False):
        from .point_selection import segment
        from .bottle import object_points
        from .visual_multiview import resolve_views
        raw_views=self.store.views(ident);view_key=('resolved',ident);signature=tuple(self.store.view_ids(ident))
        cached=self.reference_cache.get(view_key)
        if cached is None or cached[0]!=signature:self.reference_cache[view_key]=(signature,resolve_views(raw_views))
        views=self.reference_cache[view_key][1]
        # 旧参照现场升级为整分割区域仿射特征，仅在内存缓存；不可变原始文件不改写。
        banks=[]
        for meta,arrays in views:
            key=(ident,meta.get('created'),meta.get('metric_registered',True),str(meta.get('reference_from_capture','')))
            if key not in self.reference_cache:
                from .visual_registration import image_features,project_features
                try:
                    pixels,desc=image_features(arrays['image'],arrays['mask'],augment=True)
                    enriched=project_features(arrays,self.cfg,matrix(meta['camera_pose']),pixels,desc)
                except ValueError:
                    enriched={k:arrays[k] for k in ('pixels','points','descriptors')}
                self.reference_cache[key]=(meta,{**arrays,**enriched})
            banks.append(self.reference_cache[key])
        current=features(frame,self.cfg,T) if current is None else current
        from .visual_multiview import seed_match
        delta,stats,matched,arrays=seed_match(banks,current)
        from .visual_box_selection import projected_box,segment_box
        box=projected_box(arrays['cloud'],delta,T,frame)
        recognition=dict(recognition_view='初始/关联视角' if matched.get('metric_registered',True) else '补录外观',recognition_inliers=stats['inliers'])
        # 同一RGB-D帧内最多再扩大一次提示框，处理细长物体的局部/截断分割。
        # 不新增参照、不要求用户重新录入，每个候选仍要通过深度与路径验证。
        proposals=[box]
        if position_only:
            x1,y1,x2,y2=map(float,box);mx=(x2-x1)*.15;my=(y2-y1)*.15
            expanded=[max(0.,x1-mx),max(0.,y1-my),min(1.,x2+mx),min(1.,y2+my)]
            if not np.allclose(box,expanded):proposals.append(expanded)
        self.last_observed_mask=None
        for attempt,prompt in enumerate(proposals):
            try:
                result=self.observe_box(raw_views,banks,frame,T,prompt,position_only,recognition)
                result[1]['segmentation_attempts']=attempt+1
                return result
            except ValueError as exc:last_error=exc
        self.save_observation_failure(ident,frame,T,str(last_error))
        raise last_error

    def save_observation_failure(self,ident,frame,T,error):
        """仅覆盖当前运行目录的一份诊断帧，便于离线排查；不写物体库。"""
        if not hasattr(self,'run'):return
        try:
            import os
            data={k:frame[k] for k in ('image','depth','K','D')}
            if self.last_observed_mask is not None:data['mask']=self.last_observed_mask
            target=self.run/'visual_observation_failed.npz';tmp=target.with_suffix('.tmp')
            with tmp.open('wb') as stream:np.savez_compressed(stream,**data,T=T,reference=np.array(ident),error=np.array(error))
            os.replace(tmp,target)
        except (OSError,ValueError,KeyError):pass  # 诊断落盘失败不能改变识别或执行结果。

    def observe_box(self,raw_views,banks,frame,T,box,position_only,recognition):
        from .visual_box_selection import segment_box
        main=segment_box(self.perception,frame,box)
        items=[main]+main.get('_alternatives',[]);accepted=[];errors=[]
        for item in items:
            try:accepted.append(self.locate_mask(raw_views,banks,frame,T,item,position_only,recognition))
            except ValueError as exc:errors.append(exc)
        if not accepted:raise errors[0]
        # 多个分割候选必须指向同一位置；不能碰巧选一个能让动作通过的解。
        anchor=matrix(raw_views[0][0]['object_pose'])
        positions=[(r[0]@anchor)[:3,3] for r in accepted]
        if any(np.linalg.norm(a-b)>.015 for a in positions for b in positions):
            raise ValueError('物体分割候选对应了不同位置，请减少遮挡后重新预览；原物体记录无需重录')
        result=min(accepted,key=lambda r:r[1]['rms_mm'])
        result[1].update(segmentation_candidates=len(items),geometry_candidates=len(accepted))
        self.last_observed_mask=result[2]['mask']
        return result

    def locate_mask(self,raw_views,banks,frame,T,item,position_only,recognition):
        from .bottle import object_points
        self.last_observed_mask=item['mask']
        whole=dict(points=np.empty((0,3)),pixels=np.empty((0,2)),descriptors=np.empty((0,128),np.float32))
        canonical=[v for v in banks if v[0].get('metric_registered',True)]
        cloud,_,_=object_points(frame,item['mask'],T,self.cfg)

        try:
            whole=features(frame,self.cfg,T,item['mask'],augment=True)
            delta,stats,matched,arrays=register_views(canonical,whole)
            if position_only and stats['angle_resolution_deg']>3:
                raise ValueError('纹理过于集中，改用完整轮廓估计位置')
            stats['surface_overlap']=validate_surface(arrays['cloud'],cloud,delta)
            stats.update(method='texture_rigid',method_label='物体纹理＋深度定位',orientation_observed=True)
        except ValueError as exc:
            if not position_only:
                raise ValueError('物体已认出，但当前表面无法可靠确定初始朝向。若动作只需跟随物体位置，请选择“只纠正位置”；补录已参与识别，无需重新录入。') from exc
            from .visual_shape_position import position_from_shape
            # 必须与初始帧比较，不能拿补录时随意摆放的位置替换动作参照。
            first,original=raw_views[0]
            observed={**frame,'mask':item['mask'],'cloud':cloud}
            delta,stats=position_from_shape(original,first['camera_pose'],observed,T,self.cfg)
            stats['surface_overlap']=validate_surface(original['cloud'],cloud,delta)
            if stats['surface_overlap']<.55:
                raise ValueError('物体已认出，但深度形状重叠不足；请露出完整物体后重新预览')
        from .segmentation_display import display_contours
        stats.update(recognition,object_box=item.get('box'),mask_contours=display_contours(item))
        return delta,stats,item,whole,cloud
    def save_view(self,prepared):
        meta=prepared['meta'];arrays=prepared['arrays'];ident=prepared['reference']
        if ident:
            if ident in self.store._index().get('removed',[]):raise ValueError('原物体已删除，请重新选择物体')
            self.store.add_view(ident,meta,arrays)
        else:ident=self.store.save(meta,arrays)
        return dict(reference=ident,name=meta['name'],features=len(arrays['points']),supplement=bool(prepared['reference']),
                    metric_registered=meta.get('metric_registered',True),view_count=len(self.store.views(ident)),item=prepared['item'])
    def handle(self,cmd):
        self.cancelled(cmd);action=cmd['action']
        if action=='visual_commit_view':
            pending=getattr(self,'pending_view',None)
            if not pending or cmd.get('preview_token')!=pending['token'] or time.monotonic()-pending['created']>120:raise ValueError('分割预览已失效，请重新框选')
            self.cancelled(cmd);result=self.save_view(pending);self.pending_view=None;self.frozen=None
            return result
        if action=='visual_freeze':
            self.pending_view=None
            self.robot.refresh_measured();base=self.robot.base_sample();frame,q,T=self.sample()
            token=uuid.uuid4().hex;self.frozen=(token,time.monotonic(),frame,q,T,base)
            path=self.run/'visual_reference_scene.png';cv2.imwrite(str(path),frame['image'])
            return dict(token=token,image=str(path),shape=list(frame['image'].shape))
        if action in ('visual_remember','visual_add_view'):
            if self.frozen is None or cmd['token']!=self.frozen[0] or time.monotonic()-self.frozen[1]>120:raise ValueError('请重新取景后点选物体')
            _,_,frame,q,T,base=self.frozen;self.pending_view=None;self.stable(q,base)
            if 'box' in cmd:
                from .visual_box_selection import segment_box
                item=segment_box(self.perception,frame,cmd['box'])
            else:
                from .point_selection import segment
                item=segment(self.perception,frame,self.products,cmd['x'],cmd['y'],describe=False)
            from .bottle import object_points
            cloud,_,_=object_points(frame,item['mask'],T,self.cfg)
            try:f=features(frame,self.cfg,T,item['mask'],augment=True)
            except ValueError:
                if action!='visual_add_view':raise
                # 无纹理的一面也可补录原始整块RGB-D；不凭空生成配准对应点。
                f=dict(points=np.empty((0,3)),pixels=np.empty((0,2)),descriptors=np.empty((0,128),np.float32))
            object_pose=np.eye(4);object_pose[:3,3]=np.median(cloud,axis=0) if not len(f['points']) else np.median(f['points'],axis=0)
            self.stable(q,base);self.cancelled(cmd)
            meta=dict(version=1,name=str(cmd.get('name','参照物')).strip()[:40] or '参照物',object_pose=object_pose.tolist(),
                      camera_pose=T.tolist(),camera_mount=self.mount(q,T).tolist(),joints=q,base_anchor=base,frame=self.geometry.root,model_sha256=self.model_hash(),created=time.time())
            arrays=dict(**f,image=frame['image'],depth=frame['depth'],mask=item['mask'],cloud=cloud,K=frame['K'],D=frame['D'])
            if action=='visual_add_view':
                ident=cmd['reference'];original,_=self.store.load(ident)
                if original['model_sha256']!=self.model_hash():raise ValueError('本机模型变化，请重新记录')
                self.check_mount(original,q,T)
                from .visual_multiview import prepare_view,resolve_views
                meta,arrays=prepare_view(original,resolve_views(self.store.views(ident)),meta,arrays)
                self.stable(q,base);self.cancelled(cmd)
            else:ident=None
            from .segmentation_display import display_contours
            prepared=dict(meta=meta,arrays=arrays,reference=ident,item=dict(id=0,name=meta['name'],box=item['box'],mask_contours=display_contours(item)))
            if cmd.get('preview_only'):
                prepared.update(token=uuid.uuid4().hex,created=time.monotonic());self.pending_view=prepared
                return dict(preview_token=prepared['token'],name=meta['name'],item=prepared['item'],features=len(f['points']),metric_registered=meta.get('metric_registered',True))
            result=self.save_view(prepared);self.frozen=None;return result
        if action=='visual_bind':
            # 兼容旧客户端请求：保存也只用已有参照，不读相机、不识别或刷新硬件。
            from .visual_reference import make_binding
            value=make_binding(self.store,cmd['reference'],cmd['side'],cmd['mode'],cmd['pose'],self.geometry)
            return dict(binding=value,diagnostics=dict(method='stored_initial_reference'))
        if action!='visual_correct':raise ValueError('未知物体参照操作')
        from .task_flow import full_record
        record=full_record(cmd['record']);bindings=record_bindings(record)
        if not bindings:raise ValueError('当前动作没有关联物体的姿态')
        self.robot.refresh_measured();base=self.robot.base_sample();frame,source,T=self.sample()
        current=None;observations={};diagnostics=[]
        for ident in dict.fromkeys(v['reference'] for v in bindings.values()):
            self.cancelled(cmd);meta,arrays=self.store.load(ident)
            if meta['model_sha256']!=self.model_hash() or meta['frame']!=self.geometry.root:raise ValueError('本机模型或坐标系改变，请重新示教物体参照')
            self.check_mount(meta,source,T)
            position_only=all(v['mode']=='position' for v in bindings.values() if v['reference']==ident)
            delta,stats,_,_,_=self.observe(ident,frame,T,current,position_only=position_only);old=matrix(meta['object_pose']);new=delta@old
            if any(v['reference']==ident and v['mode']=='rigid' for v in bindings.values()) and stats['angle_resolution_deg']>3:
                raise ValueError('参照旋转证据不足，请重新观察或将该姿态设为“只纠正位置”')
            from scipy.spatial.transform import Rotation
            distance=float(np.linalg.norm(new[:3,3]-old[:3,3]));angle=float(np.degrees(Rotation.from_matrix(delta[:3,:3]).magnitude()))
            if distance>.20 or angle>30:raise ValueError(f'参照变化{distance*1000:.0f}mm/{angle:.1f}°超出本版局部纠偏范围（200mm/30°），请调整停靠位置')
            observations[ident]=(old,new);diagnostics.append(dict(reference=ident,name=meta['name'],offset_mm=((new[:3,3]-old[:3,3])*1000).tolist(),angle_deg=angle,**stats))
        result=clone_without_bindings(record);previous=source;checked=0
        from .pose_ik_worker import PoseSolver
        for i,row in enumerate(result['keyframes']):
            self.cancelled(cmd);bound=bindings.get(str(i))
            if bound:
                side=bound['side']
                if side not in self.solvers:self.solvers[side]=PoseSolver({**self.cfg,'arm':side})
                solver=self.solvers[side];_,new=observations[bound['reference']];old=matrix(bound['taught_object_pose'])
                # 只改变选中七轴；腰腿、另一只手臂及夹爪仍取该示教帧原值。
                target=corrected_tcp(old,new,bound['relative_tcp'],bound['mode'])
                if np.linalg.norm(new[:3,3]-old[:3,3])>.20:raise ValueError('相对该姿态示教位置变化超过200mm，请调整停靠点')
                solved=solver.calculate(dict(operation='solve',source=row['joints_deg'],seed=previous,target=target.tolist()))
                row['joints_deg'].update(solved['joints']);checked+=solved['checked_states']
            previous=row['joints_deg']
        # 复用实际执行器的C2平滑/夹爪时序，并检查平滑后的整条路径。
        from .teaching_trajectory import plan,evaluate
        staged,segments,_=plan(result,source,self.store.directory.parent)
        sides={v['side'] for v in bindings.values()}
        for seg in segments:
            n=max(2,int(np.ceil(max(abs(seg['target'][k]-seg['start'][k]) for k in source)/.25)))
            for u in np.linspace(0,1,n+1):
                self.cancelled(cmd);q=evaluate(seg,float(u))
                for side in sides:
                    guard=self.solvers[side].planner.guard
                    guard.begin(q);guard.check(q)
                checked+=1
        self.stable(source,base);self.cancelled(cmd)
        from .core import atomic_json
        image=self.run/'visual_action_current.png';cv2.imwrite(str(image),frame['image'])
        report=dict(record=result,source=source,base_anchor=base,diagnostics=diagnostics,checked_states=checked,created=time.monotonic(),image=str(image),shape=list(frame['image'].shape))
        atomic_json(self.run/'visual_action_last_preview.json',report)
        return report
