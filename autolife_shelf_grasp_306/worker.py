"""Persistent inference/planning process. Line JSON protocol, read-only robot I/O."""
import argparse
import copy
import hashlib
import contextlib
import json
import queue
import sys
import threading
import time
import uuid
from pathlib import Path
import cv2
import numpy as np
import yaml
from .core import ProductStore,atomic_json,execution_blockers,reset_blockers,corrected_target,product_category,match
from .bottle import estimate,combine,DepthQualityError,planning_product
from .object_geometry import estimate_object
from .segmentation_display import display_contours
from .unregistered import target_id,temporary_product,estimate_temporary,combine_product
from .geometry_memory import GeometryMemory,partial_views
from .bottle import image_cropped
from .auto_grasp import CandidateUnavailable
from .arm_selection import ArmPlannerPool,plan_arm,require_same_arm,gripper_name
from .calibration_profiles import CalibrationProfiles
from .task_recognition import task_detection_request,infer_scene


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--run',required=True)
    args=parser.parse_args();root=Path(args.root);run=Path(args.run)
    protocol=sys.stdout
    def emit(kind,**values):
        protocol.write(json.dumps({'event':kind,**values},ensure_ascii=False,allow_nan=False)+'\n');protocol.flush()
    # Torch/vendor banners must not corrupt the JSON IPC channel.
    with contextlib.redirect_stdout(sys.stderr):
        camera=None;robot=None
        try:
            from .camera import Camera
            from .perception import Perception
            from .robot import Robot
            from .planning import Planner
            from autolife_handeye_calibration_306.core import Kinematics,TORSO,BODY,NECK
            from autolife_handeye_calibration_306.verification_core import camera_transform,load_geometry
            import rclpy
            cfg=yaml.safe_load((root/'config/grasp.yaml').read_text('utf-8'))
            runtime_path=root/'data/settings.json'
            mode='all';task_recognition=None
            cfg['semantic_prompt_overrides']={}
            if runtime_path.exists():
                saved=json.loads(runtime_path.read_text('utf-8'))
                cfg['approach_yaw_deg']=float(saved.get('approach_yaw_deg',cfg['approach_yaw_deg']))
                mode=saved.get('recognition_mode','all')
                if mode not in ('beverages','all'):mode='all'
                cfg['semantic_prompt_overrides']=saved.get('semantic_prompts',{})
            commands=queue.Queue(maxsize=16)
            def read_commands():
                for line in sys.stdin:
                    try:commands.put_nowait(json.loads(line))
                    except (ValueError,queue.Full):continue
                commands.put({'action':'quit'})
            threading.Thread(target=read_commands,daemon=True).start()
            camera=Camera(cfg,run/'preview.jpg')
            rclpy.init(args=[]);robot=Robot();store=ProductStore(root/'data')
            emit('status',text='加载识别模型…')
            try:
                perception=Perception(cfg,root)
            except Exception as exc:
                # Keep RGB, but never accept planning/motion in degraded mode.
                import traceback
                traceback.print_exc(file=sys.stderr)
                error='识别不可用（保留RGB预览）：'+str(exc)
                emit('fatal',text=error)
                while True:
                    cmd=commands.get()
                    if cmd.get('action')=='quit':break
                    emit('error',text=error,request_id=cmd.get('request_id',0))
                return
            fk=Kinematics(cfg['urdf_path'])
            display_fk=Kinematics(cfg['urdf_path'])
            profiles=CalibrationProfiles(cfg,fk)
            calibration=None;calibration_sha=None;tcp=None
            if Path(cfg['calibration']).is_file():
                calibration,calibration_sha=profiles.resolve(cfg)
                tcp=load_geometry(calibration,fk,cfg['tcp_offset_mm'],robot_id=cfg.get('robot_id'))
            planner=None;scene=None;items=[];selected=None;plan=None
            arm_pool=ArmPlannerPool(cfg,calibration,fk,Planner,profiles)
            geometry_memory=GeometryMemory(root/'data/observed_geometry.json')
            emit('ready',blockers=execution_blockers(cfg),products=len(store.products),
                 calibration=Path(cfg['calibration']).parent.name,mode=mode,
                 detector_backend=cfg.get('detector_backend'),semantic_prompt=cfg['semantic_prompt_overrides'].get(mode,''))
            def snapshot(depth=False):
                deadline=time.monotonic()+1.5
                while True:
                    frame,error=camera.get()
                    if frame is not None and time.monotonic()-frame['received']<.3 and (not depth or frame.get('depth') is not None):return frame
                    if time.monotonic()>deadline:raise ValueError(error or '等待新鲜且配对的RGB-D图像')
                    time.sleep(.02)
            def transform(frame):
                if calibration is None:raise ValueError('请先完成306本机手眼标定，并在config/grasp.yaml中设置calibration路径；当前仅可识别录入')
                q=robot.sample(frame['received'],stationary=True)
                return q,camera_transform(calibration,fk,q,frame)
            def overlay_items(values):
                return [dict(id=v['id'],name=v['name'],box=list(v['box']),mask_contours=display_contours(v)) for v in values]
            def project_live_path(points,source_q):
                # Pin planned points in the stationary chassis frame. Head/body
                # motion is reprojected from fresh FK; navigation invalidates it.
                from .demo_protocol import check_base_anchor
                path_calibration=calibration
                ground=fk.fk('Link_Zero_Point',TORSO,source_q)
                fixed=np.asarray(points)@ground[:3,:3].T+ground[:3,3]
                try:anchor=robot.base_sample()
                except ValueError:anchor=None
                def project(frame):
                    if anchor is None:raise ValueError('缺少底盘参考')
                    check_base_anchor(robot.base_sample(),anchor)
                    current=robot.sample(frame['received'],stationary=False)
                    camera_ground=display_fk.fk('Link_Zero_Point',TORSO,current)@camera_transform(path_calibration,display_fk,current,frame,robot_id=cfg.get('robot_id'))
                    inv=np.linalg.inv(camera_ground);points=fixed@inv[:3,:3].T+inv[:3,3]
                    if np.any(points[:,2]<=.05):raise ValueError('路径已离开相机前方')
                    return cv2.projectPoints(points,np.zeros(3),np.zeros(3),frame['K'],frame['D'])[0][:,0,:].tolist()
                return project
            def public_items():
                return [{'id':v['id'],'name':v['name'],'box':v['box'],'mask_contours':display_contours(v),'matched':v['product'] is not None,
                         'product_id':target_id(v,run.name),
                         'geometry':v['product'].get('geometry','cylinder') if v['product'] else None,
                         'category':product_category(v['product']) if v['product'] else None,
                         'dimensions_mm':([v['product']['diameter_m']*1000,v['product']['height_m']*1000,
                            v['product'].get('depth_m',v['product']['diameter_m'])*1000] if v['product'] else None),
                         'grasp_fraction':v.get('grasp_fraction'),
                         'score':round(v['match']['score'],3)} for v in items]
            def catalog():
                return [{k:v for k,v in p.items() if k!='features'} | {'category':product_category(p)} for p in store.products]
            def refresh_items():
                available=store.products
                for item in items:
                    m=match(item['feature'],available,cfg['match_threshold'],cfg['match_margin'])
                    p=next((p for p in available if p['id']==m['id']),None)
                    item.update(product=p,match=m,name=p['name'] if p else '未录入 / 待确认')
                    item.pop('grasp_fraction',None)
            while True:
                try:cmd=commands.get(timeout=.2)
                except queue.Empty:continue
                ident=cmd.get('request_id',0)
                try:
                    action=cmd['action']
                    if action=='quit':break
                    if action=='auto_prepare':
                        from .prepare import ready_after_place,checked_reset_plan
                        q=robot.sample(stationary=True)
                        if ready_after_place(cfg,q):
                            atomic_json(run/'prepare_result.json',dict(status='prepared',q=q,token=uuid.uuid4().hex,time=time.monotonic(),source='automatic_checked'))
                            emit('auto_prepared',request_id=ident)
                        else:
                            plan=None;scene=None;items=[];selected=None
                            if planner is None:planner=Planner(cfg)
                            result=checked_reset_plan(planner,cfg,q)
                            atomic_json(root/'data/last_reset_plan.json',result)
                            emit('reset_planned',request_id=ident,mode='hold_then_vr',blockers=reset_blockers(cfg))
                    elif action=='prepare':
                        from .prepare import checked_reset_plan
                        plan=None;scene=None;items=[];selected=None
                        if planner is None:planner=Planner(cfg)
                        result=checked_reset_plan(planner,cfg,robot.sample(stationary=True))
                        atomic_json(root/'data/last_reset_plan.json',result)
                        emit('reset_planned',request_id=ident,mode='hold_then_vr',blockers=reset_blockers(cfg))
                    elif action=='prepared':
                        receipt=json.loads((run/'prepare_result.json').read_text('utf-8'))
                        if receipt.get('status')!='prepared':raise ValueError('复位没有完成')
                        scene=None;items=[];plan=None
                        emit('prepared',request_id=ident)
                    elif action in ('demo_place','standalone_place'):
                        from .demo_config import validate,require_placeable,footprint_radius
                        from .demo_surface import fuse_stationary_depth,obstacle_boxes,retryable_surface_error,retryable_placement_error
                        from .demo_planning import plan_place,remove_robot_points
                        from .semantic_placement import semantic_table_surface
                        from .placement_support import footprint_outline
                        from .demo_observation import plan_observation
                        from .demo_protocol import check_base_anchor
                        from .bottle import object_points
                        from .placement_request import standalone_source,check_held_gripper
                        if action=='standalone_place':
                            context,grasp,result=standalone_source(run)
                        else:
                            context=json.loads((run/'demo_run.json').read_text())
                            grasp=json.loads((run/('plan_'+context['grasp_plan_id']+'.json')).read_text())
                            result=json.loads((run/'demo_grasp_result.json').read_text())
                            base=json.loads((run/'demo_base_result.json').read_text())
                            if result.get('status')!='completed' or result.get('plan_id')!=grasp['plan_id'] or base.get('status')!='completed':
                                raise ValueError('抓取或底盘阶段尚未成功，不能规划放置')
                        side=require_same_arm(grasp,result)
                        calibration,calibration_sha=profiles.for_held_plan(grasp)
                        # Keep the shared camera/perception config object alive;
                        # restoring held-arm geometry must not lose current prompts.
                        cfg.update({k:v for k,v in grasp['config'].items() if k!='semantic_prompt_overrides'})
                        if grasp.get('calibration_sha256')!=calibration_sha:
                            raise ValueError('抓取后相机标定改变，请检查持物记录')
                        tcp=np.asarray(grasp.get('physical_tcp',grasp['tcp']),float)
                        planner=arm_pool.for_config(cfg)
                        attempt_path=run/'observation_attempt.json'
                        if attempt_path.exists():
                            attempt=json.loads(attempt_path.read_text())
                            observed_path=run/'demo_observe_result.json'
                            observed=json.loads(observed_path.read_text()) if observed_path.exists() else {}
                            if (attempt.get('run_id')!=context['run_id'] or observed.get('status')!='observed' or
                                    observed.get('plan_id')!=attempt.get('plan_id')):
                                raise ValueError('观察姿态尚未成功完成，不能继续放置')
                        settings=validate(context['settings'])
                        from .local_contact import held_tcp
                        placement_tcp=held_tcp(grasp,tcp)
                        require_placeable(grasp['product'])
                        emit('status',text='分割桌面，检查局部支撑与可达放置点…')
                        if planner is None:planner=Planner(cfg)
                        planner.cfg=dict(cfg)
                        observation_started=time.monotonic();point_seconds=0.
                        frames=[];environments=[];seen=set();deadline=time.monotonic()+5;first_q=None;base_anchor=robot.base_sample()
                        while len(frames)<5 and time.monotonic()<deadline:
                            if (run/'stop').exists():raise ValueError('用户停止演示')
                            frame=snapshot(depth=True)
                            if frame['frame_id'] in seen:time.sleep(.025);continue
                            seen.add(frame['frame_id']);q,T=transform(frame)
                            check_held_gripper(q,result)
                            check_base_anchor(robot.base_sample(),base_anchor)
                            if first_q is None:first_q=dict(q)
                            elif any(abs(q[n]-first_q[n])>.4 for n in first_q):
                                raise ValueError('识别桌面时关节发生移动，请停稳后重试')
                            if frames and (not np.array_equal(frame['K'],frames[0]['K']) or not np.array_equal(frame['D'],frames[0]['D'])):
                                raise ValueError('观察期间相机内参改变，请重试')
                            point_started=time.monotonic()
                            ground=fk.fk('Link_Zero_Point',TORSO,q)
                            depth=frame['depth'].astype(float)*cfg['depth_scale_m']
                            mask=np.isfinite(depth)&(depth>.25)&(depth<1.8)
                            cloud,_,_=object_points(frame,mask,ground@T,cfg)
                            environment=remove_robot_points(planner,q,placement_tcp,grasp['product'],cloud)
                            frames.append(frame);environments.append(environment)
                            point_seconds+=time.monotonic()-point_started
                        if len(frames)<5:raise ValueError('未获得五帧新鲜且停稳的桌面深度')
                        capture_finished=time.monotonic()
                        fused={**frame,'depth':fuse_stationary_depth([f['depth'] for f in frames],cfg['depth_scale_m'])}
                        cloud,_,_=object_points(fused,fused['depth']>0,ground@T,cfg)
                        environment=remove_robot_points(planner,q,placement_tcp,grasp['product'],cloud)
                        diag={'frame_ids':[int(f['frame_id']) for f in frames],'method':'stationary_pixel_median_5',
                              'timing_seconds':{'acquisition_and_clouds':round(capture_finished-observation_started,4),
                                                'per_frame_point_processing':round(point_seconds,4),
                                                'fused_depth_processing':round(time.monotonic()-capture_finished,4)}}
                        cfg['up_torso']=(ground[:3,:3].T@np.array([0.,0.,1.])).tolist()
                        planner.cfg=dict(cfg)
                        event='place_planned';table=None;surface_ready=False
                        try:
                            points,table,table_item=semantic_table_surface(perception,fused,ground@T,cfg,environment,
                                settings,footprint_radius(grasp['product']),
                                lambda p:remove_robot_points(planner,q,placement_tcp,grasp['product'],p),diag,
                                support_outline=footprint_outline(grasp['product'],ground[:3,:3]),
                                support_offset=(ground[:3,:3]@np.asarray(cfg.get('target_offset_torso_mm',[0,0,0]))/1000)[:2])
                            surface_ready=True;obstacles_started=time.monotonic()
                            table['obstacles']=obstacle_boxes(environment,table,frames=environments,camera_origin=(ground@T)[:3,3])
                            diag['timing_seconds']['obstacle_grouping']=round(time.monotonic()-obstacles_started,4)
                            diag['timing_seconds']['recognition_total']=round(time.monotonic()-observation_started,4)
                            # Save exact planning inputs, not end-of-grasp feedback:
                            # this permits deterministic diagnosis without a new motion trial.
                            atomic_json(run/'place_input.json',dict(source_q=q,tcp=placement_tcp.tolist(),
                                points=points.tolist(),table=table,settings=settings,
                                grasp_plan_id=context['grasp_plan_id'],config=cfg,
                                camera_ground=(ground@T).tolist(),camera_K=frame['K'].tolist(),
                                camera_D=frame['D'].tolist()))
                            emit('status',text='筛选可达落点，再检查下降、松爪与收回路径…')
                            planning_started=time.monotonic()
                            plan=plan_place(planner,q,placement_tcp,grasp,points,table,settings)
                            diag['timing_seconds']['path_planning']=round(time.monotonic()-planning_started,4)
                        except ValueError as exc:
                            diag['reason']=str(exc)
                            diag['failure_stage']='planning' if surface_ready else 'surface'
                            cv2.imwrite(str(run/'table_failure.jpg'),frame['image'])
                            np.savez_compressed(run/'table_failure.npz',environment=environment,
                                                obstacle_environment=np.concatenate(environments),
                                                frame_point_counts=np.array([len(v) for v in environments]))
                            if surface_ready:
                                refresh=int(cmd.get('_place_refresh',0))
                                diag['fresh_retry']=refresh
                                if retryable_placement_error(getattr(planner,'place_diagnostics',[]),refresh) and not (run/'stop').exists():
                                    atomic_json(run/f'place_retry_{refresh}.json',dict(surface=diag,
                                        planning=getattr(planner,'place_diagnostics',[])))
                                    emit('status',text=f'放置通道深度复核（{refresh+1}/2），保持夹持…')
                                    commands.put_nowait({**cmd,'_place_refresh':refresh+1})
                                    continue
                                raise ValueError('桌面已识别，放置路径未通过检查：'+str(exc)) from exc
                            refresh=int(cmd.get('_place_refresh',0))
                            diag['fresh_retry']=refresh
                            if retryable_surface_error(exc,refresh) and not (run/'stop').exists():
                                emit('status',text=f'桌面深度有波动，重新确认空位（{refresh+1}/2）…')
                                commands.put_nowait({**cmd,'_place_refresh':refresh+1})
                                continue
                            attempt=run/'observation_attempt.json'
                            if attempt.exists():raise ValueError('已调整观察姿态，仍无法放置：'+str(exc)) from exc
                            if table is None:
                                if not all(k in diag for k in ('coef','minimum','maximum','plane_points')) or np.linalg.norm(diag['coef'][:2])>np.tan(np.deg2rad(8)):
                                    raise
                                table={k:diag[k] for k in ('coef','minimum','maximum')}
                                table['support_method']='observed_semantic_plane'
                                table['obstacles']=obstacle_boxes(environment,table,frames=environments,camera_origin=(ground@T)[:3,3])
                            held=result.get('q',{}).get(gripper_name(side))
                            if held is None or abs(q[gripper_name(side)]-held)>8:
                                raise ValueError('夹爪状态与抓取结束时不同，不能自动抬手；请重新开始本轮演示') from exc
                            if (run/'stop').exists():raise ValueError('用户停止演示')
                            a,b,c=table['coef'];lo=table['minimum'];hi=table['maximum']
                            corners=np.array([[x,y,a*x+b*y+c] for x in (lo[0],hi[0]) for y in (lo[1],hi[1])])
                            camera_ground=ground@T;cam=corners@np.linalg.inv(camera_ground)[:3,:3].T+np.linalg.inv(camera_ground)[:3,3]
                            if np.any(cam[:,2]<=.05):raise ValueError('桌面投影不可靠，请调整观察角度')
                            uv=cv2.projectPoints(cam,np.zeros(3),np.zeros(3),frame['K'],frame['D'])[0][:,0,:]
                            h,w=frame['image'].shape[:2];roi=[max(0.,uv[:,0].min()),max(0.,uv[:,1].min()),min(w,uv[:,0].max()),min(h,uv[:,1].max())]
                            emit('status',text='持瓶遮挡或路径受限，检查一次轻抬观察姿态…')
                            try:
                                plan=plan_observation(planner,q,placement_tcp,grasp,table,{'T':camera_ground,'K':frame['K'],'D':frame['D']},roi)
                            except ValueError as observation_error:
                                diag['observation_reason']=str(observation_error)
                                raise ValueError('桌面支撑确认失败：'+str(exc)+'；观察调整未执行：'+str(observation_error)) from observation_error
                            event='observe_planned'
                        finally:
                            diag['timing_seconds']['observation_and_planning_total']=round(time.monotonic()-observation_started,4)
                            atomic_json(run/'table_diagnostics.json',diag)
                            atomic_json(run/'place_diagnostics.json',getattr(planner,'place_diagnostics',[]))
                        plan.update(config=dict(cfg),plan_id=uuid.uuid4().hex,run_id=run.name,
                            arm=side,tool_geometry=grasp.get('tool_geometry',{}),
                            placement_mode='standalone' if action=='standalone_place' else 'demo',placement_request_id=context['run_id'],
                            parent_plan_id=grasp['plan_id'],demo_settings=settings,
                            calibration_sha256=calibration_sha,tcp_offset_mm=cfg['tcp_offset_mm'],created_monotonic=time.monotonic(),base_anchor=base_anchor)
                        filename='place_'+plan['plan_id']+'.json';atomic_json(run/filename,plan)
                        if event=='observe_planned':
                            atomic_json(run/'observation_attempt.json',{'plan_id':plan['plan_id'],'run_id':context['run_id']})
                            emit(event,request_id=ident,filename=filename,plan_id=plan['plan_id'],product_id=plan['product']['id'])
                            continue
                        camera_point=np.linalg.inv(T)@np.r_[plan['center'],1.]
                        pixels=cv2.projectPoints(camera_point[:3].reshape(1,3),np.zeros(3),np.zeros(3),frame['K'],frame['D'])[0][:,0,:].tolist() if camera_point[2]>0 else []
                        cv2.imwrite(str(run/'scene.jpg'),frame['image'])
                        camera.overlay.set(frame,[dict(id=0,name='放置桌面',box=table_item['box'],mask_contours=display_contours(table_item))],ident)
                        camera.overlay.set_path(project_live_path([segment['tcp_goal'] for segment in plan['segments']],q),ident)
                        emit('place_planned',request_id=ident,filename=filename,plan_id=plan['plan_id'],
                             product_id=plan['product']['id'],pixels=pixels,shape=list(frame['image'].shape),
                             table_height=table['coef'][2],seconds=plan['seconds'],
                             table_segmentation=dict(id=0,name='放置桌面',box=table_item['box'],mask_contours=display_contours(table_item)))
                    elif action in ('scan','set_mode','set_prompt','scan_task'):
                        task_recognition=None
                        if action=='scan_task':
                            store.reload()
                            task_recognition=task_detection_request(cmd.get('target',''),store.products)
                            perception.previous=[];items=[];scene=None;plan=None
                        if action=='set_prompt':
                            from .grounded_sam2 import normalize_prompt
                            if cfg.get('detector_backend')!='grounded_sam2':raise ValueError('当前识别器未启用文字分割')
                            text=str(cmd.get('text','')).strip();normalize_prompt(text)
                            cfg['semantic_prompt_overrides']={**cfg['semantic_prompt_overrides'],mode:text}
                            saved=json.loads(runtime_path.read_text('utf-8')) if runtime_path.exists() else {}
                            atomic_json(runtime_path,{**saved,'semantic_prompts':cfg['semantic_prompt_overrides']})
                            perception.previous=[];items=[];scene=None;plan=None
                        if action=='set_mode':
                            if cmd['mode'] not in ('all','beverages'):raise ValueError('识别模式无效')
                            mode=cmd['mode'];perception.previous=[];items=[];scene=None;plan=None
                            saved=json.loads(runtime_path.read_text('utf-8')) if runtime_path.exists() else {}
                            atomic_json(runtime_path,{**saved,'recognition_mode':mode})
                        plan=None;selected=None;scene=snapshot(depth=True)
                        store.reload();items,elapsed=infer_scene(perception,scene,store,mode,task_recognition)
                        atomic_json(run/'recognition_diagnostics.json',dict(request_id=ident,action=action,
                            task_recognition=task_recognition,manual_mode=mode,
                            detector=dict(perception.segmenter.timings) if perception.segmenter else {},
                            items=[dict(id=v['id'],name=v['name'],label=v.get('semantic_label'),
                                product_id=v['product']['id'] if v.get('product') else None) for v in items]))
                        cv2.imwrite(str(run/'scene.jpg'),scene['image'])
                        camera.overlay.set(scene,overlay_items(items),ident)
                        emit('scene',request_id=ident,items=public_items(),shape=list(scene['image'].shape),elapsed_ms=elapsed,mode=mode,
                             semantic_prompt=cfg['semantic_prompt_overrides'].get(mode,''),
                             semantic_prompt_resolved=(perception.segmenter.timings.get('prompt','') if perception.segmenter else ''),
                             task_recognition=task_recognition)
                    elif action in ('catalog','update_product','remove_sample','remove_product'):
                        store.reload()
                        if action=='update_product':store.update(cmd['product_id'],**cmd['values'])
                        elif action=='remove_sample':store.remove_example(cmd['product_id'],cmd['index'])
                        elif action=='remove_product':store.remove_product(cmd['product_id'])
                        plan=None;selected=None;refresh_items()
                        camera.overlay.rename(overlay_items(items),ident)
                        emit('catalog',request_id=ident,products=catalog(),items=public_items(),product_id=cmd.get('product_id'))
                    elif action=='enroll':
                        item=next(v for v in items if v['id']==int(cmd['id']))
                        p=store.add(cmd['name'],float(cmd['diameter_mm']),float(cmd['height_mm']),item['feature'],item['crop'],
                                    cmd.get('geometry','cylinder'),cmd.get('depth_mm'),cmd.get('category','general'))
                        refresh_items();item.update(product=p,name=p['name']);plan=None
                        camera.overlay.rename(overlay_items(items),ident)
                        emit('enrolled',request_id=ident,text=f'已录入 {p["name"]} · {len(p["features"])} 张样本',items=public_items(),select_id=item['id'])
                    elif action=='grasp_point':
                        item=next(v for v in items if v['id']==int(cmd['id']))
                        if not item['product'] or item['product'].get('geometry')!='bag_edge':raise ValueError('只有袋口类型需要选择夹取点')
                        h,w=item['mask'].shape;x=float(cmd['x'])*w;y=float(cmd['y'])*h
                        x0,y0,x1,y1=item['box']
                        if not (0<=x<w and 0<=y<h and item['mask'][int(y),int(x)]):raise ValueError('请点在袋口实体区域内，避开背景和提绳')
                        if y>y0+(y1-y0)*.55:raise ValueError('请选择袋子上半部的袋口，不要夹底部')
                        item['grasp_fraction']=[(x-x0)/(x1-x0),(y-y0)/(y1-y0)]
                        plan=None
                        camera.overlay.rename(overlay_items(items),ident)
                        emit('enrolled',request_id=ident,text='袋口夹取点已设置；请先预览路径',items=public_items(),select_id=item['id'])
                    elif action=='plan':
                        # Begin each request from the unchanged base (left) profile.
                        # The chosen arm then owns camera extrinsics AND its TCP.
                        cfg['calibration']=profiles.base['calibration']
                        calibration,calibration_sha=profiles.resolve(cfg)
                        from .target_offsets import checked_profiles,defaults
                        cfg['target_offsets_by_arm_torso_mm']=checked_profiles(cmd.get('target_offsets_by_arm_torso_mm',defaults(arm_pool.base)))
                        plan=None;request_started=time.monotonic()
                        (run/'depth_quality.json').unlink(missing_ok=True)
                        chosen=next(v for v in items if v['id']==int(cmd['id']))
                        emit('status',text='重新识别目标并检查深度…')
                        # Re-detect current target. Never execute a cached bounding box.
                        store.reload();frame=snapshot(depth=True);fresh,_=infer_scene(perception,frame,store,mode,task_recognition)
                        from .selection import resolve_selected
                        selected,identity=resolve_selected(chosen,fresh)
                        atomic_json(run/'target_identity.json',identity)
                        if selected is None:raise ValueError(identity['reason'])
                        # Preserve the selection/reference image coordinates for
                        # enrollment and bag-rim picking. Display flow already
                        # follows the live RGB; fresh masks above are for planning.
                        camera.overlay.rename(overlay_items(items),ident)
                        temporary=selected['product'] is None
                        if temporary:
                            selected={**selected,'product':temporary_product(selected,run.name)}
                            emit('status',text='未录入物品：自动估计尺寸并检查多帧深度…')
                        dimension_reference=geometry_memory.find(selected) if temporary and image_cropped(selected['mask']) else None
                        recognition_finished=time.monotonic()
                        shape=selected['product'].get('geometry','cylinder');grasp_pixel=None
                        if shape=='bag_edge':
                            fraction=chosen.get('grasp_fraction')
                            if not fraction:raise ValueError('请先点击“选择袋口夹取点”并在图像中选点')
                            x0,y0,x1,y1=selected['box'];grasp_pixel=[x0+fraction[0]*(x1-x0),y0+fraction[1]*(y1-y0)]
                            a,b,c,d=chosen['box'];old_pixel=[a+fraction[0]*(c-a),b+fraction[1]*(d-b)]
                            if np.linalg.norm(np.asarray(grasp_pixel)-old_pixel)>8:
                                raise ValueError('袋口或分割轮廓已变化，请重新识别并选择袋口夹取点')
                            if not selected['mask'][int(grasp_pixel[1]),int(grasp_pixel[0])]:raise ValueError('袋口形状/位置改变，请重新选择夹取点')
                        samples=[];ids=set();quality_rejections=[];deadline=time.monotonic()+3.;mask=selected['mask'];source_q=None
                        paired_frames=[];measured_product=copy.deepcopy(selected['product'])
                        fused=None;fusion_error='';fused_product=None
                        while len(ids)<10 and time.monotonic()<deadline:
                            frame=snapshot(depth=True)
                            if frame['frame_id'] in ids:time.sleep(.02);continue
                            q,T=transform(frame)
                            if source_q is None:source_q=dict(q)
                            if max(abs(q[n]-source_q[n]) for n in BODY+NECK)>.5:raise ValueError('身体或头部已移动，请重新规划')
                            up=fk.fk('Link_Zero_Point',TORSO,q)[:3,:3].T@np.array([0.,0.,1.])
                            ids.add(frame['frame_id'])
                            paired_frames.append((frame,dict(q),up.copy()))
                            try:
                                if temporary:sample=estimate_temporary(frame,mask,T,up,selected['product'],cfg,dimension_reference)
                                else:sample=(estimate(frame,mask,T,up,selected['product'],cfg) if shape=='cylinder'
                                    else estimate_object(frame,mask,T,up,selected['product'],cfg,grasp_pixel))
                            except DepthQualityError as exc:
                                quality_rejections.append(str(exc))
                                if len(quality_rejections)==1:emit('status',text='正在重新确认物体深度：'+str(exc))
                                continue
                            samples.append(sample)
                            if len(samples)>=5:
                                try:
                                    fused_product=combine_product(selected['product'],samples) if temporary else None
                                    fused=combine(samples,indices=fused_product['dimension_inlier_indices'] if temporary else None)
                                except ValueError as exc:
                                    fusion_error=str(exc)
                                    if len(samples)==5:emit('status',text='尺寸边缘有波动，补充新帧确认…')
                                    continue
                                break
                        atomic_json(run/'depth_quality.json',{'product':selected['product']['name'],
                            'request_id':ident,
                            'diameter_mm':selected['product'].get('diameter_m',0)*1000,'attempted_frames':len(ids),
                            'valid_frames':len(samples),'rejected_frames':quality_rejections,
                            'fusion_error':fusion_error if fused is None else '',
                            'fusion':fused[1] if fused is not None else None,
                            'fused_product':fused_product if fused is not None else None,
                            'accepted_frames':[{'center':np.asarray(c).tolist(),**stats} for c,stats in samples]})
                        if len(samples)<5 and quality_rejections:
                            raise CandidateUnavailable(f'仅{len(samples)}/{len(ids)}帧深度合格，未生成运动路径；'+quality_rejections[-1])
                        if fused is None:raise CandidateUnavailable(fusion_error or '有效深度帧不足，请重试当前步骤')
                        center,geometry=fused;cfg['up_torso']=up.tolist()
                        if temporary:
                            selected['product']=fused_product
                            geometry.update(temporary=True,dimension_source=selected['product']['dimension_source'])
                            if not geometry.get('image_cropped'):
                                views=partial_views(frame['image'],mask)
                                if views:selected['partial_features']=perception.embed(views)
                            geometry_memory.remember(selected,selected['product'],geometry)
                        emit('status',text='检查预抓取 → 抓取 → 抬起 → 收回…')
                        depth_finished=time.monotonic()
                        planning_started=time.monotonic()
                        planned_product=planning_product(selected['product'],geometry)
                        # Re-estimate the full shape from identical paired RGB-D
                        # observations in the selected arm's calibrated camera frame.
                        # Do not mix a right TCP with left camera coordinates.
                        arm_observations={}
                        def observation_for_arm(side,side_cfg):
                            data,digest=profiles.resolve(side_cfg)
                            if digest==calibration_sha:
                                value=(fused[0],copy.deepcopy(geometry),copy.deepcopy(selected['product']),T)
                            else:
                                estimates=[];rejected=[];last_T=None
                                for observed,observed_q,observed_up in paired_frames:
                                    if (run/'stop').exists():raise ValueError('用户停止')
                                    last_T=camera_transform(data,fk,observed_q,observed,robot_id=cfg.get('robot_id'))
                                    try:
                                        if temporary:
                                            sample=estimate_temporary(observed,mask,last_T,observed_up,measured_product,side_cfg,dimension_reference)
                                        else:
                                            sample=(estimate(observed,mask,last_T,observed_up,measured_product,side_cfg) if shape=='cylinder'
                                                else estimate_object(observed,mask,last_T,observed_up,measured_product,side_cfg,grasp_pixel))
                                        estimates.append(sample)
                                    except DepthQualityError as exc:rejected.append(str(exc))
                                if len(estimates)<5:raise ValueError('该手臂标定下有效深度帧不足：'+(rejected[-1] if rejected else '不足5帧'))
                                product=combine_product(measured_product,estimates) if temporary else copy.deepcopy(measured_product)
                                position,stats=combine(estimates,indices=product['dimension_inlier_indices'] if temporary else None)
                                if temporary:stats.update(temporary=True,dimension_source=product['dimension_source'])
                                value=(position,stats,product,last_T)
                            arm_observations[side]=value
                            return value[0],planning_product(value[2],value[1])
                        atomic_json(run/'grasp_input.json',dict(source_q=q,
                            center=center.tolist(),product=planned_product,config={**arm_pool.base,'target_offsets_by_arm_torso_mm':cfg['target_offsets_by_arm_torso_mm']},
                            up_torso=up.tolist(),geometry=geometry,request_id=ident,center_uncorrected=True))
                        try:
                            plan,planner,selected_cfg,tcp,center,correction=arm_pool.plan(q,center,planned_product,cfg,
                                stopped=lambda:(run/'stop').exists(),on_status=lambda text:emit('status',text=text),
                                observation_for_arm=observation_for_arm)
                            cfg.update(selected_cfg)
                            calibration,calibration_sha=profiles.resolve(cfg)
                            _,geometry,selected['product'],T=arm_observations[cfg['arm']]
                            planned_product=planning_product(selected['product'],geometry)
                            if cfg.get('calibration_profile',{}).get('test_only'):
                                emit('status',text='右臂使用本次测试标定（待实物精度验证）')
                        except ValueError as exc:raise CandidateUnavailable(str(exc)) from exc
                        finally:atomic_json(run/'grasp_diagnostics.json',dict(arms=arm_pool.attempts,failures=arm_pool.diagnostics))
                        geometry.update(correction)
                        atomic_json(root/'data/last_geometry.json',{'center':center.tolist(),'up_torso':up.tolist(),**geometry})
                        atomic_json(run/'grasp_input.json',dict(source_q=q,tcp=tcp.tolist(),
                            center=center.tolist(),product=planned_product,config=cfg,geometry=geometry,request_id=ident))
                        plan['timing_seconds']={'recognition':round(recognition_finished-request_started,3),
                            'depth':round(depth_finished-recognition_finished,3),
                            'model_load':round(planning_started-depth_finished,3),
                            'planning':round(time.monotonic()-planning_started,3),
                            'total':round(time.monotonic()-request_started,3)}
                        plan['calibration']=cfg['calibration'];plan['created_monotonic']=time.monotonic()
                        if hashlib.sha256(Path(cfg['calibration']).read_bytes()).hexdigest()!=calibration_sha:
                            raise ValueError('标定文件已改变，请重开抓取程序')
                        plan['calibration_sha256']=calibration_sha;plan['tcp_offset_mm']=cfg['tcp_offset_mm']
                        plan['config']=dict(cfg);plan['frame_id']=frame['frame_id']
                        from .grip_width import closing_command
                        plan['grip_command']=closing_command(plan['grasp_width_m'],cmd.get('grip_tightening_mm',5.),cfg)
                        plan['geometry']=geometry
                        receipt_path=run/'prepare_result.json'
                        receipt=json.loads(receipt_path.read_text('utf-8')) if receipt_path.exists() else {}
                        plan['prepare_token']=receipt.get('token') if receipt.get('status')=='prepared' else None
                        plan['plan_id']=uuid.uuid4().hex;plan['run_id']=run.name
                        plan['instance_id']=selected['id'];plan['request_id']=ident
                        filename='plan_'+plan['plan_id']+'.json'
                        atomic_json(run/filename,plan)
                        atomic_json(root/'data/last_plan.json',plan)
                        points=np.array([segment['tcp_goal'] for segment in plan['segments']])
                        if plan.get('grasp_contact_mode')=='front_local':
                            points=np.array([planner.pose(segment['waypoints'][-1],plan['physical_tcp'])[0] for segment in plan['segments']])
                        inv=np.linalg.inv(T);camera_points=points@inv[:3,:3].T+inv[:3,3]
                        if np.any(camera_points[:,2]<=0):pixels=[]
                        else:pixels=cv2.projectPoints(camera_points,np.zeros(3),np.zeros(3),frame['K'],frame['D'])[0][:,0,:].tolist()
                        cv2.imwrite(str(run/'scene.jpg'),frame['image'])
                        camera.overlay.set_path(project_live_path(points,q),ident)
                        emit('planned',request_id=ident,center=center.tolist(),pixels=pixels,shape=list(frame['image'].shape),
                             grip_command=plan['grip_command'],
                             arm=cfg['arm'],tool_geometry=cfg.get('tool_geometry',{}),
                             plan_id=plan['plan_id'],filename=filename,prepare_token=plan['prepare_token'],
                             target={'instance_id':selected['id'],'product_id':selected['product']['id'],'name':selected['name']},
                             estimated_dimensions_mm=([selected['product'][k]*1000 for k in ('diameter_m','height_m','depth_m')] if temporary else None),
                             orientation_mode=plan['orientation_mode'],orientation_deviation=plan['max_grasp_orientation_deviation_deg'],
                             approach_yaw_deg=plan['approach_yaw_deg'],
                             grasp_contact_mode=plan.get('grasp_contact_mode','center'),
                             partial_view=bool(geometry.get('image_cropped')),
                             geometry_reference=selected['product'].get('geometry_reference'),
                             contact_center=plan.get('contact_center',center.tolist()),
                             withdraw_orientation_mode=plan.get('withdraw_orientation_mode'),
                             carry_protocol_version=plan.get('carry_protocol_version'),
                             max_withdraw_tilt_deg=plan.get('max_withdraw_tilt_deg',0.),
                             seconds=round(time.monotonic()-request_started,2),planning_seconds=plan['seconds'],
                             checked=plan['checked_states'],
                             blockers=execution_blockers(cfg)+([] if plan['prepare_token'] else ['复位尚未成功，请先复位并准备']))
                    else:raise ValueError('未知操作')
                except Exception as exc:
                    atomic_json(run/'last_error.json',{'action':cmd.get('action'),'request_id':ident,'reason':str(exc),'time':time.time()})
                    plan=None;emit('error',request_id=ident,text=str(exc),
                                   candidate_unavailable=isinstance(exc,CandidateUnavailable))
        except Exception as exc:emit('fatal',text=str(exc))
        finally:
            if camera:camera.close()
            if robot:robot.close()


if __name__=='__main__':main()
