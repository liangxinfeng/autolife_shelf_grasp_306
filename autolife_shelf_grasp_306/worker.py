"""Persistent inference/planning process. Line JSON protocol, read-only robot I/O."""
import argparse
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
            from autolife_handeye_calibration_307.core import Kinematics,TORSO,BODY,NECK
            from autolife_handeye_calibration_307.verification_core import camera_transform,load_geometry
            import rclpy
            cfg=yaml.safe_load((root/'config/grasp.yaml').read_text('utf-8'))
            runtime_path=root/'data/settings.json'
            mode='all'
            if runtime_path.exists():
                saved=json.loads(runtime_path.read_text('utf-8'))
                cfg['approach_yaw_deg']=float(saved.get('approach_yaw_deg',cfg['approach_yaw_deg']))
                mode=saved.get('recognition_mode','all')
                if mode not in ('beverages','all'):mode='all'
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
            perception=Perception(cfg,root)
            fk=Kinematics(cfg['urdf_path'])
            calibration=None;calibration_sha=None;tcp=None
            if Path(cfg['calibration']).is_file():
                calibration_bytes=Path(cfg['calibration']).read_bytes()
                calibration=json.loads(calibration_bytes);calibration_sha=hashlib.sha256(calibration_bytes).hexdigest()
                tcp=load_geometry(calibration,fk,cfg['tcp_offset_mm'])
            planner=None;scene=None;items=[];selected=None;plan=None
            emit('ready',blockers=execution_blockers(cfg),products=len(store.products),
                 calibration=Path(cfg['calibration']).parent.name,mode=mode)
            def snapshot(depth=False):
                deadline=time.monotonic()+1.5
                while True:
                    frame,error=camera.get()
                    if frame is not None and time.monotonic()-frame['received']<.3 and (not depth or frame.get('depth') is not None):return frame
                    if time.monotonic()>deadline:raise ValueError(error or '等待新鲜且配对的RGB-D图像')
                    time.sleep(.02)
            def transform(frame):
                if calibration is None:raise ValueError('请先完成307本机手眼标定，并在config/grasp.yaml中设置calibration路径；当前仅可识别录入')
                q=robot.sample(frame['received'],stationary=True)
                return q,camera_transform(calibration,fk,q,frame)
            def public_items():
                return [{'id':v['id'],'name':v['name'],'box':v['box'],'matched':v['product'] is not None,
                         'product_id':v['product']['id'] if v['product'] else None,
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
                    if action=='prepare':
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
                    elif action=='demo_place':
                        from .demo_config import validate,require_drink
                        from .demo_surface import surfaces,fuse_stationary_depth,obstacle_boxes,retryable_surface_error
                        from .demo_planning import plan_place,remove_robot_points
                        from .demo_observation import plan_observation
                        from .demo_protocol import check_base_anchor
                        from .bottle import object_points
                        context=json.loads((run/'demo_run.json').read_text())
                        attempt_path=run/'observation_attempt.json'
                        if attempt_path.exists():
                            attempt=json.loads(attempt_path.read_text())
                            observed_path=run/'demo_observe_result.json'
                            observed=json.loads(observed_path.read_text()) if observed_path.exists() else {}
                            if (attempt.get('run_id')!=context['run_id'] or observed.get('status')!='observed' or
                                    observed.get('plan_id')!=attempt.get('plan_id')):
                                raise ValueError('观察姿态尚未成功完成，不能继续放置')
                        settings=validate(context['settings'])
                        parent=run/('plan_'+context['grasp_plan_id']+'.json')
                        grasp=json.loads(parent.read_text());require_drink(grasp['product'])
                        result=json.loads((run/'demo_grasp_result.json').read_text())
                        base=json.loads((run/'demo_base_result.json').read_text())
                        if result.get('status')!='completed' or result.get('plan_id')!=grasp['plan_id'] or base.get('status')!='completed':
                            raise ValueError('抓取或底盘阶段尚未成功，不能规划放置')
                        emit('status',text='识别桌面与空闲放置区域…')
                        if planner is None:planner=Planner(cfg)
                        planner.cfg=dict(cfg)
                        frames=[];environments=[];seen=set();deadline=time.monotonic()+5;first_q=None;base_anchor=robot.base_sample()
                        while len(frames)<5 and time.monotonic()<deadline:
                            if (run/'stop').exists():raise ValueError('用户停止演示')
                            frame=snapshot(depth=True)
                            if frame['frame_id'] in seen:time.sleep(.025);continue
                            seen.add(frame['frame_id']);q,T=transform(frame)
                            check_base_anchor(robot.base_sample(),base_anchor)
                            if first_q is None:first_q=dict(q)
                            elif any(abs(q[n]-first_q[n])>.4 for n in first_q):
                                raise ValueError('识别桌面时关节发生移动，请停稳后重试')
                            if frames and (not np.array_equal(frame['K'],frames[0]['K']) or not np.array_equal(frame['D'],frames[0]['D'])):
                                raise ValueError('观察期间相机内参改变，请重试')
                            ground=fk.fk('Link_Zero_Point',TORSO,q)
                            depth=frame['depth'].astype(float)*cfg['depth_scale_m']
                            mask=np.isfinite(depth)&(depth>.25)&(depth<1.8)
                            cloud,_,_=object_points(frame,mask,ground@T,cfg)
                            environment=remove_robot_points(planner,q,tcp,grasp['product'],cloud)
                            frames.append(frame);environments.append(environment)
                        if len(frames)<5:raise ValueError('未获得五帧新鲜且停稳的桌面深度')
                        fused={**frame,'depth':fuse_stationary_depth([f['depth'] for f in frames],cfg['depth_scale_m'])}
                        cloud,_,_=object_points(fused,fused['depth']>0,ground@T,cfg)
                        environment=remove_robot_points(planner,q,tcp,grasp['product'],cloud)
                        diag={'frame_ids':[int(f['frame_id']) for f in frames],'method':'stationary_pixel_median_5'}
                        cfg['up_torso']=(ground[:3,:3].T@np.array([0.,0.,1.])).tolist()
                        planner.cfg=dict(cfg)
                        event='place_planned';table=None
                        try:
                            points,table=surfaces(environment,settings,grasp['product']['diameter_m']/2,diag)
                            table['obstacles']=obstacle_boxes(environment,table,frames=environments)
                            emit('status',text='检查放置、松爪与空手收回路径…')
                            plan=plan_place(planner,q,tcp,grasp,points,table,settings)
                        except ValueError as exc:
                            diag['reason']=str(exc)
                            cv2.imwrite(str(run/'table_failure.jpg'),frame['image'])
                            np.savez_compressed(run/'table_failure.npz',environment=environment,
                                                obstacle_environment=np.concatenate(environments),
                                                frame_point_counts=np.array([len(v) for v in environments]))
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
                                table['obstacles']=obstacle_boxes(environment,table,frames=environments)
                            held=result.get('q',{}).get('Joint_Left_Gripper')
                            if held is None or abs(q['Joint_Left_Gripper']-held)>8:
                                raise ValueError('夹爪状态与抓取结束时不同，不能自动抬手；请重新开始本轮演示') from exc
                            if (run/'stop').exists():raise ValueError('用户停止演示')
                            a,b,c=table['coef'];lo=table['minimum'];hi=table['maximum']
                            corners=np.array([[x,y,a*x+b*y+c] for x in (lo[0],hi[0]) for y in (lo[1],hi[1])])
                            camera_ground=ground@T;cam=corners@np.linalg.inv(camera_ground)[:3,:3].T+np.linalg.inv(camera_ground)[:3,3]
                            if np.any(cam[:,2]<=.05):raise ValueError('桌面投影不可靠，请调整观察角度')
                            uv=cv2.projectPoints(cam,np.zeros(3),np.zeros(3),frame['K'],frame['D'])[0][:,0,:]
                            h,w=frame['image'].shape[:2];roi=[max(0.,uv[:,0].min()),max(0.,uv[:,1].min()),min(w,uv[:,0].max()),min(h,uv[:,1].max())]
                            emit('status',text='持瓶遮挡或路径受限，检查一次轻抬观察姿态…')
                            plan=plan_observation(planner,q,tcp,grasp,table,{'T':camera_ground,'K':frame['K'],'D':frame['D']},roi)
                            event='observe_planned'
                        finally:
                            atomic_json(run/'table_diagnostics.json',diag)
                            atomic_json(run/'place_diagnostics.json',getattr(planner,'place_diagnostics',[]))
                        plan.update(config=dict(cfg),plan_id=uuid.uuid4().hex,run_id=run.name,
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
                        emit('place_planned',request_id=ident,filename=filename,plan_id=plan['plan_id'],
                             product_id=plan['product']['id'],pixels=pixels,shape=list(frame['image'].shape),
                             table_height=table['coef'][2],seconds=plan['seconds'])
                    elif action in ('scan','set_mode'):
                        if action=='set_mode':
                            if cmd['mode'] not in ('all','beverages'):raise ValueError('识别模式无效')
                            mode=cmd['mode'];perception.previous=[];items=[];scene=None;plan=None
                            saved=json.loads(runtime_path.read_text('utf-8')) if runtime_path.exists() else {}
                            atomic_json(runtime_path,{**saved,'recognition_mode':mode})
                        plan=None;selected=None;scene=snapshot(depth=True)
                        store.reload();items,elapsed=perception.infer(scene,store,mode)
                        cv2.imwrite(str(run/'scene.jpg'),scene['image'])
                        emit('scene',request_id=ident,items=public_items(),shape=list(scene['image'].shape),elapsed_ms=elapsed,mode=mode)
                    elif action in ('catalog','update_product','remove_sample','remove_product'):
                        store.reload()
                        if action=='update_product':store.update(cmd['product_id'],**cmd['values'])
                        elif action=='remove_sample':store.remove_example(cmd['product_id'],cmd['index'])
                        elif action=='remove_product':store.remove_product(cmd['product_id'])
                        plan=None;selected=None;refresh_items()
                        emit('catalog',request_id=ident,products=catalog(),items=public_items(),product_id=cmd.get('product_id'))
                    elif action=='enroll':
                        item=next(v for v in items if v['id']==int(cmd['id']))
                        p=store.add(cmd['name'],float(cmd['diameter_mm']),float(cmd['height_mm']),item['feature'],item['crop'],
                                    cmd.get('geometry','cylinder'),cmd.get('depth_mm'),cmd.get('category','general'))
                        refresh_items();item.update(product=p,name=p['name']);plan=None
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
                        emit('enrolled',request_id=ident,text='袋口夹取点已设置；请先预览路径',items=public_items(),select_id=item['id'])
                    elif action=='plan':
                        plan=None;request_started=time.monotonic()
                        chosen=next(v for v in items if v['id']==int(cmd['id']))
                        if chosen['product'] is None:raise ValueError('先为选中的饮料录入商品信息')
                        emit('status',text='重新识别目标并检查深度…')
                        # Re-detect current target. Never execute a cached bounding box.
                        store.reload();frame=snapshot(depth=True);fresh,_=perception.infer(frame,store,mode)
                        selected=next((v for v in fresh if v['id']==chosen['id']),None)
                        if selected is None or selected['product'] is None or selected['product']['id']!=chosen['product']['id']:
                            raise ValueError('选中目标已变化，请重新识别并选择')
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
                        while len(samples)<5 and len(ids)<10 and time.monotonic()<deadline:
                            frame=snapshot(depth=True)
                            if frame['frame_id'] in ids:time.sleep(.02);continue
                            q,T=transform(frame)
                            if source_q is None:source_q=dict(q)
                            if max(abs(q[n]-source_q[n]) for n in BODY+NECK)>.5:raise ValueError('身体或头部已移动，请重新规划')
                            up=fk.fk('Link_Zero_Point',TORSO,q)[:3,:3].T@np.array([0.,0.,1.])
                            ids.add(frame['frame_id'])
                            try:
                                sample=(estimate(frame,mask,T,up,selected['product'],cfg) if shape=='cylinder'
                                    else estimate_object(frame,mask,T,up,selected['product'],cfg,grasp_pixel))
                            except DepthQualityError as exc:
                                quality_rejections.append(str(exc))
                                if len(quality_rejections)==1:emit('status',text='瓶身深度短时波动，正在重新采集…')
                                continue
                            samples.append(sample)
                        atomic_json(run/'depth_quality.json',{'product':selected['product']['name'],
                            'diameter_mm':selected['product']['diameter_m']*1000,'attempted_frames':len(ids),
                            'valid_frames':len(samples),'rejected_frames':quality_rejections})
                        if len(samples)<5 and quality_rejections:
                            raise ValueError(f'仅{len(samples)}/{len(ids)}帧深度合格，未生成运动路径；'+quality_rejections[-1])
                        center,geometry=combine(samples);cfg['up_torso']=up.tolist()
                        center,correction=corrected_target(center,cfg.get('target_offset_torso_mm',[0.,0.,0.]))
                        geometry.update(correction)
                        atomic_json(root/'data/last_geometry.json',{'center':center.tolist(),'up_torso':up.tolist(),**geometry})
                        emit('status',text='检查预抓取 → 抓取 → 抬起 → 收回…')
                        depth_finished=time.monotonic()
                        if planner is None:planner=Planner(cfg)
                        planner.cfg=dict(cfg)
                        planning_started=time.monotonic()
                        plan=planner.plan(q,tcp,center,planning_product(selected['product'],geometry))
                        plan['timing_seconds']={'recognition':round(recognition_finished-request_started,3),
                            'depth':round(depth_finished-recognition_finished,3),
                            'model_load':round(planning_started-depth_finished,3),
                            'planning':round(time.monotonic()-planning_started,3),
                            'total':round(time.monotonic()-request_started,3)}
                        plan['calibration']=cfg['calibration'];plan['created_monotonic']=time.monotonic()
                        if hashlib.sha256(Path(cfg['calibration']).read_bytes()).hexdigest()!=calibration_sha:
                            raise ValueError('标定文件已改变，请重开抓取程序')
                        plan['calibration_sha256']=calibration_sha;plan['tcp_offset_mm']=cfg['tcp_offset_mm']
                        plan['config']=cfg;plan['frame_id']=frame['frame_id']
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
                        inv=np.linalg.inv(T);camera_points=points@inv[:3,:3].T+inv[:3,3]
                        if np.any(camera_points[:,2]<=0):pixels=[]
                        else:pixels=cv2.projectPoints(camera_points,np.zeros(3),np.zeros(3),frame['K'],frame['D'])[0][:,0,:].tolist()
                        cv2.imwrite(str(run/'scene.jpg'),frame['image'])
                        emit('planned',request_id=ident,center=center.tolist(),pixels=pixels,shape=list(frame['image'].shape),
                             plan_id=plan['plan_id'],filename=filename,prepare_token=plan['prepare_token'],
                             target={'instance_id':selected['id'],'product_id':selected['product']['id'],'name':selected['name']},
                             orientation_mode=plan['orientation_mode'],orientation_deviation=plan['max_grasp_orientation_deviation_deg'],
                             approach_yaw_deg=plan['approach_yaw_deg'],
                             withdraw_orientation_mode=plan.get('withdraw_orientation_mode'),
                             carry_protocol_version=plan.get('carry_protocol_version'),
                             max_withdraw_tilt_deg=plan.get('max_withdraw_tilt_deg',0.),
                             seconds=round(time.monotonic()-request_started,2),planning_seconds=plan['seconds'],
                             checked=plan['checked_states'],
                             blockers=execution_blockers(cfg)+([] if plan['prepare_token'] else ['复位尚未成功，请先复位并准备']))
                    else:raise ValueError('未知操作')
                except Exception as exc:
                    atomic_json(run/'last_error.json',{'action':cmd.get('action'),'request_id':ident,'reason':str(exc),'time':time.time()})
                    plan=None;emit('error',request_id=ident,text=str(exc))
        except Exception as exc:emit('fatal',text=str(exc))
        finally:
            if camera:camera.close()
            if robot:robot.close()


if __name__=='__main__':main()
