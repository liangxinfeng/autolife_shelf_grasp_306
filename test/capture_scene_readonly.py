"""Capture an RGB-D planning fixture; absolutely no hardware publishers."""
import json
import time
from pathlib import Path
import yaml
import numpy as np
import cv2
import rclpy
from autolife_shelf_grasp_306.camera import Camera
from autolife_shelf_grasp_306.robot import Robot
from autolife_shelf_grasp_306.perception import Perception
from autolife_shelf_grasp_306.core import ProductStore,depth_cloud
from autolife_handeye_calibration_307.core import Kinematics,TORSO
from autolife_handeye_calibration_307.verification_core import camera_transform

root=Path('/home/ubuntu/ros2_ws/src/autolife_shelf_grasp_306');cfg=yaml.safe_load((root/'config/grasp.yaml').read_text())
data=json.loads(Path(cfg['calibration']).read_text());fk=Kinematics(cfg['urdf_path'])
rclpy.init(args=[]);robot=Robot();camera=Camera(cfg,'/tmp/shelf306_support_camera.jpg')
try:
    model=Perception(cfg,root);store=ProductStore(root/'data');deadline=time.monotonic()+8
    while time.monotonic()<deadline:
        f,error=camera.get()
        if f is None or f.get('depth') is None:time.sleep(.02);continue
        items,_=model.infer(f,store)
        q=robot.sample(f['received'],stationary=True);T=camera_transform(data,fk,q,f)
        up=fk.fk('Link_Zero_Point',TORSO,q)[:3,:3].T@np.array([0.,0.,1.])
        fields={'image':f['image'],'depth':f['depth'],'K':f['K'],'D':f['D'],'T':T,'up':up,
                'meta':np.array(json.dumps({'q':q,'intrinsics':f['intrinsics'],'frame_id':f['frame_id'],
                       'products':[v['product'] for v in items],'boxes':[v['box'] for v in items]}))}
        for i,item in enumerate(items):
            fields['mask'+str(i)]=item['mask'];fields['cloud'+str(i)]=depth_cloud(f,item['mask'],cfg)
        np.savez_compressed('/tmp/shelf306_support_fixture.npz',**fields)
        cv2.imwrite('/tmp/shelf306_support_fixture.jpg',f['image'])
        print(json.dumps({'items':len(items),'names':[v['name'] for v in items],'up':up.tolist(),'T':T.tolist(),'q':q},ensure_ascii=False));break
finally:camera.close();robot.close();rclpy.shutdown()
