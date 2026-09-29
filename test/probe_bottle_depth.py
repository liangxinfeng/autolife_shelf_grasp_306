"""Five independent live bottle masks/depth checks, with no robot commands."""
import json
import time
from pathlib import Path
import numpy as np
import yaml
import rclpy
from autolife_shelf_grasp_306.camera import Camera
from autolife_shelf_grasp_306.robot import Robot
from autolife_shelf_grasp_306.perception import Perception
from autolife_shelf_grasp_306.core import ProductStore,depth_cloud
from autolife_handeye_calibration_307.core import Kinematics
from autolife_handeye_calibration_307.verification_core import camera_transform

root=Path('/home/ubuntu/ros2_ws/src/autolife_shelf_grasp_306');cfg=yaml.safe_load((root/'config/grasp.yaml').read_text())
data=json.loads(Path(cfg['calibration']).read_text());fk=Kinematics(cfg['urdf_path'])
rclpy.init(args=[]);robot=Robot();camera=Camera(cfg,'/tmp/shelf306_depth_probe.jpg')
try:
    model=Perception(cfg,root);store=ProductStore(root/'data');samples=[];seen=set();deadline=time.monotonic()+8
    while len(samples)<5 and time.monotonic()<deadline:
        f,error=camera.get()
        if f is None or f.get('depth') is None or f['frame_id'] in seen:time.sleep(.02);continue
        items,_=model.infer(f,store)
        if not items:continue
        item=max(items,key=lambda v:np.sum(v['mask']));cloud=depth_cloud(f,item['mask'],cfg)
        q=robot.sample(f['received'],stationary=True);T=camera_transform(data,fk,q,f)
        points=cloud@T[:3,:3].T+T[:3,3]
        samples.append({'frame_id':f['frame_id'],'pixels':int(item['mask'].sum()),'depth_points':len(cloud),
                        'median_depth_m':float(np.median(cloud[:,2])),
                        'median_surface_torso_m':np.median(points,axis=0).tolist(),
                        'z_5_95_m':np.percentile(points[:,2],[5,95]).tolist()})
        seen.add(f['frame_id'])
    if len(samples)<5:raise ValueError('没有取得5组有效饮料深度')
    centers=np.array([v['median_surface_torso_m'] for v in samples])
    result={'samples':samples,'surface_spread_mm':float(np.max(np.linalg.norm(centers-np.median(centers,axis=0),axis=1))*1000),
            'note':'可见表面，不是瓶子中心；未校验真实坐标精度；无任何运动'}
    Path('/tmp/shelf306_bottle_depth.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
finally:camera.close();robot.close();rclpy.shutdown()
