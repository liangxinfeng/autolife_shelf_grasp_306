"""Run explicitly for installation verification. Never imports execution/bridge."""
import json
import time
from pathlib import Path
import tempfile
import yaml
import rclpy
import numpy as np
from autolife_shelf_grasp_306.camera import Camera
from autolife_shelf_grasp_306.robot import Robot
from autolife_shelf_grasp_306.perception import Perception
from autolife_shelf_grasp_306.core import ProductStore,match
from autolife_shelf_grasp_306.planning import Planner
from autolife_handeye_calibration_306.verification_core import camera_transform,load_geometry

root=Path('/home/ubuntu/ros2_ws/src/autolife_shelf_grasp_306')
cfg=yaml.safe_load((root/'config/grasp.yaml').read_text());rclpy.init(args=[])
camera=Camera(cfg,'/tmp/shelf300_probe.jpg');robot=Robot();result={}
try:
    seen=set();paired=set();until=time.monotonic()+5
    while time.monotonic()<until:
        frame,error=camera.get()
        if frame:
            seen.add(frame['frame_id'])
            if frame.get('depth') is not None:paired.add(frame['frame_id'])
        time.sleep(.01)
    q=robot.sample(stationary=True)
    result.update(rgb_frames=len(seen),paired_frames=len(paired),joints=len(q),robot_error=robot.last_error,camera_error=error)
    begin=time.monotonic();model=Perception(cfg,root);result['model_load_s']=round(time.monotonic()-begin,2)
    elapsed=[];counts=[];last=None
    for _ in range(5):
        frame,error=camera.get()
        items,ms=model.infer(frame,ProductStore(root/'data'));elapsed.append(ms);counts.append(len(items));last=items
    result.update(inference_ms=elapsed,detected_counts=counts,shape=list(frame['image'].shape))
    feature=model.embed([frame['image']])[0]
    result['embedding_dimension']=len(feature)
    result['self_retrieval_test']=match(feature,[{'id':'same_image','features':[feature.tolist()]}])['id']=='same_image'
    planner=Planner(cfg);result['collision_objects']=len(planner.guard.kin.geom_model.geometryObjects)
    data=json.loads(Path(cfg['calibration']).read_text());frame,_=camera.get();q=robot.sample(frame['received'])
    T=camera_transform(data,planner.fk,q,frame);tcp=load_geometry(data,planner.fk,cfg['tcp_offset_mm'])
    result['T_torso_camera']=T.tolist();result['tcp_m']=tcp.tolist()
    result['hardware_publishers_created']=False
    Path('/tmp/shelf300_probe_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
finally:camera.close();robot.close();rclpy.shutdown()
