"""Live segmentation/depth statistics only; does not enroll or move hardware."""
import json,time,tempfile,os
from pathlib import Path
import cv2,numpy as np,yaml,rclpy
from autolife_shelf_grasp_306.camera import Camera
from autolife_shelf_grasp_306.robot import Robot
from autolife_shelf_grasp_306.perception import Perception
from autolife_shelf_grasp_306.core import ProductStore
from autolife_shelf_grasp_306.bottle import object_points,horizontal_basis
from autolife_handeye_calibration_306.core import Kinematics,TORSO
from autolife_handeye_calibration_306.verification_core import camera_transform

os.environ['ROS_DOMAIN_ID']='0';os.environ['RMW_IMPLEMENTATION']='rmw_cyclonedds_cpp'
os.environ['CYCLONEDDS_URI']='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
root=Path('/home/ubuntu/ros2_ws/src/autolife_shelf_grasp_306')
cfg=yaml.safe_load((root/'config/grasp.yaml').read_text())
cfg['generic_weights']='/tmp/shelf300_FastSAM-s.pt'
cfg['classes']=[24,26,39,40,41,46,47,48,49,50,51,52,53,54,55]
rclpy.init(args=[]);camera=Camera(cfg,'/tmp/shelf300_multi_preview.jpg');robot=Robot()
try:
    model=Perception(cfg,root);store=ProductStore(root/'data');fk=Kinematics(cfg['urdf_path'])
    calibration=json.loads(Path(cfg['calibration']).read_text())
    for iteration in range(3):
        frame,_=camera.get();items,elapsed=model.infer(frame,store)
        if iteration<2:print('INFERENCE',elapsed,'candidates',len(items));continue
        q=robot.sample(frame['received']);T=camera_transform(calibration,fk,q,frame)
        up=fk.fk('Link_Zero_Point',TORSO,q)[:3,:3].T@np.array([0.,0.,1.]);u,v,n=horizontal_basis(up)
        image=frame['image'].copy();records=[]
        for item in items:
            a,b,c,d=map(int,item['box']);cv2.rectangle(image,(a,b),(c,d),(0,220,100),2)
            cv2.putText(image,str(item['id']),(a,b-6),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,220,100),2)
            row={'id':item['id'],'box':item['box'],'source':item['source'],'matched_name':item['name']}
            try:
                points,pixels,_=object_points(frame,item['mask'],T,cfg)
                row.update(depth_points=len(points),visible_height_mm=float(np.ptp(np.percentile(points@n,[1,99]))*1000))
            except ValueError as exc:row['depth_error']=str(exc)
            records.append(row)
        cv2.imwrite('/tmp/shelf300_multi_candidates.jpg',image)
        Path('/tmp/shelf300_multi_candidates.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
        print('INFERENCE',elapsed,json.dumps(records,ensure_ascii=False))
finally:camera.close();robot.close();rclpy.shutdown()
