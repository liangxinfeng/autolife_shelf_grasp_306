"""One-shot class-unfiltered detection on current RGB; no motion node."""
import time,json
from pathlib import Path
import yaml,cv2
from ultralytics import YOLO,FastSAM
from autolife_shelf_grasp_306.camera import Camera

root=Path('/home/ubuntu/ros2_ws/src/autolife_shelf_grasp_306')
cfg=yaml.safe_load((root/'config/grasp.yaml').read_text())
camera=Camera(cfg,'/tmp/shelf306_objects_preview.jpg')
try:
    deadline=time.monotonic()+5;frame=None
    while frame is None and time.monotonic()<deadline:frame,_=camera.get();time.sleep(.05)
    if frame is None:raise RuntimeError('camera unavailable')
    model=YOLO(cfg['yolo_weights']);result=model.predict(frame['image'],conf=.20,imgsz=640,device=cfg['device'],verbose=False)[0]
    print(json.dumps([{'class':model.names[int(c)],'score':round(float(s),3),'box':b.tolist()}
        for c,s,b in zip(result.boxes.cls,result.boxes.conf,result.boxes.xyxy)],ensure_ascii=False))
    cv2.imwrite('/tmp/shelf306_objects_detection.jpg',result.plot())
    weights=Path('/tmp/shelf306_FastSAM-s.pt')
    if weights.is_file():
        fast=FastSAM(str(weights));start=time.monotonic()
        result=fast.predict(frame['image'],conf=.35,imgsz=640,retina_masks=True,device=cfg['device'],verbose=False)[0]
        print('FastSAM',time.monotonic()-start,[(round(float(s),3),[int(v) for v in b]) for s,b in zip(result.boxes.conf,result.boxes.xyxy)])
        cv2.imwrite('/tmp/shelf306_objects_fastsam.jpg',result.plot())
finally:camera.close()
