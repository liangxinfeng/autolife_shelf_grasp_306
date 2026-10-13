"""Read-only camera. Capacity-one RGB-D; display is independent of inference."""
import importlib.util
import threading
import time
import os
from pathlib import Path
import cv2
import numpy as np
from autolife_handeye_calibration_306.depth_target import aligned_depth
from .live_overlay import LiveOverlay,encode_packet


class Camera:
    def __init__(self, cfg, preview):
        self.cfg=cfg; self.preview=Path(preview); self.latest=None; self.error='等待相机'
        self.overlay=LiveOverlay()
        self.lock=threading.Lock(); self.stop=threading.Event()
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()

    def get(self):
        with self.lock:return self.latest,self.error

    def run(self):
        consumers=[]
        try:
            source=next((Path(self.cfg['vendor_python_path'])/'autolife_robot_sdk/utils').glob('camera_shm*.so'))
            spec=importlib.util.spec_from_file_location('camera_shm',source)
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            for suffix in ('rgbd_head_color','rgbd_head_depth'):
                opts={key:f'/{prefix}_{suffix}' for key,prefix in [
                    ('data_shm_name','camera_image_buffer'),('meta_shm_name','camera_metadata_struct'),
                    ('read_mutex_name','read_mutex_lock'),('write_lock_name','write_data_lock'),
                    ('writer_queue_name','writer_queue_sem'),('intrinsics_shm_name','camera_intrinsics_struct')]}
                consumers.append(mod.SHMCameraFrameConsumer({'selected_protocol':'v1','protocols':{'v1':opts}},name='shelf_readonly'))
            last=None;intr=None;last_intr=0.;last_preview=0.;cached_depth=None
            while not self.stop.is_set():
                try:
                    now=time.monotonic()
                    if now-last_intr>2:
                        intr=dict(consumers[0].get_intrinsics());di=dict(consumers[1].get_intrinsics());last_intr=now
                    # Read both continuously; never accept mismatched frames for geometry.
                    color=consumers[0].get_latest(nonblock=True)
                    pair=consumers[1].get_latest(nonblock=True)
                    if pair is not None:cached_depth=(np.array(pair[0],copy=True),pair[1])
                    if color is None or color[1]==last:self.stop.wait(.008);continue
                    image=np.array(color[0],copy=True); ident=int(color[1]);last=ident
                    frame={'image':image,'received':now,'frame_id':ident,'intrinsics':intr,
                           'K':np.array([[intr['fx'],0,intr['ppx']],[0,intr['fy'],intr['ppy']],[0,0,1.]]),
                           'D':np.array(intr['coeffs'],float),'depth_error':'等待同帧深度'}
                    if cached_depth is not None:
                        try:
                            aligned_depth(image,cached_depth[0],ident,cached_depth[1],intr,di)
                            frame.update(depth=cached_depth[0],depth_id=int(cached_depth[1]),depth_intrinsics=di,depth_error='')
                        except ValueError as exc:frame['depth_error']=str(exc)
                    with self.lock:self.latest=frame;self.error=''
                    if now-last_preview>=.08:
                        # BGR ABI is the same as the existing calibration camera UI.
                        small=cv2.resize(image,(960,round(image.shape[0]*960/image.shape[1])))
                        ok,data=cv2.imencode('.jpg',small,[cv2.IMWRITE_JPEG_QUALITY,85])
                        if ok:
                            temp=self.preview.with_suffix('.tmp');temp.write_bytes(data.tobytes());os.replace(temp,self.preview)
                            # Display failures must never invalidate RGB-D used by
                            # planning. The packet binds labels to exactly this RGB.
                            try:meta=self.overlay.render(frame)
                            except Exception:
                                meta=dict(revision=None,items=[],path=[],path_revision=None,
                                          shape=list(image.shape),frame_id=ident,received=now,lost=1,path_status='unavailable')
                            packet=self.preview.with_suffix('.npz');tmp=packet.with_suffix('.npz.tmp')
                            tmp.write_bytes(encode_packet(data,meta));os.replace(tmp,packet)
                        last_preview=now
                    self.stop.wait(.008)
                except Exception as exc:
                    with self.lock:self.error=str(exc);self.latest=None
                    self.stop.wait(.2)
        except Exception as exc:self.error='相机不可用：'+str(exc)
        finally:
            for c in consumers:
                try:c.close()
                except Exception:pass

    def close(self):self.stop.set();self.thread.join(2.)
