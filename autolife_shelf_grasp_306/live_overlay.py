"""Display-only, bounded optical-flow overlays. Never a source of grasp geometry.

RGB and overlays are published in one atomic packet. Masks are propagated from
the last SAM segmentation, not represented as fresh semantic inference. Lost
tracks are hidden, and 3-D path projection fails closed on stale robot feedback.
"""
import copy
import io
import json
import threading
import time
import cv2
import numpy as np


def encode_packet(jpeg,meta):
    buffer=io.BytesIO()
    np.savez(buffer,jpeg=np.asarray(jpeg,dtype=np.uint8).ravel(),meta=json.dumps(meta,ensure_ascii=False))
    return buffer.getvalue()


def decode_packet(data):
    with np.load(io.BytesIO(data),allow_pickle=False) as packet:
        return packet['jpeg'].tobytes(),json.loads(str(packet['meta']))


def gray_image(image,width=640):
    scale=min(1.,width/image.shape[1])
    small=cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)))
    return cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),scale


def transform_points(points,matrix):
    p=np.asarray(points,float).reshape(-1,2)
    return (p@matrix[:2,:2].T+matrix[:2,2]).tolist()


class LiveOverlay:
    def __init__(self):
        self.lock=threading.Lock();self.pending=None;self.renames=None;self.revision=0
        self.tracks=[];self.previous=None;self.scale=1.;self.projector=None;self.previous_stamp=None

    def set(self,frame,items,revision):
        # The inference frame is immutable. This slot replaces older requests.
        with self.lock:
            self.pending=(frame['image'],copy.deepcopy(items),revision)
            self.projector=None;self.renames=None

    def rename(self,items,revision):
        with self.lock:self.renames=(copy.deepcopy(items),revision)

    def set_path(self,projector,revision):
        with self.lock:self.projector=(projector,revision)

    def _features(self,gray,item):
        mask=np.zeros(gray.shape,np.uint8)
        polygons=item.get('mask_contours',[])
        if polygons:cv2.fillPoly(mask,[np.rint(np.asarray(p)*self.scale).astype(np.int32) for p in polygons],255)
        else:
            x0,y0,x1,y1=np.rint(np.asarray(item['box'])*self.scale).astype(int)
            cv2.rectangle(mask,(x0,y0),(x1,y1),255,-1)
        return cv2.goodFeaturesToTrack(gray,maxCorners=48,qualityLevel=.008,minDistance=4,mask=mask,blockSize=3)

    def render(self,frame):
        gray,scale=gray_image(frame['image'])
        with self.lock:
            pending=self.pending;self.pending=None;renames=self.renames;self.renames=None;projector=self.projector
        if pending is not None:
            source,items,self.revision=pending;self.previous,self.scale=gray_image(source);self.tracks=[]
            for item in items:
                self.tracks.append(dict(item=item,matrix=np.eye(3),points=self._features(self.previous,item),lost=False))
            self.previous_stamp=None
        if renames is not None:
            items,self.revision=renames;lookup={v['id']:v for v in items}
            for t in self.tracks:
                if t['item']['id'] in lookup:t['item']['name']=lookup[t['item']['id']]['name']
        incompatible=self.previous is not None and (self.previous.shape!=gray.shape or self.scale!=scale)
        gap=self.previous_stamp is not None and frame['received']-self.previous_stamp>.6
        items=[];lost=0
        for track in self.tracks:
            if incompatible or gap:track['lost']=True
            if not track['lost'] and self.previous is not None:
                pts=track['points']
                if pts is None or len(pts)<4:track['lost']=True
                else:
                    nxt,status,_=cv2.calcOpticalFlowPyrLK(self.previous,gray,pts,None,winSize=(21,21),maxLevel=3)
                    if nxt is None or status is None:track['lost']=True
                    else:
                        back,valid,_=cv2.calcOpticalFlowPyrLK(gray,self.previous,nxt,None,winSize=(21,21),maxLevel=3)
                        good=(np.zeros(len(pts),dtype=bool) if back is None or valid is None else
                              (status.ravel()>0)&(valid.ravel()>0)&(np.linalg.norm(back-pts,axis=2).ravel()<1.5))
                        before=pts[good].reshape(-1,2);after=nxt[good].reshape(-1,2)
                        if len(before)<4:track['lost']=True
                        else:
                            affine,inliers=cv2.estimateAffinePartial2D(before,after,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=150)
                            if affine is None or inliers is None or inliers.sum()<4 or inliers.mean()<.6:track['lost']=True
                            else:
                                size=np.linalg.norm(affine[:,0])
                                residual=np.median(np.linalg.norm(np.asarray(transform_points(before,np.vstack([affine,[0,0,1]])))-after,axis=1))
                                if not .8<size<1.25 or residual>2.:track['lost']=True
                                else:
                                    native=np.vstack([affine,[0,0,1]]);native[:2,2]/=scale
                                    track['matrix']=native@track['matrix'];track['points']=nxt[good][inliers.ravel()>0]
            if track['lost']:lost+=1;continue
            item=copy.deepcopy(track['item']);matrix=track['matrix']
            item['mask_contours']=[transform_points(p,matrix) for p in item.get('mask_contours',[])]
            x0,y0,x1,y1=item['box'];corners=np.asarray(transform_points([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],matrix))
            item['box']=[*corners.min(axis=0),*corners.max(axis=0)]
            item['display_to_reference']=np.linalg.inv(matrix).tolist();item['display_tracking']=True
            items.append(item)
            # Replenish points on the propagated object, not on a new identity.
            if track['points'] is not None and len(track['points'])<16:
                more=self._features(gray,item)
                if more is not None and len(more)>=4:track['points']=more
        self.previous=gray;self.previous_stamp=frame['received'];self.scale=scale
        path=[];path_status='none'
        if projector is not None:
            try:path=projector[0](frame);path_status='projected'
            except (ValueError,KeyError,TypeError):path_status='unavailable'
        return dict(revision=self.revision,path_revision=projector[1] if projector else None,
                    items=items,path=path,path_status=path_status,lost=lost,
                    shape=list(frame['image'].shape),frame_id=frame['frame_id'],received=frame['received'])
