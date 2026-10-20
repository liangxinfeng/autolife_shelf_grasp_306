"""Instance segmentation + unchanged DINOv2 SKU retrieval; models loaded once."""
import time
from pathlib import Path
import cv2
import numpy as np
from .core import crop_object, match,product_category


def model_paths(cfg, root):
    """Resolve every local asset against the package, never the shell cwd."""
    root = Path(root).expanduser().resolve()
    defaults = {'generic_weights': 'models/FastSAM-s.pt',
                'yoloe_text_encoder': 'models/mobileclip_blt.ts'}
    grounded=cfg.get('detector_backend')=='grounded_sam2'
    keys = ['dino_repo','dino_weights'] if grounded else ['yolo_weights', 'dino_repo', 'dino_weights', 'generic_weights']
    if not grounded and (cfg.get('detector_backend') == 'yoloe' or 'yoloe' in str(cfg.get('yolo_weights', ''))):
        keys.append('yoloe_text_encoder')
    paths = {}
    missing = []
    for key in keys:
        value = str(cfg.get(key, defaults.get(key, ''))).strip()
        if not value:
            missing.append(f'{key}: 未配置')
            continue
        p = Path(value).expanduser()
        p = (p if p.is_absolute() else root / p).resolve()
        check = p / 'hubconf.py' if key == 'dino_repo' else p
        if not check.is_file() or check.stat().st_size == 0:
            missing.append(f'{key}: {check}')
        paths[key] = p
    if missing:
        raise ValueError('模型文件缺失或为空：\n' + '\n'.join(missing))
    if grounded:
        from .grounded_sam2 import local_model_paths
        paths.update(local_model_paths(cfg,root))
    return paths


def duplicate_mask(mask,box,previous):
    """Do not erase a small object merely because a large support mask covers it."""
    a=np.asarray(box)
    for item in previous:
        b=np.asarray(item['box'])
        inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
        iou=inter/max(1.,np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter)
        overlap=np.logical_and(mask,item['mask']).sum()/max(1,min(mask.sum(),item['mask'].sum()))
        if iou>.45 and overlap>.80:return True
    return False


class Perception:
    def __init__(self,cfg,root):
        import torch
        self.torch=torch;self.cfg=cfg
        self.device=cfg['device'];torch.set_num_threads(2)
        if self.device.startswith('cuda') and not torch.cuda.is_available():raise ValueError('CUDA不可用')
        assets=model_paths(cfg,root)
        self.segmenter=None
        if cfg.get('detector_backend')=='grounded_sam2':
            from .grounded_sam2 import GroundedSAM2
            self.segmenter=GroundedSAM2(cfg,root)
        else:self._load_legacy(assets)
        repo=assets['dino_repo'];weights=assets['dino_weights']
        self.dino=torch.hub.load(str(repo),'dinov2_vits14',source='local',pretrained=False)
        self.dino.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
        self.dino=self.dino.eval().to(self.device)
        self.mean=torch.tensor([.485,.456,.406],device=self.device)[None,:,None,None]
        self.std=torch.tensor([.229,.224,.225],device=self.device)[None,:,None,None]
        self.embed([np.full((224,224,3),127,np.uint8)])
        self.previous=[];self.counter=0

    def _load_legacy(self,assets):
        """Explicit rollback backend only; not loaded by Grounded SAM 2."""
        from ultralytics import YOLO,FastSAM
        torch=self.torch;cfg=self.cfg;yolo=assets['yolo_weights']
        self.yolo=YOLO(str(yolo))
        self.mode_classes={'beverages':[39,40,41],'all':cfg['classes']}
        if 'yoloe_text_encoder' in assets:
            from ultralytics.nn.text_model import MobileCLIPTS
            configured=cfg.get('yoloe_prompts',{})
            by_mode={m:list(dict.fromkeys(str(x).strip() for x in configured.get(m,[]) if str(x).strip()))
                     for m in ('beverages','all')}
            if not all(by_mode.values()):
                raise ValueError('YOLOE缺少beverages/all识别提示词配置')
            names=list(dict.fromkeys(by_mode['all']+by_mode['beverages']))
            self.yolo.to(self.device)
            self.yolo.model.clip_model=MobileCLIPTS(torch.device(self.device),weight=str(assets['yoloe_text_encoder']))
            embeddings=self.yolo.model.get_text_pe(names,cache_clip_model=True)
            self.yolo.set_classes(names,embeddings)
            del self.yolo.model.clip_model
            self.mode_classes={m:[names.index(x) for x in by_mode[m]] for m in by_mode}
        self.yolo.predict(np.zeros((640,640,3),np.uint8),imgsz=cfg['image_size'],device=self.device,verbose=False)
        generic=assets['generic_weights']
        self.generic=FastSAM(str(generic))
        self.generic.predict(np.zeros((640,640,3),np.uint8),imgsz=cfg['image_size'],device=self.device,verbose=False)

    def embed(self,crops):
        torch=self.torch
        # Match enrollment and queries identically: padded object, no aspect distortion.
        batch=[]
        for crop in crops:
            h,w=crop.shape[:2];side=max(h,w);canvas=np.full((side,side,3),127,np.uint8)
            canvas[(side-h)//2:(side-h)//2+h,(side-w)//2:(side-w)//2+w]=crop
            batch.append(cv2.resize(canvas,(224,224))[:,:,::-1].copy())
        x=torch.from_numpy(np.stack(batch)).permute(0,3,1,2).float().to(self.device)/255.
        with torch.inference_mode():features=self.dino((x-self.mean)/self.std)
        features=torch.nn.functional.normalize(features,dim=1)
        return features.cpu().numpy()

    def _legacy_items(self,frame,mode):
        image=frame['image']
        result=self.yolo.predict(image,imgsz=self.cfg['image_size'],conf=self.cfg['confidence'],
            classes=self.mode_classes[mode],device=self.device,verbose=False,retina_masks=True,max_det=25)[0]
        items=[]
        outputs=[(result,'yolo')]
        if mode=='all':
            generic=self.generic.predict(image,imgsz=self.cfg['image_size'],conf=.60,
                device=self.device,verbose=False,retina_masks=True,max_det=40)[0]
            outputs.append((generic,'generic'))
        # Modes limit segmentation proposals only. Names, geometry and samples
        # always come from the same catalog, regardless of enrollment mode.
        for output,source in outputs:
            if len(items)>=12:break
            if output.masks is None:continue
            for mask,box,score in zip(output.masks.data.cpu().numpy(),output.boxes.xyxy.cpu().numpy(),output.boxes.conf.cpu().numpy()):
                mask=cv2.resize(mask,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)>.5
                if mask.sum()<100:continue
                if source=='generic':
                    h,w=image.shape[:2];x0,y0,x1,y1=box
                    if min(x0,y0,w-x1,h-y1)<3 or mask.mean()>.18:continue
                    # Suppress nested cap/label fragments and duplicate YOLO masks.
                    if duplicate_mask(mask,box,items):continue
                    depth=frame.get('depth')
                    if depth is not None:
                        z=depth[mask].astype(float)*self.cfg['depth_scale_m'];z=z[(z>.2)&(z<1.4)]
                        if len(z)<80:continue
                        width=(x1-x0)*np.median(z)/frame['K'][0,0]
                        if not .015<width<.4:continue
                crop=crop_object(image,mask)
                items.append({'box':box.tolist(),'mask':mask,'crop':crop,'detection_score':float(score),'source':source})
                if len(items)>=12:break
        return items

    def infer(self,frame,store,mode='all',prompt=None):
        if mode not in ('beverages','all'):raise ValueError('识别模式无效')
        start=time.monotonic()
        segmenter=getattr(self,'segmenter',None)
        if prompt is not None and segmenter is None:raise ValueError('任务按名称识别需要文字分割模型')
        items=(segmenter.proposals(frame,mode,prompt=prompt) if prompt is not None else segmenter.proposals(frame,mode)) if segmenter is not None else self._legacy_items(frame,mode)
        products=store.products
        if items:
            features=self.embed([v['crop'] for v in items]);used=set()
            for item,feature in zip(items,features):
                candidate=None;best=.4
                a=np.array(item['box'])
                for previous in self.previous:
                    if previous['id'] in used:continue
                    b=np.array(previous['box']);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
                    iou=inter/max(1.,np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter)
                    if iou>best:best=iou;candidate=previous['id']
                if candidate is None:self.counter+=1;candidate=self.counter
                used.add(candidate)
                m=match(feature,products,self.cfg['match_threshold'],self.cfg['match_margin'])
                p=next((v for v in store.products if v['id']==m['id']),None)
                item.update(id=candidate,feature=feature,product=p,match=m,
                            name=p['name'] if p else ('未录入 · '+item['semantic_label'] if item.get('semantic_label') else '未录入 / 待确认'))
        self.previous=[{'id':v['id'],'box':v['box']} for v in items]
        return items,round((time.monotonic()-start)*1000,1)
