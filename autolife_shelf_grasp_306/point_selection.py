"""Local point-prompt segmentation and advisory naming; no robot/network I/O."""
import time
import cv2
import numpy as np

NAME_GROUPS=(
 'bottle. beverage can. cup. mug. bowl. plate. box. bag. tissue. towel. clothing. sponge.',
 'mobile phone. mouse. remote control. charger. cable. scissors. tape roll. pen. book. key. glasses.',
 'apple. banana. orange. toy. stuffed toy. clothes hanger. spoon. fork. kettle. detergent bottle. trash can.')

def point_pixel(x,y,shape):
    x,y=float(x),float(y);h,w=shape[:2]
    if not np.isfinite([x,y]).all() or not 0<=x<1 or not 0<=y<1:
        raise ValueError('请点在图像内部的物体上')
    return min(w-1,int(x*w)),min(h-1,int(y*h))

def choose_mask(masks,qualities,pixel,quality_min=.5):
    masks=np.asarray(masks,bool);quality=np.asarray(qualities,float).reshape(-1)
    if masks.ndim!=3 or len(masks)!=len(quality):raise ValueError('点选分割输出尺寸无效')
    x,y=pixel;candidates=[]
    for mask,q in zip(masks,quality):
        if not np.isfinite(q) or q<quality_min or not mask[y,x]:continue
        # Keep only the connected component containing the actual clicked pixel.
        _,components=cv2.connectedComponents(mask.astype(np.uint8),connectivity=8)
        component=components==components[y,x]
        if component.sum()<100 or component.mean()>.70:continue
        candidates.append((float(q),component))
    if not candidates:raise ValueError('此点没有可靠物体轮廓，请点物体中心、避开背景或重新取景')
    candidates.sort(key=lambda pair:pair[0],reverse=True)
    return candidates[0][1],candidates[0][0]

def visual_description(image,mask):
    yy,xx=np.nonzero(mask)
    if not len(xx):raise ValueError('空分割')
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)[mask]
    h,s,v=np.median(hsv,axis=0)
    color=('黑色' if v<50 else '白色' if s<35 and v>190 else '灰色' if s<45 else
           '红色' if h<10 or h>170 else '橙色' if h<22 else '黄色' if h<35 else
           '绿色' if h<85 else '蓝色' if h<135 else '紫色')
    ratio=(np.ptp(xx)+1)/(np.ptp(yy)+1)
    shape='长条状物体' if ratio>2.5 or ratio<.4 else '物体'
    return color+shape

def _name_candidates(segmenter,image,mask):
    """Use a bounded local vocabulary; names are suggestions, never grasp truth."""
    from .grounded_sam2 import select_boxes,ALIASES
    torch=segmenter.torch;rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    y,x=np.nonzero(mask);box=np.array([x.min(),y.min(),x.max()+1,y.max()+1]);area=np.prod(box[2:]-box[:2]);ranked=[]
    for text in NAME_GROUPS:
        inputs=segmenter.detector_processor(images=rgb,text=text,return_tensors='pt',
            size={'shortest_edge':640,'longest_edge':1066}).to(segmenter.device)
        with torch.inference_mode():outputs=segmenter.detector(**inputs)
        result=segmenter.detector_processor.post_process_grounded_object_detection(outputs,inputs.input_ids,
            threshold=.30,text_threshold=.20,target_sizes=[image.shape[:2]])[0]
        labels=result.get('text_labels',result.get('labels',[]))
        proposals=select_boxes(result['boxes'].detach().cpu().numpy(),result['scores'].detach().cpu().numpy(),labels,image.shape,threshold=.30,limit=20)
        for item in proposals:
            b=item['box'];inter=np.maximum(0,np.minimum(box[2:],b[2:])-np.maximum(box[:2],b[:2])).prod()
            iou=inter/max(1.,area+np.prod(b[2:]-b[:2])-inter)
            if iou>=.5:ranked.append((item['score']+.2*iou,item['label'],item['score']))
    ranked.sort(reverse=True)
    if not ranked:return '',0.
    # Prefer the shortest known Chinese noun for a returned English phrase.
    label=ranked[0][1].strip();cn=min((k for k,v in ALIASES.items() if v==label),key=len,default='')
    extra={'trash can':'垃圾桶','detergent bottle':'洗涤剂瓶','sponge':'海绵'}
    return (cn or extra.get(label,label)),ranked[0][2]

def segment(perception,frame,store,x,y,*,describe=True):
    from .core import crop_object,match
    segmenter=perception.segmenter
    if segmenter is None:raise ValueError('点选分割需要SAM2模型')
    image=frame['image'];pixel=point_pixel(x,y,image.shape);torch=segmenter.torch
    inputs=segmenter.sam_processor(images=cv2.cvtColor(image,cv2.COLOR_BGR2RGB),
        input_points=[[[list(pixel)]]],input_labels=[[[1]]],return_tensors='pt').to(segmenter.device)
    with torch.inference_mode():outputs=segmenter.sam(**inputs,multimask_output=True)
    masks=segmenter.sam_processor.post_process_masks(outputs.pred_masks.detach().cpu(),inputs['original_sizes'])[0]
    masks=masks.numpy().reshape(-1,*image.shape[:2])
    mask,quality=choose_mask(masks,outputs.iou_scores.detach().cpu().numpy(),pixel,float(perception.cfg.get('sam2_mask_quality',.5)))
    crop=crop_object(image,mask);feature=perception.embed([crop])[0]
    m=match(feature,store.products,perception.cfg['match_threshold'],perception.cfg['match_margin'])
    product=next((p for p in store.products if p['id']==m['id']),None)
    description=visual_description(image,mask);suggestion='';score=0.;naming_error=''
    if product:suggestion=product['name'];score=m['score']
    elif describe:
        try:suggestion,score=_name_candidates(segmenter,image,mask)
        except Exception as exc:naming_error=str(exc)[:180]  # segmentation remains usable without naming
    name=product['name'] if product else ('疑似'+suggestion+'（待确认）' if suggestion else description+'（名称待确认）')
    yy,xx=np.nonzero(mask);perception.counter+=1
    return dict(id=perception.counter,box=[float(xx.min()),float(yy.min()),float(xx.max()+1),float(yy.max()+1)],
        mask=mask,crop=crop,feature=feature,product=product,match=m,name=name,
        semantic_label='',source='point_sam2',point_selection=True,point_xy=[float(x),float(y)],
        detection_score=quality,mask_score=quality,suggested_name=suggestion or description,
        visual_description=description,naming_score=float(score),naming_error=naming_error)

def reobserve(perception,frame,store,chosen):
    """Fresh point mask, then the existing mask/appearance identity gate.

    The old image is never reused as grasp geometry. If the object moved away
    or a different object occupies the clicked point, identity must fail closed.
    """
    mask=np.asarray(chosen['mask'],np.uint8)
    dist=cv2.distanceTransform(mask,cv2.DIST_L2,5);y,x=np.unravel_index(np.argmax(dist),dist.shape)
    item=segment(perception,frame,store,(x+.5)/mask.shape[1],(y+.5)/mask.shape[0],describe=False)
    item.update(name=chosen['name'],suggested_name=chosen.get('suggested_name',''),
                visual_description=chosen.get('visual_description',''))
    return [item]
