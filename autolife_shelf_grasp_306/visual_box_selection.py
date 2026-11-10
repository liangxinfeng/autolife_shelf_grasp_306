"""矩形提示只限定目标，由SAM2分割真实轮廓；不把框内背景当成物体。"""
import numpy as np
import cv2


def pixel_box(value,shape):
    a=np.asarray(value,float);h,w=shape[:2]
    if a.shape!=(4,) or not np.isfinite(a).all() or np.any(a<0) or np.any(a>1):raise ValueError('请在图片内拖动矩形框')
    x0,x1=sorted([a[0]*w,a[2]*w]);y0,y1=sorted([a[1]*h,a[3]*h])
    if x1-x0<8 or y1-y0<8:raise ValueError('框太小，请框住整个物体后松开鼠标')
    return np.array([x0,y0,x1,y1],np.float32)


def dominant_component(mask):
    """主体占绝大多数时剔除离散背景，不把笔/桌面碎片算进物体包围框。"""
    mask=np.asarray(mask,bool)
    num,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    if num>2:
        areas=stats[1:,cv2.CC_STAT_AREA];largest=1+int(np.argmax(areas))
        if areas.max()>=.80*areas.sum():return labels==largest
    return mask


def rank_box_masks(masks,qualities,box,quality_min=.5):
    masks=np.asarray(masks,bool);qualities=np.asarray(qualities,float).reshape(-1)
    if masks.ndim!=3 or len(masks)!=len(qualities):raise ValueError('框选分割输出无效')
    x0,y0,x1,y1=box;area=max(1.,(x1-x0)*(y1-y0));candidates=[]
    for mask,q in zip(masks,qualities):
        mask=dominant_component(mask)
        if not np.isfinite(q) or q<quality_min or mask.sum()<100:continue
        yy,xx=np.nonzero(mask);inside=(xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)
        if inside.mean()<.8:continue
        b=np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1]);inter=np.maximum(0,np.minimum(b[2:],box[2:])-np.maximum(b[:2],box[:2])).prod()
        iou=float(inter/max(1.,area+np.prod(b[2:]-b[:2])-inter))
        candidates.append((.55*float(q)+.45*iou,mask,float(q)))
    if not candidates:raise ValueError('框内未分出可靠物体，请重新框住一个完整物体，减少桌面和其他物品')
    return sorted(candidates,key=lambda v:v[0],reverse=True)


def choose_box_mask(masks,qualities,box,quality_min=.5):
    _,mask,q=rank_box_masks(masks,qualities,box,quality_min)[0]
    return mask,q


def segment_box(perception,frame,value):
    s=perception.segmenter
    if s is None:raise ValueError('框选物体需要SAM2视觉后台就绪')
    image=frame['image'];box=pixel_box(value,image.shape)
    inputs=s.sam_processor(images=cv2.cvtColor(image,cv2.COLOR_BGR2RGB),input_boxes=[[box.tolist()]],return_tensors='pt').to(s.device)
    with s.torch.inference_mode():out=s.sam(**inputs,multimask_output=True)
    masks=s.sam_processor.post_process_masks(out.pred_masks.detach().cpu(),inputs['original_sizes'])[0].numpy().reshape(-1,*image.shape[:2])
    ranked=rank_box_masks(masks,out.iou_scores.detach().cpu().numpy(),box,float(perception.cfg.get('sam2_mask_quality',.5)))
    items=[]
    for _,mask,score in ranked:
        yy,xx=np.nonzero(mask)
        items.append(dict(id=0,name='框选物体',mask=mask,box=[float(xx.min()),float(yy.min()),float(xx.max()+1),float(yy.max()+1)],mask_score=score,source='box_sam2'))
    # 保存页仍展示第一候选供确认；自动纠偏利用其他候选做三维验证，避免框占满者优先。
    result=dict(items[0]);result['_alternatives']=items[1:]
    return result



def projected_box(cloud,delta,camera_pose,frame):
    """将已匹配的整物体表面投影成框，避免复识别又退化为只分割标签点。"""
    points=np.asarray(cloud,float).reshape(-1,3)
    world=points@delta[:3,:3].T+delta[:3,3];inv=np.linalg.inv(camera_pose)
    camera=world@inv[:3,:3].T+inv[:3,3];good=np.isfinite(camera).all(1)&(camera[:,2]>.05)
    if good.sum()<8:raise ValueError('匹配物体未完整进入当前相机视野，请重新取景')
    pixels=cv2.projectPoints(camera[good],np.zeros(3),np.zeros(3),frame['K'],frame['D'])[0].reshape(-1,2)
    h,w=frame['image'].shape[:2];lo=np.percentile(pixels,.5,axis=0)-4;hi=np.percentile(pixels,99.5,axis=0)+4
    result=[max(0.,lo[0]/w),max(0.,lo[1]/h),min(1.,hi[0]/w),min(1.,hi[1]/h)]
    pixel_box(result,frame['image'].shape)
    return result
