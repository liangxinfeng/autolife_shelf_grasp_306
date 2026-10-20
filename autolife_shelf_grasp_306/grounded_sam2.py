"""Local text grounding + SAM 2.1 instance masks. No robot or network calls."""
import re
import time
from pathlib import Path
import cv2
import numpy as np

ALIASES={'瓶子':'bottle','饮料':'beverage bottle','饮料瓶':'beverage bottle','水瓶':'water bottle',
         '矿泉水瓶':'water bottle','塑料瓶':'plastic bottle','杯子':'cup','纸杯':'paper cup',
         '罐子':'can','易拉罐':'beverage can','口香糖':'chewing gum container','口香糖罐':'chewing gum container',
         '西瓜':'watermelon','西瓜模型':'watermelon shaped object','袋子':'bag','纸袋':'paper bag',
         '塑料袋':'plastic bag','购物袋':'shopping bag','盒子':'box','纸盒':'cardboard box',
         '桌子':'table','桌面':'tabletop','椅子':'chair','毛巾':'towel','衣服':'clothing','手机':'mobile phone',
         '鼠标':'mouse','电脑鼠标':'mouse','无线鼠标':'wireless mouse','有线鼠标':'wired mouse',
         '键盘':'keyboard','电脑':'computer','笔记本电脑':'laptop','笔记本':'notebook','显示器':'monitor',
         '遥控器':'remote control','耳机':'headphones','音箱':'speaker','音响':'speaker','充电器':'charger',
         '充电宝':'power bank','数据线':'cable','剪刀':'scissors','胶带':'tape roll','胶带卷':'tape roll',
         '笔':'pen','圆珠笔':'ballpoint pen','铅笔':'pencil','书':'book','书本':'book','本子':'notebook',
         '钥匙':'key','眼镜':'glasses','钱包':'wallet','雨伞':'umbrella','玩具':'toy','积木':'toy block',
         '保温杯':'thermos','马克杯':'mug','玻璃杯':'glass','碗':'bowl','盘子':'plate','勺子':'spoon',
         '叉子':'fork','筷子':'chopsticks','水壶':'kettle','茶壶':'teapot','瓶':'bottle','杯':'cup',
         '纸巾':'tissue','纸巾盒':'tissue box','纸箱':'cardboard box','塑料盒':'plastic box',
         '背包':'backpack','手提袋':'bag','购物包':'shopping bag','苹果':'apple','香蕉':'banana',
         '橙子':'orange','橘子':'mandarin orange','梨':'pear','桃子':'peach','水果':'fruit',
         '毛绒玩具':'stuffed toy','西瓜玩具':'watermelon shaped toy','衣架':'clothes hanger',
         '洗衣液':'laundry detergent bottle','洗衣液瓶':'laundry detergent bottle','洗发水':'shampoo bottle',
         '箱子':'box','包装盒':'packaged box','抽纸':'tissue pack','纸巾包':'tissue pack','矿泉水':'water bottle'}
COLORS={'红色':'red','绿色':'green','蓝色':'blue','黄色':'yellow','白色':'white','黑色':'black',
        '灰色':'gray','银色':'silver','金色':'gold','粉色':'pink','粉红色':'pink','紫色':'purple',
        '橙色':'orange','棕色':'brown','透明':'transparent',
        '红':'red','绿':'green','蓝':'blue','黄':'yellow','白':'white','黑':'black'}
MODIFIERS={**COLORS,'塑料':'plastic','金属':'metal','木制':'wooden','玻璃':'glass',
           '无线':'wireless','有线':'wired','小号':'small','大号':'large','小':'small','大':'large'}


def translate_phrase(phrase):
    """Deterministic local noun+attribute mapping; never drop unknown constraints."""
    if not re.search(r'[\u3400-\u9fff]',phrase):return phrase
    for prefix in ('请帮我识别一下','帮我识别一下','请识别一下','识别一下','请帮我识别','帮我识别','请识别','识别'):
        if phrase.startswith(prefix):phrase=phrase[len(prefix):].strip();break
    phrase=re.sub(r'\s+','',phrase)
    attributes=[];remaining=phrase
    while remaining:
        if remaining in ALIASES:return ' '.join(attributes+[ALIASES[remaining]])
        prefix=next((s for s in sorted(MODIFIERS,key=len,reverse=True) if remaining.startswith(s)),None)
        if prefix is None:break
        attributes.append(MODIFIERS[prefix]);remaining=remaining[len(prefix):].removeprefix('的')
    raise ValueError(f'未能完整理解中文描述“{phrase}”。请用常见物品名（如盒子、鼠标），颜色可选；也可用英文，不会忽略不认识的限定词')


def normalize_prompt(text):
    """Known Chinese nouns/colors are aliases, not an unrestricted translator."""
    text=str(text).strip()
    if len(text)>300:raise ValueError('描述过长，请用简短物品名称，多个物品用逗号分隔')
    phrases=[]
    for phrase in re.split(r'[.,;，；。、\n和与]+',text):
        phrase=phrase.strip().lower()
        if not phrase:continue
        phrase=translate_phrase(phrase)
        if not re.fullmatch(r"[a-z0-9][a-z0-9 '\-]*",phrase):raise ValueError('请用物品名称或英文短语，多个名称用逗号分隔')
        if phrase not in phrases:phrases.append(phrase)
    if len(phrases)>16:raise ValueError('一次最多指定16类物品，请缩小识别范围')
    return '. '.join(phrases)+('.' if phrases else '')


def prompt_for(cfg,mode):
    if mode not in ('beverages','all'):raise ValueError('识别模式无效')
    override=cfg.get('semantic_prompt_overrides',{}).get(mode,'')
    defaults=cfg.get('grounded_sam2_prompts',{})
    text=normalize_prompt(override or defaults.get(mode,''))
    if not text:raise ValueError('语义识别范围为空，请输入物品名称')
    return text


# Category expansion only for exact, unqualified nouns. Never turn "white box"
# into "box", which would silently remove the user's explicit color constraint.
NAME_VARIANTS={
    'box':('box','carton','packaged box','tissue box'),
    'bottle':('bottle','water bottle','beverage bottle'),
    'tissue':('tissue','tissue paper','tissue pack','tissue box'),
    'bag':('bag','paper bag','plastic bag'),
    'can':('can','beverage can','tin can'),
    'mouse':('computer mouse','mouse'),
}


def detection_query(text,cfg):
    normalized=normalize_prompt(text);phrases=normalized.rstrip('.').split('. ') if normalized else []
    expanded=[];generic=bool(phrases) and all(p in NAME_VARIANTS for p in phrases)
    for phrase in phrases:
        for variant in NAME_VARIANTS.get(phrase,(phrase,)):
            if variant not in expanded:expanded.append(variant)
    # Keep the query bounded; never omit one of the user's requested categories.
    if len(expanded)>24:expanded=phrases
    threshold=float(cfg.get('grounded_box_threshold',.25))
    if generic:threshold=min(threshold,float(cfg.get('grounded_name_box_threshold',.25)))
    if not .1<=threshold<=.9:raise ValueError('语义检测阈值无效')
    return '. '.join(expanded)+('.' if expanded else ''),threshold


def category_label_matches(requested,label):
    phrases=normalize_prompt(requested).rstrip('.').split('. ')
    if not phrases or not all(p in NAME_VARIANTS for p in phrases):return True
    anchors={'box':{'box','carton'},'bottle':{'bottle'},'tissue':{'tissue'},
             'bag':{'bag'},'can':{'can'},'mouse':{'mouse'}}
    tokens=set(re.findall(r'[a-z]+',str(label).lower()))
    return any(tokens&anchors[p] for p in phrases)


def local_model_paths(cfg,root):
    result={};missing=[]
    for key in ('grounding_dino_model','sam2_model'):
        value=str(cfg.get(key,'')).strip()
        folder=Path(value).expanduser();folder=(folder if folder.is_absolute() else Path(root)/folder).resolve()
        required=['config.json','model.safetensors','preprocessor_config.json']
        if key=='grounding_dino_model':required+=['tokenizer_config.json','tokenizer.json']
        if not value or any(not (folder/n).is_file() or not (folder/n).stat().st_size for n in required):missing.append(str(folder))
        result[key]=folder
    if missing:raise ValueError('Grounded SAM 2模型未完整安装：\n'+'\n'.join(missing))
    return result


def select_boxes(boxes,scores,labels,shape,threshold=.25,limit=12):
    """Class-agnostic NMS merges synonym detections before SAM's decoder."""
    boxes=np.asarray(boxes,float).reshape(-1,4);scores=np.asarray(scores,float).reshape(-1)
    if not len(boxes) and not len(scores):return []
    if len(boxes)!=len(scores) or len(labels)!=len(scores):raise ValueError('检测结果数量不一致')
    h,w=shape[:2];chosen=[]
    for i in np.argsort(-scores,kind='stable'):
        box=boxes[i].copy()
        if not np.isfinite(np.r_[box,scores[i]]).all() or scores[i]<threshold:continue
        box[[0,2]]=np.clip(box[[0,2]],0,w);box[[1,3]]=np.clip(box[[1,3]],0,h)
        if min(box[2:]-box[:2])<2:continue
        area=np.prod(box[2:]-box[:2]);duplicate=False
        for previous in chosen:
            b=previous['box'];inter=np.maximum(0,np.minimum(box[2:],b[2:])-np.maximum(box[:2],b[:2])).prod()
            if inter/max(1.,area+np.prod(b[2:]-b[:2])-inter)>.55:duplicate=True;break
        if duplicate:continue
        chosen.append(dict(box=box,score=float(scores[i]),label=str(labels[i])))
        if len(chosen)>=limit:break
    return chosen


class GroundedSAM2:
    def __init__(self,cfg,root):
        import torch
        from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection,Sam2Processor,Sam2Model
        self.cfg=cfg;self.torch=torch;self.device=cfg['device'];self.timings={}
        assets=local_model_paths(cfg,root)
        for mode in ('all','beverages'):prompt_for(cfg,mode)
        self.detector_processor=AutoProcessor.from_pretrained(str(assets['grounding_dino_model']),local_files_only=True)
        self.detector=AutoModelForZeroShotObjectDetection.from_pretrained(
            str(assets['grounding_dino_model']),local_files_only=True,use_safetensors=True).eval().to(self.device)
        self.sam_processor=Sam2Processor.from_pretrained(str(assets['sam2_model']),local_files_only=True)
        self.sam=Sam2Model.from_pretrained(str(assets['sam2_model']),local_files_only=True,use_safetensors=True).eval().to(self.device)
        # Use torch's existing implementation; never compile/download operators
        # in the running robot environment. RGB/depth/control stay separate.
        if hasattr(self.detector.config,'disable_custom_kernels'):self.detector.config.disable_custom_kernels=True

    def proposals(self,frame,mode,prompt=None):
        from .core import crop_object
        from .perception import duplicate_mask
        torch=self.torch;image=frame['image'];rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        requested=prompt_for(self.cfg,mode) if prompt is None else normalize_prompt(prompt)
        text,threshold=detection_query(requested,self.cfg)
        if not text:raise ValueError('语义识别范围为空')
        start=time.monotonic()
        inputs=self.detector_processor(images=rgb,text=text,return_tensors='pt',
            size={'shortest_edge':640,'longest_edge':1066}).to(self.device)
        if inputs['input_ids'].shape[-1]>256:raise ValueError('识别描述超出模型长度，请减少物品类别')
        with torch.inference_mode():outputs=self.detector(**inputs)
        result=self.detector_processor.post_process_grounded_object_detection(outputs,inputs.input_ids,
            threshold=threshold,
            text_threshold=float(self.cfg.get('grounded_text_threshold',.20)),target_sizes=[image.shape[:2]])[0]
        labels=result['text_labels'] if 'text_labels' in result else result.get('labels',[])
        selected=select_boxes(result['boxes'].detach().cpu().numpy(),result['scores'].detach().cpu().numpy(),
                              labels,image.shape,threshold=threshold)
        # GroundingDINO can return only an ambiguous fragment such as "pack"
        # from "tissue pack". A lower category threshold must not accept that
        # as evidence for the requested object category.
        selected=[v for v in selected if category_label_matches(requested,v['label'])]
        grounded=time.monotonic();items=[]
        if selected:
            sam_inputs=self.sam_processor(images=rgb,input_boxes=[[v['box'].tolist() for v in selected]],return_tensors='pt').to(self.device)
            with torch.inference_mode():sam_outputs=self.sam(**sam_inputs,multimask_output=False)
            masks=self.sam_processor.post_process_masks(sam_outputs.pred_masks.detach().cpu(),sam_inputs['original_sizes'])[0]
            masks=masks[:,0].numpy().astype(bool)
            qualities=sam_outputs.iou_scores.detach().cpu().numpy().reshape(len(selected),-1)[:,0]
            if masks.shape!=(len(selected),*image.shape[:2]):raise ValueError('SAM2分割尺寸与RGB不一致')
            for proposal,mask,quality in zip(selected,masks,qualities):
                if mask.sum()<100 or not np.isfinite(quality) or quality<float(self.cfg.get('sam2_mask_quality',.5)):continue
                # Keep the actual mask (not just the rectangle) for RGB-D.
                yy,xx=np.nonzero(mask);box=[float(xx.min()),float(yy.min()),float(xx.max()+1),float(yy.max()+1)]
                if duplicate_mask(mask,box,items):continue
                items.append(dict(box=box,mask=mask,crop=crop_object(image,mask),source='grounded_sam2',
                    detection_score=proposal['score'],semantic_label=proposal['label'],mask_score=float(quality)))
        self.timings={'grounding_ms':round((grounded-start)*1000,1),'segmentation_ms':round((time.monotonic()-grounded)*1000,1),
                      'prompt':text,'requested_prompt':requested,'box_threshold':threshold,'candidates':len(selected)}
        return items
