"""动作参照复用抓取库的DINOv2整物体特征；不读取硬件、不生成运动。

外观与几何分离：补录的任意位置只提供外观，初始帧仍是唯一位姿基准。
"""
import numpy as np
from .core import crop_object


class AmbiguousObject(ValueError):
    """多个实体外观近似时，不能用路径是否可达来替用户挑对象。"""


def normalized(values):
    a=np.asarray(values,dtype=np.float32)
    if a.ndim!=2 or not len(a) or not np.isfinite(a).all():raise ValueError('物体外观特征无效')
    norms=np.linalg.norm(a,axis=1,keepdims=True)
    if np.any(norms<1e-8):raise ValueError('物体外观特征为空')
    return a/norms


def select_match(query,reference,threshold=.72,margin=.06):
    """取最相近已录视角，避免正面/背面平均反而惩罚已录的背面。

margin比较当前画面中不同实体，而不是同一物体的两个补录视角。
返回相似度而非置信概率；几何可靠性另行判断。
"""
    scores=normalized(query)@normalized(reference).T
    best=scores.max(axis=1);order=np.argsort(-best,kind='stable');index=int(order[0])
    if best[index]<threshold:return None
    if len(order)>1 and best[index]-best[order[1]]<margin:
        raise AmbiguousObject('画面中有多个外观近似的已记忆物体，请只保留本次交互对象可见；无需重新录入')
    return index,int(scores[index].argmax()),float(best[index])


def memory_prompt(cfg,views):
    """物体记忆检索独立于抓取页的临时文字筛选；补录不改变初始参照。"""
    from .grounded_sam2 import normalize_prompt
    # 自定义名字不一定是类别；不能解释的名字仍使用原有广泛候选检索。
    name=str(views[0][0].get('name','')) if views else ''
    try:prompt=normalize_prompt(name)
    except ValueError:prompt=''
    default=cfg.get('grounded_sam2_prompts',{}).get('all','bottle. cup. box. bag. tissue. trash bin lid. cabinet door.')
    # 保留通用候选，避免用户自定义英文编号变成唯一检测条件。
    phrases=(prompt+normalize_prompt(default)).split('.')
    unique=list(dict.fromkeys(x.strip() for x in phrases if x.strip()))[:16]
    return normalize_prompt('. '.join(unique))


class ObjectAppearanceMemory:
    def __init__(self,perception,cfg):
        self.perception=perception;self.cfg=cfg;self.cache={}

    def reference_features(self,store,ident,views):
        signature=tuple(store.view_ids(ident))
        cached=self.cache.get(ident)
        if cached is None or cached[0]!=signature:
            crops=[crop_object(a['image'],a['mask']) for _,a in views]
            # 分批限制显存；模型复用已有抓取进程，不重复加载。
            values=np.concatenate([self.perception.embed(crops[i:i+8]) for i in range(0,len(crops),8)])
            self.cache[ident]=(signature,normalized(values))
            while len(self.cache)>32:self.cache.pop(next(iter(self.cache)))
        return self.cache[ident][1]

    def recognize(self,store,ident,views,frame):
        reference=self.reference_features(store,ident,views)
        segmenter=getattr(self.perception,'segmenter',None)
        if segmenter is not None:
            items=segmenter.proposals(frame,'all',prompt=memory_prompt(self.cfg,views))
        else:
            items=self.perception._legacy_items(frame,'all')
        if not items:return None
        # 不复用含背景的矩形crop，登记和查询都使用同一分割灰底预处理。
        query=self.perception.embed([crop_object(frame['image'],x['mask']) for x in items])
        match=select_match(query,reference,float(self.cfg.get('match_threshold',.72)),float(self.cfg.get('match_margin',.06)))
        if match is None:return None
        index,view,score=match
        return items[index],dict(recognition_method='dinov2_object',recognition_view='初始帧' if view==0 else f'补录视角{view}',
                                recognition_similarity=score,recognition_candidates=len(items),recognition_views=len(views))
