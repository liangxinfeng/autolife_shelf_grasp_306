"""Task-local recognition scope, independent of the manual detection preset."""
import re
from .grounded_sam2 import ALIASES, NAME_VARIANTS, normalize_prompt


def task_detection_request(target, products):
    target=str(target).strip()
    if not target:raise ValueError('任务物品名称为空')
    matched=[p for p in products if p['name']==target]
    try:
        prompt=normalize_prompt(target)
    except ValueError:
        if not matched:raise
        prompt=''
    # A generic/qualified object name is still a semantic request when a
    # catalog item happens to have the same name (e.g. "盒子"). Brand names
    # retain exact SKU retrieval; never substitute any bottle for "怡宝".
    nouns=set(ALIASES.values())|set(NAME_VARIANTS)
    phrases=prompt.rstrip('.').split('. ') if prompt else []
    described=bool(phrases) and all(any(p==n or p.endswith(' '+n) for n in nouns) for p in phrases)
    semantic=bool(prompt) and (not matched or described)
    if not semantic:
        shapes={p.get('geometry','cylinder') for p in matched}
        prompt=('bottle. can. cup.' if shapes=={'cylinder'} else
                'box. container. bag. bottle. can. cup.')
    return dict(target=target,mode='all',prompt=prompt,semantic=semantic,
                product_ids=[p['id'] for p in matched])


def infer_scene(perception, frame, store, mode, request=None):
    if request is None:return perception.infer(frame,store,mode)
    return perception.infer(frame,store,request['mode'],prompt=request['prompt'])
