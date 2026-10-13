"""Recent full-view dimensions, never cached target coordinates or commands."""
import json,time
from pathlib import Path
import numpy as np
from .core import atomic_json,unit,crop_object


def partial_views(image,mask):
    """Store appearance of clipping, without estimating any missing geometry."""
    ys=np.where(mask)[0]
    if not len(ys):return []
    lo,hi=int(ys.min()),int(ys.max());views=[]
    for fraction in (.1,.2,.3,.4):
        cut=int(hi-(hi-lo)*fraction)
        try:views.append(crop_object(image[:cut],mask[:cut]))
        except ValueError:pass
        cut=int(lo+(hi-lo)*fraction)
        try:views.append(crop_object(image[cut:],mask[cut:]))
        except ValueError:pass
    return views


class GeometryMemory:
    def __init__(self,path):self.path=Path(path)

    def entries(self,now):
        try:
            rows=json.loads(self.path.read_text())
            return [r for r in rows if isinstance(r,dict) and 0<=now-float(r.get('time',0))<=1800]
        except (OSError,ValueError,TypeError):return []

    def remember(self,item,product,geometry,now=None):
        if geometry.get('image_cropped') or not product.get('temporary') or product.get('dimension_frames',0)<5:return
        now=time.time() if now is None else now
        feature=unit(item['feature']);label=str(item.get('semantic_label','')).strip().lower()
        if not label:return
        dimensions={k:float(product[k]) for k in ('diameter_m','height_m','depth_m')}
        if not np.isfinite(list(dimensions.values())).all():return
        variants=np.asarray(item.get('partial_features',[]),float)
        if variants.size and (variants.ndim!=2 or variants.shape[1]!=len(feature) or not np.isfinite(variants).all()):return
        row=dict(time=now,label=label,feature=feature.tolist(),geometry=product['geometry'],**dimensions)
        if variants.size:row['partial_features']=[unit(f).tolist() for f in variants[:8]]
        entries=self.entries(now)
        entries=[r for r in entries if not (r.get('label')==label and np.asarray(r.get('feature',[])).shape==feature.shape
                  and np.dot(r['feature'],feature)>.97)]
        atomic_json(self.path,(entries+[row])[-32:])

    def find(self,item,now=None):
        now=time.time() if now is None else now
        feature=unit(item['feature']);label=str(item.get('semantic_label','')).strip().lower()
        scored=[]
        for row in self.entries(now):
            f=np.asarray(row.get('feature',[]),float)
            if row.get('label')!=label or f.shape!=feature.shape or not np.isfinite(f).all():continue
            variants=np.asarray([f,*row.get('partial_features',[])],float)
            if variants.ndim!=2 or variants.shape[1]!=len(feature) or not np.isfinite(variants).all():continue
            score=max(float(unit(v)@feature) for v in variants)
            if score>=.94:scored.append((score,row))
        scored.sort(key=lambda x:x[0],reverse=True)
        if not scored:return None
        if len(scored)>1 and scored[0][0]-scored[1][0]<.02:
            a,b=[np.array([r[k] for k in ('diameter_m','height_m','depth_m')]) for _,r in scored[:2]]
            if np.any(abs(a-b)>.008):return None
        row=scored[0][1]
        return {k:row[k] for k in ('geometry','diameter_m','height_m','depth_m')}|dict(reference_age_s=now-row['time'])
