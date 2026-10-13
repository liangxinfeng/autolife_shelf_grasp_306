"""Reassociate the user's selected instance, not a detector's transient ID."""
import numpy as np


def resolve_selected(chosen,fresh):
    ranked=[];diagnostics=[]
    for item in fresh:
        a,b=np.asarray(chosen['mask'],bool),np.asarray(item['mask'],bool)
        if a.shape!=b.shape or not a.any() or not b.any():continue
        intersection=float(np.count_nonzero(a&b));union=float(np.count_nonzero(a|b))
        iou=intersection/max(1.,union);coverage=intersection/max(1.,min(a.sum(),b.sum()))
        x,y=np.asarray(chosen['feature'],float),np.asarray(item['feature'],float)
        if x.shape!=y.shape or not np.isfinite(np.r_[x,y]).all():continue
        similarity=float(x@y/max(1e-12,np.linalg.norm(x)*np.linalg.norm(y)))
        ay,ax=np.nonzero(a);by,bx=np.nonzero(b)
        diagonal=max(1.,np.hypot(np.ptp(ax),np.ptp(ay)))
        shift=float(np.hypot(ax.mean()-bx.mean(),ay.mean()-by.mean())/diagonal)
        old_product=chosen.get('product');new_product=item.get('product')
        conflicting=bool(old_product and new_product and old_product['id']!=new_product['id'])
        ok=(not conflicting and shift<=.30 and
            ((iou>=.55 and similarity>=.85) or (iou>=.40 and coverage>=.75 and similarity>=.92)))
        score=.55*iou+.35*similarity+.10*coverage
        diagnostics.append(dict(id=item['id'],iou=iou,coverage=coverage,similarity=similarity,
                                normalized_shift=shift,conflicting_product=conflicting,accepted=bool(ok)))
        if ok:ranked.append((score,item))
    ranked.sort(key=lambda pair:pair[0],reverse=True)
    reason=('当前画面未可靠找到所选物品，请重新识别' if not ranked else
            '附近有多个相似目标，无法唯一确认，请重新选择' if len(ranked)>1 and ranked[0][0]-ranked[1][0]<.08 else '')
    if reason:return None,dict(reason=reason,candidates=diagnostics)
    # Recognition can cross the catalog threshold without the physical object
    # changing. Keep the user's catalog/temporary identity for this request;
    # all geometry comes from the new mask and newly paired RGB-D frames.
    winner=ranked[0][1]
    product=(winner.get('product') or chosen.get('product')) if chosen.get('product') else None
    selected={**winner,'id':chosen['id'],'product':product,'name':chosen.get('name',winner.get('name',''))}
    return selected,dict(selected_id=chosen['id'],detector_id=winner['id'],candidates=diagnostics)
