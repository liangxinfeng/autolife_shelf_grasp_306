"""One selected semantic table -> observed support, full-scene obstacles.

No new network/model and no robot commands. The table model reuses the loaded
Grounded SAM 2 instance; grasp prompts and SKU tracking remain untouched.
"""
import time
import numpy as np
from .bottle import object_points
from .demo_surface import surfaces


def semantic_table_surface(perception,frame,camera_ground,cfg,environment,settings,radius,
                           remove_self,diagnostics,support_outline=None,support_offset=(0.,0.)):
    if perception.segmenter is None:raise ValueError('自动放置需要启用桌面语义分割')
    started=time.monotonic()
    proposals=perception.segmenter.proposals(frame,'all',prompt='table. tabletop.')
    diagnostics.setdefault('timing_seconds',{})['table_segmentation']=round(time.monotonic()-started,4)
    geometry_started=time.monotonic()
    if not proposals:raise ValueError('未识别到桌面，请让桌面进入头部相机视野')
    cloud,pixels,_=object_points(frame,frame['depth']>0,camera_ground,cfg)
    # Existing empirical TCP target correction shifts the planned contact
    # centre. Cover its magnitude so the shifted bottle still fits the patch.
    correction_padding=float(np.linalg.norm(cfg.get('target_offset_torso_mm',[0,0,0])))/1000
    # Reject intersecting duplicate masks without merging distinct tables.
    masks=[];options=[];failures=[]
    for item in sorted(proposals,key=lambda p:-p['detection_score'])[:4]:
        mask=item['mask']
        if any(np.logical_and(mask,m).sum()/max(1,min(mask.sum(),m.sum()))>.8 for m in masks):continue
        masks.append(mask)
        seed=cloud[mask[pixels[:,1],pixels[:,0]]]
        seed=remove_self(seed);diag={}
        try:
            candidates,table=surfaces(environment,settings,radius+correction_padding,diag,support_points=seed,
                                     support_outline=support_outline,support_offset=support_offset)
            diag['target_correction_padding_m']=correction_padding
            # Prefer the nearest reachable-side surface. Actual IK ranks
            # positions next; this does not pretend that nearest means reachable.
            distance=float(np.min(np.linalg.norm(candidates[:,:2],axis=1)))
            options.append((distance,candidates,table,diag,item))
        except ValueError as exc:
            failures.append((diag,str(exc),item))
    diagnostics['semantic_tables']=len(masks)
    diagnostics['timing_seconds']['surface_geometry']=round(time.monotonic()-geometry_started,4)
    if not options:
        if failures:
            diag,error,item=max(failures,key=lambda x:x[0].get('plane_points',0))
            diagnostics.update(diag,semantic_box=item['box'])
            raise ValueError(error)
        raise ValueError('选中桌面有效深度不足，请调整观察位置')
    _,points,table,diag,item=min(options,key=lambda v:v[0])
    diagnostics.update(diag,semantic_box=item['box'],selected_table_score=item['detection_score'],
                       method='grounded_sam2_local_support',candidate_count=len(points))
    table['semantic_box']=item['box'];table['support_method']='observed_semantic_plane'
    return points,table,item
