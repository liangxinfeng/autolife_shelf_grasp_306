"""Saved RGB-D only: no robot node, no motor commands."""
import json
from pathlib import Path
import numpy as np
import yaml
from autolife_shelf_grasp_306.bottle import estimate
from autolife_shelf_grasp_306.planning import Planner
from autolife_handeye_calibration_306.verification_core import load_geometry

root=Path('/home/ubuntu/ros2_ws/src/autolife_shelf_grasp_306')
cfg=yaml.safe_load((Path(__file__).parents[1]/'config/grasp.yaml').read_text())
f=np.load('/tmp/shelf306_support_fixture.npz');meta=json.loads(str(f['meta']))
frame={k:f[k] for k in ['image','depth','K','D']}
frame.update(frame_id=meta['frame_id'],depth_id=meta['frame_id'],intrinsics=meta['intrinsics'],depth_intrinsics=meta['intrinsics'])
centers=[]
for i,p in enumerate(meta['products']):
    try:
        c,stats=estimate(frame,f['mask'+str(i)],f['T'],f['up'],p,cfg)
        centers.append((p,c));print('BOTTLE_GEOMETRY',p['name'],c.tolist(),stats,flush=True)
    except Exception as e:print('BOTTLE_GEOMETRY_FAILED',p['name'],str(e),flush=True)
cfg['up_torso']=f['up'].tolist();planner=Planner(cfg)
reset=planner.reset_plan(meta['q'])
print('ACTUAL_RESET_PREFLIGHT',reset['checked_states'],flush=True)
cal=json.loads(Path(cfg['calibration']).read_text());tcp=load_geometry(cal,planner.fk,cfg['tcp_offset_mm'])
for p,c in centers:
    try:
        route=planner.plan(reset['end_q'],tcp,c,p)
        print('ACTUAL_BOTTLE_PATH',p['name'],route['checked_states'],route['seconds'],flush=True)
    except Exception as e:print('ACTUAL_PATH_REJECTED',p['name'],str(e),flush=True)
