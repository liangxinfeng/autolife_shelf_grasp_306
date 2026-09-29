"""Offline pose feasibility measurements from saved RGB-D, no ROS node."""
import json
from pathlib import Path
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from autolife_shelf_grasp_306.planning import Planner
from autolife_shelf_grasp_306.bottle import estimate,horizontal_basis
from autolife_handeye_calibration_307.verification_core import load_geometry

root=Path('/home/ubuntu/ros2_ws/src/autolife_shelf_grasp_306')
cfg=yaml.safe_load((root/'config/grasp.yaml').read_text());p=Planner(cfg)
f=np.load('/tmp/shelf306_support_fixture.npz');m=json.loads(str(f['meta']))
frame={k:f[k] for k in ['image','depth','K','D']}
frame.update(frame_id=m['frame_id'],depth_id=m['frame_id'],intrinsics=m['intrinsics'],depth_intrinsics=m['intrinsics'])
q=p.reset_plan(m['q'])['end_q'];seed=np.deg2rad([q[n] for n in p.names]);tcp=load_geometry(json.loads(Path(cfg['calibration']).read_text()),p.fk,15.)
tool=Rotation.from_quat(cfg['T_wrist_tool_quat_xyzw']).as_matrix()
base=np.column_stack(horizontal_basis(f['up']));r0=p.pose(q,tcp)[1]
desired=min([base@tool.T,base@np.diag([1,-1,-1])@tool.T],key=lambda r:Rotation.from_matrix(r0.T@r).magnitude())
for i,product in enumerate(m['products']):
    center,_=estimate(frame,f['mask'+str(i)],f['T'],f['up'],product,cfg)
    for w in [.1,.02,.005,0.]:
        def res(x):
            pt,r=p.pose({**q,**dict(zip(p.names,np.rad2deg(x)))},tcp)
            return np.r_[pt-center,w*Rotation.from_matrix(desired.T@r).as_rotvec(),.0002*(x-seed)]
        fit=least_squares(res,seed,bounds=p.bounds,max_nfev=200,gtol=1e-9,ftol=1e-9,xtol=1e-9)
        qq={**q,**dict(zip(p.names,np.rad2deg(fit.x)))};pt,r=p.pose(qq,tcp)
        try:p.guard.check(qq);collision='clear'
        except Exception as e:collision=str(e)
        print(product['name'],'weight',w,'pos_mm',np.linalg.norm(pt-center)*1000,'angle_deg',np.rad2deg(Rotation.from_matrix(desired.T@r).magnitude()),'q',np.rad2deg(fit.x).round(1).tolist(),'collision',collision,flush=True)
