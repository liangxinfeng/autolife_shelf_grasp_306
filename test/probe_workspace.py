"""Hypothetical target sweep using saved joints; never moves hardware."""
import json
from pathlib import Path
import numpy as np
import yaml
from autolife_shelf_grasp_306.planning import Planner
from autolife_handeye_calibration_306.verification_core import load_geometry

cfg=yaml.safe_load((Path(__file__).parents[1]/'config/grasp.yaml').read_text())
f=np.load('/tmp/shelf306_support_fixture.npz');m=json.loads(str(f['meta']))
cfg['up_torso']=f['up'].tolist();p=Planner(cfg);q=p.reset_plan(m['q'])['end_q']
tcp=load_geometry(json.loads(Path(cfg['calibration']).read_text()),p.fk,cfg['tcp_offset_mm'])
for x,y,z in [(.737,-.152,-.096),(.737,-.031,-.075),(.65,-.30,-.15),(.72,-.3,-.15),(.76,-.3,-.10),(.70,-.45,-.15),(.65,-.4,-.25),(.70,-.1,-.30)]:
    try:
        plan=p.plan(q,tcp,np.array([x,y,z]),{'diameter_m':.07,'height_m':.20})
        max_step=max(max(abs(a[n]-b[n]) for n in p.names) for s in plan['segments'] for a,b in zip(s['waypoints'],s['waypoints'][1:]))
        error=max(np.linalg.norm(p.pose(s['waypoints'][-1],tcp)[0]-s['tcp_goal']) for s in plan['segments'])
        assert max_step<=8.00001 and error<=.004
        grasp_r=p.pose(plan['segments'][1]['waypoints'][-1],tcp)[1]
        from scipy.spatial.transform import Rotation
        for s in plan['segments'][2:]:
            for w in s['waypoints']:
                assert Rotation.from_matrix(grasp_r.T@p.pose(w,tcp)[1]).magnitude()<=np.deg2rad(5.00001 if plan['orientation_mode']=='flexible' else 3.00001)
        print('PASS',x,y,z,plan['orientation_mode'],plan['max_orientation_deviation_deg'],plan['checked_states'],plan['seconds'],'max_step',round(max_step,2),'error_mm',round(error*1000,2),flush=True)
    except ValueError as e:print('REJECT',x,y,z,str(e),flush=True)
