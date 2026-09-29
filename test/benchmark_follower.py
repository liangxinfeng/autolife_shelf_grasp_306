"""Same first-order motor simulation for old/new path-clock logic. No ROS."""
import numpy as np
from autolife_handeye_calibration_307.verification_trajectory import JointPath
from autolife_shelf_grasp_306.motion import Follower,configure_grasp_path

def simulate(partial):
    path=configure_grasp_path(JointPath(['a'],[{'a':0.},{'a':60.}]),{})
    f=Follower(path,0,3.);q=0.;target=0.;max_lead=0.
    for i in range(1,6000):
        now=i/60;dt=1/60;q+=min(1.,3.*dt)*(target-q)
        if partial:
            command=f.step({'a':q},now)
            if command:target=command['a']
        else:
            remaining=path.length-f.progress
            speed=min(path.max_speed,np.sqrt(max(0,2*path.acceleration*remaining)))
            f.velocity=min(speed,f.velocity+path.acceleration*dt)
            proposed=min(path.length,f.progress+f.velocity*dt);value=path.at(proposed)[0]
            if abs(value-q)>3:f.velocity=0.
            else:f.progress=proposed;target=value
        max_lead=max(max_lead,abs(target-q))
        if f.final and abs(q-60)<.35:return {'seconds':round(now,3),'max_lead_deg':round(max_lead,6)}
    raise RuntimeError('simulation timed out')

if __name__=='__main__':print({'old':simulate(False),'new':simulate(True),'measured_robot':False})
