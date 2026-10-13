import numpy as np
from autolife_handeye_calibration_306.verification_trajectory import JointPath
from autolife_shelf_grasp_306.motion import configure_grasp_path,Follower

def test_lagged_motor_has_continuous_progress_not_repeated_hard_clipping():
    p=configure_grasp_path(JointPath(['a'],[{'a':0.},{'a':80.}]),{})
    f=Follower(p,0,8);q=0.;target=0.;vel=[]
    for i in range(1,3000):
        q+=(target-q)*(1-np.exp(-.01/.43))
        command=f.step({'a':q},i*.01)
        if command:target=command['a']
        if 15<f.progress<65:vel.append(f.velocity)
        assert abs(target-q)<=8.+1e-6
        if f.final:break
    assert f.final and min(vel)>0
    assert np.std(vel)/np.mean(vel)<.05
    assert f.limited_ticks/max(1,f.total_ticks)<.05
