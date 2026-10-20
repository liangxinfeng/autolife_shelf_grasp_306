import pytest
from autolife_shelf_grasp_306.task_flow import validate
from autolife_shelf_grasp_306.step_posture import normalize,targets,permitted_navigation_joints

def test_old_flow_unchanged():
    row=dict(kind='grasp',target='水瓶')
    assert validate(dict(steps=[row]))==dict(version=1,steps=[row])

@pytest.mark.parametrize('mode,height,pitch',[('forward',56,45),('reverse',53,-30),('keep',0,0)])
def test_round_trip_and_sign(mode,height,pitch):
    p=dict(body_mode=mode,lowering_cm=height,head_pitch_deg=pitch)
    r=validate(dict(steps=[dict(kind='navigate',target='前台',posture=p)]))
    assert validate(r)==r
    q=targets(p);assert q['Joint_Neck_Pitch']==-pitch
    if mode!='keep':assert q['Joint_Knee']*(1 if mode=='forward' else -1)>0

@pytest.mark.parametrize('p',[dict(body_mode='reverse',lowering_cm=54),dict(body_mode='forward',lowering_cm=57),
 dict(head_pitch_deg=46),dict(head_pitch_deg=-31),dict(head_pitch_deg=float('nan')),dict(lowering_cm=float('inf')),dict(body_mode='bad')])
def test_bad_settings_rejected(p):
    with pytest.raises(ValueError):normalize(p)

def test_nav_corridor_reverse_forward_and_arms_not_exempt():
    start=targets(dict(body_mode='reverse',lowering_cm=30,head_pitch_deg=30))
    setting=dict(body_mode='forward',lowering_cm=40,head_pitch_deg=0);end=targets(setting)
    for t in (0,.1,.5,.8,1):
        q={n:start[n]+t*(end[n]-start[n]) for n in start}
        allowed=permitted_navigation_joints(start,q,setting)
        assert allowed==set(start) and 'Joint_Left_Elbow' not in allowed and 'Joint_Waist_Yaw' not in allowed
    q=dict(end);q['Joint_Waist_Pitch']+=10
    with pytest.raises(ValueError):permitted_navigation_joints(start,q,setting)
    q=dict(end);q['Joint_Neck_Pitch']=30
    with pytest.raises(ValueError):permitted_navigation_joints(start,q,setting)
