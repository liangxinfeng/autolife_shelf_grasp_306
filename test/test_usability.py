import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from autolife_shelf_grasp_306.point_selection import choose_mask,point_pixel,visual_description,reobserve
from autolife_shelf_grasp_306.step_posture import targets,normalize
from autolife_shelf_grasp_306.task_flow import validate
from autolife_shelf_grasp_306.grip_width import closing_command,command_from_plan

def test_recorded_height_has_no_separate_move():
    p=dict(body_mode='recorded',lowering_cm=35)
    assert targets(p)=={}
    assert normalize(p)['lowering_cm']==0
    flow=validate(dict(steps=[dict(kind='action',target='丢垃圾',posture=p)]))
    assert validate(flow)==flow
    assert targets(dict(body_mode='recorded',head_pitch_deg=10))=={'Joint_Neck_Pitch':-10.}
    with pytest.raises(ValueError):validate(dict(steps=[dict(kind='navigate',target='前台',posture=p)]))

@pytest.mark.parametrize('arm',['left','right'])
def test_measured_curve_and_plan_integrity(tmp_path,arm):
    path=tmp_path/'jaw.json'
    path.write_text(json.dumps(dict(robot_id=345,hardware='dm_type1',measured=True,
        arms={arm:[[5,330],[45,180],[95,10]]})))
    cfg=dict(arm=arm,gripper_width_calibration=str(path))
    for width,pos in [(45,180),(95,10),(25,255),(70,95)]:
        assert closing_command(width/1000,0,cfg)['target_deg']==pytest.approx(pos)
    command=closing_command(.045,4,cfg)
    plan=dict(grip_command=command,grasp_width_m=.045,config=cfg)
    assert command_from_plan(plan)==195
    for width in [0,.003,.005,.096,float('nan')]:
        with pytest.raises(ValueError):closing_command(width,0,cfg)
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='标定已变化'):command_from_plan(plan)

def test_point_component_quality_and_bounds():
    masks=np.zeros((3,100,100),bool)
    masks[0,30:60,30:60]=True;masks[0,80:95,80:95]=True
    masks[1,:,:]=True;masks[2,0:20,0:20]=True
    mask,q=choose_mask(masks,[.9,.99,.95],(45,45))
    assert mask.sum()==900 and q==.9
    for xy in [(-.1,.2),(1,.2),(float('nan'),.5)]:
        with pytest.raises(ValueError):point_pixel(*xy,(100,100))
    with pytest.raises(ValueError):choose_mask(masks,[.1,.99,.95],(45,45))
    image=np.zeros((100,100,3),np.uint8);image[:,:,0]=255
    assert visual_description(image,mask)=='蓝色物体'

def test_reobserve_uses_fresh_frame_then_identity_gate(monkeypatch):
    from autolife_shelf_grasp_306 import point_selection
    from autolife_shelf_grasp_306.selection import resolve_selected
    mask=np.zeros((100,100),bool);mask[30:60,30:60]=True
    chosen=dict(mask=mask,feature=np.array([1.,0.]),id=1,product=None,name='待确认')
    frame={'image':np.ones((100,100,3),np.uint8)}
    seen=[]
    def segment(p,f,store,x,y,describe):
        seen.append(f)
        return dict(mask=mask,feature=np.array([0.,1.]),id=2,product=None,name='other')
    monkeypatch.setattr(point_selection,'segment',segment)
    fresh=reobserve(None,frame,None,chosen)
    assert seen[0] is frame
    assert resolve_selected(chosen,fresh)[0] is None
