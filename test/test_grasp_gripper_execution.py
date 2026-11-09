import json
from types import SimpleNamespace
import numpy as np
import pytest
from autolife_shelf_grasp_306.grasp_gripper import ClosingEvidence,execute


def test_gap_is_not_contact_even_after_command_is_sent():
    e=ClosingEvidence(13,203,58.9,np.array([5,45,90]),np.array([330,180,10]))
    for t in np.linspace(0,3,150):assert e.update(t,113.9,203) is None


def test_plateau_during_ramp_is_not_contact():
    e=ClosingEvidence(13,203,58.9,np.array([5,45,90]),np.array([330,180,10]))
    for t in np.linspace(0,3,150):assert e.update(t,148,163) is None
    for t in np.linspace(3.1,4,45):result=e.update(t,148,203)
    assert result=='contact_unconfirmed'


@pytest.mark.parametrize('stop_at,expected',[(None,'at_target'),(135.,'contact_unconfirmed'),(114.,'timeout')])
def test_actual_output_reaches_command_with_bounded_slew(tmp_path,stop_at,expected):
    path=tmp_path/'curve.json';path.write_text(json.dumps(dict(robot_id=345,hardware='dm_type1',measured=True,
        arms={'right':[[5,330],[45,180],[90,10]]})))
    cfg=dict(arm='right',gripper_width_calibration=str(path),gripper_speed_scale=.5)
    q={'Joint_Left_Gripper':22.,'Joint_Right_Gripper':13.};sent=[];holds=[];time=[0.]
    node=SimpleNamespace(_active_goals={},progress_watch={},_gripper_pub=object())
    def publish(pub,data):
        assert data['left_gripper_target_joints_position']==[22.]
        sent.append((time[0],data['right_gripper_target_joints_position'][0]))
    node._publish_vendor=publish
    def checked():
        time[0]+=.02
        if sent:
            target=sent[-1][1] if stop_at is None else min(sent[-1][1],stop_at)
            q['Joint_Right_Gripper']+=float(np.clip(target-q['Joint_Right_Gripper'],-3,3))
        return dict(q)
    state=dict(targets={'Joint_Left_Gripper':22.,'Joint_Right_Gripper':203.},closing=True,tolerance=2.)
    args=(node,cfg,'right',state,checked,lambda:time[0],lambda *a:holds.append(a),lambda data:None,.0589)
    if expected=='timeout':
        with pytest.raises(ValueError,match='夹持'):execute(*args)
        assert holds
    else:assert execute(*args)==expected
    assert max(v for t,v in sent)==203 and min(v for t,v in sent)>=13
    assert max(abs((b[1]-a[1])/(b[0]-a[0])) for a,b in zip(sent,sent[1:]))<=120.01


def test_stop_precedes_next_command(tmp_path):
    path=tmp_path/'curve.json';path.write_text(json.dumps(dict(robot_id=345,hardware='dm_type1',measured=True,
        arms={'right':[[5,330],[45,180],[90,10]]})))
    cfg=dict(arm='right',gripper_width_calibration=str(path));t=[0.];sent=[]
    q={'Joint_Left_Gripper':22.,'Joint_Right_Gripper':13.}
    node=SimpleNamespace(_active_goals={},progress_watch={},_gripper_pub=object(),_publish_vendor=lambda *a:sent.append(t[0]))
    def checked():
        t[0]+=.02
        if t[0]>.15:raise ValueError('用户停止')
        return q
    with pytest.raises(ValueError,match='用户停止'):
        execute(node,cfg,'right',dict(targets={'Joint_Left_Gripper':22.,'Joint_Right_Gripper':148.},closing=True),
            checked,lambda:t[0],lambda *a:None,lambda *a:None,.059)
    assert sent and max(sent)<=.15
