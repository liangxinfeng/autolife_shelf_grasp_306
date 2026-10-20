from autolife_shelf_grasp_306.prepare import ready_after_place
from control_center_joint_control.joint_definitions import GROUPS

def setup():
    cfg=dict(initial_pose_deg={},gripper_open_deg=10.)
    q={'Joint_Left_Gripper':10.,'Joint_Right_Gripper':10.}
    for group in ('left_arm','right_arm','neck'):
        cfg['initial_pose_deg'][group]=[0.]*len(GROUPS[group]['joints'])
        q.update({n:0. for n,*_ in GROUPS[group]['joints']})
    return cfg,q

def test_receipt_requires_actual_reset_pose_and_open_jaws():
    cfg,q=setup();assert ready_after_place(cfg,q)
    for joint in q:
        assert not ready_after_place(cfg,{**q,joint:90.}),joint
        assert not ready_after_place(cfg,{**q,joint:float('nan')}),joint

def test_missing_config_or_feedback_not_ready():
    cfg,q=setup()
    assert not ready_after_place({},q)
    assert not ready_after_place(cfg,{})
