"""Vendor protective-band mismatch regression. No motors, ROS or networking."""
import copy
import json
import math
import pytest
from autolife_shelf_grasp_306.joint_limits import load_arm_policy,check_pose,validate_grasp_plan


def fixture(tmp_path,arm='left'):
    name=f'Joint_{arm.title()}_Shoulder_Outer'
    lo,hi=(-3.1415,.3141) if arm=='left' else (-.3141,3.1415)
    model=tmp_path/'model.urdf'
    model.write_text(f'<robot name="test"><joint name="{name}"><limit lower="{lo}" upper="{hi}"/></joint></robot>')
    vendor=tmp_path/'vendor.json'
    vendor.write_text(json.dumps({'MOTOR_DEAD_ZONE':{arm.upper()+'_ARM_LOWER_DEAD_ZONE':[1.],
                                                      arm.upper()+'_ARM_UPPER_DEAD_ZONE':[1.]}}))
    cfg={'urdf_path':str(model),'vendor_control_config':str(vendor),'arm':arm}
    limits={name:(-180.,180.)}
    return cfg,name,limits


@pytest.mark.parametrize('arm',['left','right'])
def test_actual_boundary_and_mirror(tmp_path,arm):
    cfg,n,ui=fixture(tmp_path,arm);policy=load_arm_policy(cfg,[n],ui)
    lo,hi=policy['bounds_deg'][n]
    if arm=='left':
        assert hi==pytest.approx(math.degrees(.3141)-1.5)
        # Incident: goal17.9224, actual17.03955, guard~16.9966.
        with pytest.raises(ValueError,match='保护区'):check_pose(policy,{n:17.922439347})
        check_pose(policy,{n:16.49})
    else:
        assert lo==pytest.approx(-math.degrees(.3141)+1.5)
        with pytest.raises(ValueError,match='保护区'):check_pose(policy,{n:-17.92})
        check_pose(policy,{n:-16.49})


def test_ui_limits_not_expanded_and_bad_input_rejected(tmp_path):
    cfg,n,ui=fixture(tmp_path);ui[n]=[-90,10]
    policy=load_arm_policy(cfg,[n],ui)
    assert policy['bounds_deg'][n]==[-90,10]
    for value in (float('nan'),float('inf'),10.1):
        with pytest.raises(ValueError):check_pose(policy,{n:value})


def test_stale_and_protected_plan_rejected(tmp_path):
    cfg,n,ui=fixture(tmp_path);policy=load_arm_policy(cfg,[n],ui)
    plan={'config':cfg,'joint_limit_policy':policy,'source_q':{n:0.},
          'segments':[{'waypoints':[{n:0.},{n:16.4}]}]}
    validate_grasp_plan(plan,[n],ui)
    bad=copy.deepcopy(plan);bad['segments'][0]['waypoints'][1][n]=17.92
    with pytest.raises(ValueError,match='保护区'):validate_grasp_plan(bad,[n],ui)
    bad=copy.deepcopy(plan);bad.pop('joint_limit_policy')
    with pytest.raises(ValueError,match='旧规划'):validate_grasp_plan(bad,[n],ui)
    # A limit file change invalidates previously approved paths, even if wider.
    from pathlib import Path
    path=Path(cfg['vendor_control_config']);data=json.loads(path.read_text())
    data['MOTOR_DEAD_ZONE']['LEFT_ARM_UPPER_DEAD_ZONE']=[.5];path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='配置已改变'):validate_grasp_plan(plan,[n],ui)


def test_execution_limit_validation_precedes_hardware_access():
    from pathlib import Path
    from autolife_shelf_grasp_306 import execution
    text=Path(execution.__file__).read_text()
    assert text.index('limit_policy=validate_grasp_plan')<text.index('rclpy.init(args=ros_args)')
    assert 'check_pose(limit_policy,command)' in text
