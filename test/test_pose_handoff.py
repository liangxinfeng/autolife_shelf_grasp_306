import ast
import json
from pathlib import Path
import pytest
from autolife_shelf_grasp_306.pose_handoff import start_pose_diagnostic


SOURCE={'Joint_Left_Elbow':110.,'Joint_Right_Elbow':-110.,
        'Joint_Neck_Pitch':-20.,'Joint_Waist_Pitch':0.,
        'Joint_Left_Gripper':157.4573875,'Joint_Right_Gripper':11.25}


@pytest.mark.parametrize('kind',['observe','place'])
@pytest.mark.parametrize('stage',['执行前','使能期间','SYNC切换期间'])
def test_loaded_jaw_settling_not_arm_pose_change(kind,stage):
    current={**SOURCE,'Joint_Left_Gripper':156.605}
    old=dict(SOURCE)
    report=start_pose_diagnostic(current,SOURCE,stage,kind)
    assert report['ok'] and SOURCE==old
    assert next(c for c in report['changes'] if c['joint']=='Joint_Left_Gripper')['tolerance_deg']==8.


@pytest.mark.parametrize('name',[n for n in SOURCE if n!='Joint_Left_Gripper'])
def test_other_joint_limits_unchanged(name):
    report=start_pose_diagnostic({**SOURCE,name:SOURCE[name]+.51},SOURCE,'SYNC切换期间','place')
    assert not report['ok'] and name in report['message']
    assert report['violations'][0]['tolerance_deg']==.5


@pytest.mark.parametrize('kind',[None,'grasp','prepare'])
def test_unloaded_operations_retain_original_jaw_limit(kind):
    assert not start_pose_diagnostic({**SOURCE,'Joint_Left_Gripper':156.6},SOURCE,'执行前',kind)['ok']


def test_large_loaded_jaw_change_is_rejected_and_named():
    r=start_pose_diagnostic({**SOURCE,'Joint_Left_Gripper':SOURCE['Joint_Left_Gripper']+8.01},SOURCE,'SYNC切换期间','place')
    assert not r['ok'] and '持瓶夹爪' in r['message'] and '8.01' in r['message']


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_invalid_values_fail(value):
    with pytest.raises(ValueError):start_pose_diagnostic({**SOURCE,'Joint_Left_Elbow':value},SOURCE,'执行前','place')
    with pytest.raises(ValueError):start_pose_diagnostic(SOURCE,{**SOURCE,'Joint_Left_Elbow':value},'执行前','place')


def test_missing_or_empty_feedback_fails():
    with pytest.raises(ValueError):start_pose_diagnostic({},SOURCE,'执行前','place')
    with pytest.raises(ValueError):start_pose_diagnostic(SOURCE,{},'执行前','place')


def test_no_rebase_across_stages():
    for drift in (.2,.4):
        assert start_pose_diagnostic({**SOURCE,'Joint_Left_Elbow':110+drift},SOURCE,'使能期间','observe')['ok']
    assert not start_pose_diagnostic({**SOURCE,'Joint_Left_Elbow':110.6},SOURCE,'SYNC切换期间','observe')['ok']


def test_executor_preserves_order_and_all_three_checks():
    path=Path(__file__).parents[1]/'autolife_shelf_grasp_306'/'execution.py'
    text=path.read_text(encoding='utf-8');ast.parse(text)
    assert text.index("check_start_pose('执行前')")<text.index('response=node.unlock()')
    assert text.index("check_start_pose('使能期间')")<text.index('session.enter(spin)')
    assert text.index('session.enter(spin)')<text.index("check_start_pose('SYNC切换期间')")<text.index('def checked():')


def test_diagnostics_are_serializable():
    assert json.loads(json.dumps(start_pose_diagnostic(SOURCE,SOURCE,'执行前','observe')))['ok']
