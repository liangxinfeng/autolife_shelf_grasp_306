import json,numpy as np,pytest
from autolife_shelf_grasp_306.demo_surface import fuse_stationary_depth
from autolife_shelf_grasp_306.demo_protocol import validate_place,check_base_anchor


def test_depth_fusion_preserves_missing_pixels_and_rejects_motion():
    frames=[np.full((24,24),1000.,dtype=float) for _ in range(5)]
    for i,f in enumerate(frames):
        f[0,0]=0
        f[1,1]=0 if i<3 else 1000
        f[2,2]=900 if i<2 else 1000
        f[3,3]=1000+[-5,5,-2,3,1][i]
    result=fuse_stationary_depth(frames,.001)
    assert result.dtype==np.uint16
    assert result[0,0]==result[1,1]==result[2,2]==0
    assert result[3,3]==1001
    assert result[4,4]==1000


def test_depth_fusion_does_not_erase_static_obstacle():
    frames=[np.full((24,24),1000.,dtype=float) for _ in range(5)]
    for f in frames:f[5:10,5:10]=850
    assert np.all(fuse_stationary_depth(frames,.001)[5:10,5:10]==850)


@pytest.mark.parametrize('count',[0,2,3,4,6])
def test_depth_fusion_requires_five_frames(count):
    with pytest.raises(ValueError):fuse_stationary_depth([np.ones((24,24))]*count,.001)


def test_base_anchor_requires_same_stationary_location():
    check_base_anchor([1,2,3.14],[1.005,2.001,-3.14])
    for current in ([1.03,2,3.14],[1,2,3.],[1,float('nan'),3.14]):
        with pytest.raises(ValueError):check_base_anchor(current,[1,2,3.14])


def test_observation_contract_never_accepts_release_or_wrong_grip(tmp_path):
    context=dict(grasp_plan_id='one',product_id='drink',settings={})
    record=dict(status='completed',plan_id='one',q={'Joint_Left_Gripper':160})
    for n,v in [('demo_run.json',context),('demo_grasp_result.json',record),('demo_base_result.json',{'status':'completed'})]:
        (tmp_path/n).write_text(json.dumps(v))
    plan=dict(kind='observe',plan_id='view1',place_protocol_version=1,observation_protocol_version=1,
              table_height_checked=True,parent_plan_id='one',product={'id':'drink'},demo_settings={},
              held_gripper_deg=160,source_q={'Joint_Left_Gripper':160},segments=[{'phase':'observe_lift'}])
    (tmp_path/'observation_attempt.json').write_text(json.dumps({'plan_id':'view1'}))
    validate_place(plan,tmp_path)
    plan['segments']=[{'phase':'place_lower'}]
    with pytest.raises(ValueError):validate_place(plan,tmp_path)
    plan['segments']=[{'phase':'observe_lift'}];plan['held_gripper_deg']=10
    with pytest.raises(ValueError):validate_place(plan,tmp_path)
    plan['held_gripper_deg']=160
    (tmp_path/'demo_observe_result.json').write_text('{}')
    with pytest.raises(ValueError,match='禁止重复'):validate_place(plan,tmp_path)
