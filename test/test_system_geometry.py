import copy,json
from pathlib import Path
import numpy as np
import pytest
from autolife_handeye_calibration_306.core import Kinematics,has_solution
from autolife_handeye_calibration_306.verification_core import camera_transform,load_geometry,camera_geometry
from autolife_shelf_grasp_306.system_geometry import resolve_missing_calibration
from autolife_shelf_grasp_306.calibration_profiles import CalibrationProfiles
from autolife_shelf_grasp_306.arm_selection import tool_geometry

@pytest.fixture
def setup(tmp_path):
    urdf='/home/ubuntu/miniconda3/envs/robot_env/lib/python3.12/site-packages/autolife_robot_sdk/descriptions/autolife_s1/urdfs/robot_v2_2.urdf'
    fk=Kinematics(urdf)
    cfg={'robot_id':300,'arm':'left','calibration':str(tmp_path/'missing/session.json'),
        'right_calibration':'','head_offsets_deg':[0.,0.,0.],'tcp_offset_mm':15.,
        'T_wrist_tool_quat_xyzw':[-.70710678,0.,0.,.70710678],'hardware_execution':False}
    return tmp_path,fk,cfg

def test_missing_calibration_resolves_both_local_arms(setup):
    root,fk,cfg=setup;out=resolve_missing_calibration(cfg,fk,root)
    assert not Path(cfg['calibration']).exists() and out['hardware_execution'] is False
    profiles=CalibrationProfiles(out,fk)
    for side in ['left','right']:
        arm,data,sha=profiles.for_arm(side)
        assert data['geometry_source']=='system_urdf'
        assert data['config']['arm']==side and data['config']['robot_id']==300
        assert data['report']['calibrated'] is False and not has_solution(data['report'])
        assert 'samples' not in data and 'T_wrist_marker' not in data['report']
        tool,tcp=tool_geometry(arm,data,fk,side)
        assert tool['tool_geometry']['provisional'] and not tool['tool_geometry']['independent_tool_calibration']
        assert tool['tool_geometry']['camera_source']=='system_urdf'
        assert np.isfinite(tcp).all()
        q={v[0]:0. for v in fk.joints.values()}
        frame={'K':np.array([[600.,0.,320.],[0.,600.,240.],[0.,0.,1.]]),'D':np.zeros(5),'image':np.zeros((480,640,3),np.uint8)}
        transform=camera_transform(data,fk,q,frame,robot_id=300)
        assert np.isfinite(transform).all() and np.allclose(transform[3],[0,0,0,1])
        frame['K'][0,0]=0
        with pytest.raises(ValueError,match='内参'):camera_transform(data,fk,q,frame,robot_id=300)
        other=copy.deepcopy(data);other['config']['robot_id']=306
        with pytest.raises(ValueError,match='不匹配'):load_geometry(other,fk,15.,robot_id=300)

def test_existing_file_remains_authoritative_and_unchanged(setup):
    root,fk,cfg=setup;p=Path(cfg['calibration']);p.parent.mkdir();p.write_text('{"measured_fixture":true}')
    before=p.read_bytes();out=resolve_missing_calibration(cfg,fk,root)
    assert out==cfg and p.read_bytes()==before
    assert not (root/'data/system_geometry').exists()

def test_real_calibration_added_later_takes_priority(setup):
    root,fk,cfg=setup;first=resolve_missing_calibration(cfg,fk,root)
    p=Path(cfg['calibration']);p.parent.mkdir();p.write_text('{}')
    second=resolve_missing_calibration(cfg,fk,root)
    assert first['calibration']!=second['calibration'] and second['calibration']==str(p)

def test_generated_profile_hash_stable_and_model_specific(setup):
    root,fk,cfg=setup;a=resolve_missing_calibration(cfg,fk,root)
    b=resolve_missing_calibration(cfg,fk,root)
    assert a==b
    data,sha=CalibrationProfiles(a,fk).resolve(a)
    data2,sha2=CalibrationProfiles(b,fk).resolve(b)
    assert sha==sha2
    fk.sha='changed-model'
    with pytest.raises(ValueError,match='URDF'):load_geometry(data,fk,robot_id=300)

def test_invalid_existing_calibration_does_not_silently_fallback(setup):
    root,fk,cfg=setup;p=Path(cfg['calibration']);p.parent.mkdir();p.write_text('broken')
    out=resolve_missing_calibration(cfg,fk,root)
    with pytest.raises(json.JSONDecodeError):CalibrationProfiles(out,fk).resolve(out)
