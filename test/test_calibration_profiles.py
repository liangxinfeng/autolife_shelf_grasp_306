import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from autolife_shelf_grasp_306.calibration_profiles import CalibrationProfiles
from autolife_shelf_grasp_306 import arm_selection as a


@pytest.fixture
def profiles(tmp_path,monkeypatch):
    from autolife_handeye_calibration_306 import verification_core as v
    monkeypatch.setattr(v,'load_geometry',lambda *a,**kw:np.zeros(3))
    for side in ('left','right'):
        (tmp_path/(side+'.json')).write_text(json.dumps({'config':{'arm':side},
            'report':{'quality_grade':'needs_validation' if side=='right' else 'pass'}}))
    cfg={'arm':'left','calibration':str(tmp_path/'left.json'),'right_calibration':str(tmp_path/'right.json'),
         'tcp_offset_mm':15.,'target_offset_torso_mm':[0,15,0],'robot_id':306}
    return CalibrationProfiles(cfg,None)


def test_independent_profiles_preserve_left_and_do_not_copy_left_trim(profiles):
    left,ld,lh=profiles.for_arm('left');right,rd,rh=profiles.for_arm('right')
    assert left['calibration']==profiles.base['calibration']
    assert right['calibration']==profiles.base['right_calibration']
    assert ld['config']['arm']=='left' and rd['config']['arm']=='right' and lh!=rh
    assert left['target_offset_torso_mm']==[0,15,0]
    assert right['target_offset_torso_mm']==[0,0,0]
    assert right['calibration_profile']['test_only']
    assert not left['calibration_profile']['test_only']


def test_changed_calibration_rejected(profiles):
    cfg,_,_=profiles.for_arm('right')
    p=Path(cfg['calibration']);p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='改变'):profiles.resolve(cfg)


def test_wrong_arm_file_rejected(profiles):
    profiles.base['right_calibration']=profiles.base['calibration']
    with pytest.raises(ValueError,match='手臂'):profiles.for_arm('right')


@pytest.mark.parametrize('side',['left','right'])
def test_held_profile_uses_own_hash_and_rejects_cross_arm(profiles,side):
    cfg,data,digest=profiles.for_arm(side);cfg['arm']=side
    plan={'config':cfg,'arm':side,'calibration_sha256':digest,'tcp_offset_mm':15.}
    assert profiles.for_held_plan(plan)==(data,digest)
    plan['calibration_sha256']='bad'
    with pytest.raises(ValueError,match='持物记录'):profiles.for_held_plan(plan)


def test_legacy_profile_still_supported(profiles):
    profiles.base.pop('right_calibration')
    cfg,data,_=profiles.for_arm('right')
    assert data['config']['arm']=='left'


@pytest.mark.parametrize('fail_right',[False,True])
def test_pool_plans_arm_specific_point_and_product(profiles,monkeypatch,fail_right):
    monkeypatch.setattr(a,'rank_arms',lambda *a:['right','left'])
    monkeypatch.setattr(a,'tool_geometry',lambda cfg,data,fk,side:({**cfg,'arm':side,'tool_geometry':{'source_arm':data['config']['arm']}},np.zeros(3)))
    observations=[];calls=[]
    class Planner:
        def __init__(self,cfg):self.cfg=cfg
        def plan(self,q,tcp,point,product):
            calls.append((self.cfg['arm'],point.tolist(),product))
            return {'product':product}
    def observe(side,cfg):
        observations.append(side)
        if side=='right' and fail_right:raise ValueError('right depth unavailable')
        return np.array([.4,.2 if side=='right' else -.2,.1]),{'arm_shape':side}
    pool=a.ArmPlannerPool(profiles.base,{},None,Planner,profiles)
    result=pool.plan({},[.4,.2,.1],{'wrong':'shared'}, {'up_torso':[0,0,1]},observation_for_arm=observe)
    side='left' if fail_right else 'right'
    assert result[0]['arm']==side and calls[0][0]==side
    assert calls[0][2]=={'arm_shape':side}
    assert calls[0][1][1]==pytest.approx(-.185 if fail_right else .2)
    assert observations==(['right','left'] if fail_right else ['right'])
    assert result[0]['tool_geometry']['source_arm']==side


def test_real_306_camera_and_tcp():
    """Offline FK only; never opens ROS, camera, driver or command publishers."""
    import yaml
    from autolife_handeye_calibration_306.core import Kinematics
    from autolife_handeye_calibration_306.verification_core import load_geometry,camera_transform
    root=Path('/home/ubuntu/ros2_ws/src/autolife_shelf_grasp_306')
    if not root.exists():pytest.skip('306 offline fixture not available')
    cfg=yaml.safe_load((root/'config/grasp.yaml').read_text())
    cfg['right_calibration']='/home/ubuntu/ros2_ws/src/autolife_handeye_calibration_306/sessions/20261012_172415_700141/session.json'
    fk=Kinematics(cfg['urdf_path']);profiles=CalibrationProfiles(cfg,fk)
    base=json.loads(Path(cfg['calibration']).read_text())
    left,ld,_=profiles.for_arm('left');right,rd,_=profiles.for_arm('right')
    assert ld==base
    for side,conf,data in [('left',left,ld),('right',right,rd)]:
        sc,tcp=a.tool_geometry(conf,data,fk,side)
        marker=np.asarray(data['report']['T_wrist_marker'])
        assert np.allclose(tcp,marker[:3,3]+marker[:3,:3]@np.array([0.,0.,-.015]))
        assert not sc['tool_geometry']['provisional']
        assert np.allclose(load_geometry(data,fk,15,robot_id=306),tcp)
    assert right['calibration_profile']['test_only']
    assert rd['report']['fixed_head_pass'] is False
    assert rd['report']['moving_head_pass'] is False
    rs=next(iter(rd['samples'].values()));ls=next(iter(ld['samples'].values()))
    assert rs['image_shape']==ls['image_shape']==[720,1280,3]
    assert np.allclose(rs['K'],ls['K'],rtol=0,atol=1e-5)
    assert np.allclose(rs['D'],ls['D'],rtol=0,atol=1e-7)
    frame={'image':np.zeros(rs['image_shape'],dtype=np.uint8),'K':np.array(rs['K']),'D':np.array(rs['D'])}
    rt=camera_transform(rd,fk,rs['q_deg'],frame,robot_id=306)
    lt=camera_transform(ld,fk,rs['q_deg'],frame,robot_id=306)
    assert np.allclose(rt,np.array(rs['H'])@np.array(rd['report']['T_mount_camera']))
    assert not np.allclose(rt,lt)
    print('REAL_PROFILE_GEOMETRY',json.dumps({'right_tcp_mm':(load_geometry(rd,fk,15,robot_id=306)*1000).tolist(),
        'camera_translation_difference_mm':float(np.linalg.norm(rt[:3,3]-lt[:3,3])*1000),
        'right_profile':right['calibration_profile'],'left_profile':left['calibration_profile']}))


def test_worker_binds_profile_before_placement_and_freezes_overlay():
    import ast
    import autolife_shelf_grasp_306.worker as worker
    source=Path(worker.__file__).read_text();ast.parse(source)
    assert source.index('profiles.for_held_plan(grasp)')<source.index("cfg.update({k:v for k,v in grasp['config'].items()")
    assert 'camera_transform(path_calibration,display_fk,current,frame' in source
    assert 'last_T=camera_transform(data,fk,observed_q,observed' in source
    assert 'observation_for_arm=observation_for_arm' in source
    assert "_,geometry,selected['product'],T=arm_observations[cfg['arm']]" in source


def test_worker_observation_callback_recomputes_geometry(tmp_path):
    """Execute the actual nested worker function with deterministic RGB-D doubles."""
    import ast
    import autolife_shelf_grasp_306.worker as worker
    tree=ast.parse(Path(worker.__file__).read_text())
    function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='observation_for_arm')
    program=compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'worker_callback','exec')
    transforms=[];estimates=[]
    def transform(data,fk,q,frame,**kwargs):
        t=np.eye(4);t[0,3]=.03;transforms.append(data['arm']);return t
    def estimate(frame,mask,T,up,product,cfg,reference):
        estimates.append(T.copy());return np.array([T[0,3]+.4,.2,.1]),{'shape':'right-shape'}
    ns=dict(profiles=SimpleNamespace(resolve=lambda cfg:({'arm':cfg['arm']},cfg['arm'])),
        calibration_sha='left',fused=(np.array([.4,-.2,.1]),{}),geometry={'shape':'left-shape'},
        selected={'product':{'name':'left-product'}},T=np.eye(4),copy=copy,
        arm_observations={},run=tmp_path,paired_frames=[({}, {},[0,0,1]) for _ in range(5)],
        camera_transform=transform,fk=None,cfg={'robot_id':306},temporary=True,mask=None,
        measured_product={'name':'template'},dimension_reference=None,estimate_temporary=estimate,
        DepthQualityError=ValueError,combine_product=lambda template,values:{'name':'right-product',
            'dimension_inlier_indices':list(range(5)),'dimension_source':'test'},
        combine=lambda samples,indices:(np.mean([s[0] for s in samples],axis=0),dict(samples[0][1])),
        planning_product=lambda product,stats:{**product,**stats})
    exec(program,ns)
    point,product=ns['observation_for_arm']('right',{'arm':'right'})
    assert np.allclose(point,[.43,.2,.1])
    assert transforms==['right']*5 and len(estimates)==5
    assert product['shape']=='right-shape' and product['name']=='right-product'
    point,product=ns['observation_for_arm']('left',{'arm':'left'})
    assert np.allclose(point,[.4,-.2,.1])
    assert len(estimates)==5 and product['shape']=='left-shape'
