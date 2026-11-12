import json,math
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from autolife_shelf_grasp_306.grasp_gripper import execute
from autolife_shelf_grasp_306.pose_geometry import ModelGeometry
from scipy.spatial.transform import Rotation

def old_geometry():
    # 原版本逐连杆公式作独立数值基准，测试不依赖现场记录或临时旧代码。
    class Reference(ModelGeometry):
        def frames(self,values):
            frames={self.root:np.eye(4)};todo=list(self.joints)
            while todo:
                ready=[j for j in todo if j['parent'] in frames]
                if not ready:raise ValueError('invalid tree')
                for j in ready:
                    t=j['origin'].copy();m=j['mimic']
                    angle=(math.radians(values.get(m['joint'],0))*float(m.get('multiplier',1))+float(m.get('offset',0))
                           if m else math.radians(values.get(j['name'],0)))
                    if 'Gripper' in j['name']:angle=0.
                    if j['kind'] in ('revolute','continuous'):
                        t[:3,:3]=t[:3,:3]@Rotation.from_rotvec(j['axis']/np.linalg.norm(j['axis'])*angle).as_matrix()
                    elif j['kind']=='prismatic':t[:3,3]+=j['axis']*values.get(j['name'],0.)
                    frames[j['child']]=frames[j['parent']]@t;todo.remove(j)
            return frames
    return Reference

@pytest.mark.parametrize('seed',range(12))
def test_all_official_frames_equal_original(seed):
    urdf=Path(__file__).resolve().parents[1]/'urdf/robot_v3_1.urdf'
    old=old_geometry()(urdf);new=ModelGeometry(urdf)
    rng=np.random.default_rng(seed)
    values={j['name']:float(rng.uniform(-150,150)) for j in old.joints}
    a=old.frames(values);b=new.frames(values)
    assert a.keys()==b.keys()
    for name in a:np.testing.assert_allclose(a[name],b[name],atol=1e-12,rtol=1e-12)

def test_mimic_prismatic_and_nominal_fingers(tmp_path):
    p=tmp_path/'m.urdf'
    p.write_text('''<robot name="test"><link name="root"/><link name="a"/><link name="b"/><link name="c"/><link name="Link_Camera_Head_Forehead"/>
    <joint name="slide" type="prismatic"><parent link="b"/><child link="c"/><axis xyz="1 0 0"/><origin rpy="0 .5 .1"/></joint>
    <joint name="Joint_Gripper" type="revolute"><parent link="c"/><child link="Link_Camera_Head_Forehead"/><axis xyz="0 1 0"/></joint>
    <joint name="first" type="continuous"><parent link="root"/><child link="a"/><axis xyz="0 0 2"/></joint>
    <joint name="follow" type="revolute"><parent link="a"/><child link="b"/><axis xyz="1 1 0"/><mimic joint="first" multiplier="-2" offset=".2"/></joint></robot>''')
    values=dict(first=71,follow=12,slide=.2,Joint_Gripper=330)
    a=old_geometry()(p).frames(values);b=ModelGeometry(p).frames(values)
    for name in a:np.testing.assert_allclose(a[name],b[name],atol=1e-12)

@pytest.mark.parametrize('arm',['left','right'])
@pytest.mark.parametrize('failure',[None,'blocked','stale','stop'])
def test_preopening_uses_measured_clearance_and_keeps_other_hand(tmp_path,arm,failure):
    path=tmp_path/'curve.json';path.write_text(json.dumps(dict(robot_id=345,hardware='dm_type1',measured=True,
        arms={arm:[[5,330],[45,180],[90,10]]})))
    cfg=dict(arm=arm,gripper_width_calibration=str(path),gripper_speed_scale=.5)
    jaw='Joint_'+arm.title()+'_Gripper';other='Joint_'+('Left' if arm=='right' else 'Right')+'_Gripper'
    q={jaw:292.4,other:22.};sent=[];holds=[];clock=[0.];reports=[]
    node=NS(_active_goals={},progress_watch={},_gripper_pub=object())
    def publish(pub,data):
        assert data[('left' if arm=='right' else 'right')+'_gripper_target_joints_position']==[22.]
        sent.append((clock[0],data[arm+'_gripper_target_joints_position'][0]))
    node._publish_vendor=publish
    def checked():
        clock[0]+=.02
        if clock[0]>.3 and failure in ('stale','stop'):raise ValueError(failure)
        if sent and failure!='blocked':q[jaw]+=float(np.clip(sent[-1][1]-q[jaw],-2.4,2.4))
        return dict(q)
    state=dict(targets={jaw:10.,other:22.},closing=False,tolerance=2.,required_opening_mm=60.)
    args=(node,cfg,arm,state,checked,lambda:clock[0],lambda *a:holds.append(a),reports.append,.05)
    if failure:
        with pytest.raises(ValueError):execute(*args)
    else:
        assert execute(*args)=='opening_clear'
        width=float(np.interp(q[jaw],[10,180,330],[90,45,5]))
        assert width>=60 and q[jaw]>12 and clock[0]<3.6
        assert holds and reports[-1]['required_opening_mm']==60
    assert sent
    assert max(abs((b[1]-a[1])/(b[0]-a[0])) for a,b in zip(sent,sent[1:]))<=120.01
