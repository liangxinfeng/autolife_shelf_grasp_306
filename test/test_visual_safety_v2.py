import json
from types import SimpleNamespace as NS
import numpy as np
import pytest
from autolife_shelf_grasp_306.visual_reference import ReferenceStore
from autolife_shelf_grasp_306.pose_gripper_intent import stable_closing,manual_tick
from control_center_joint_control.joint_definitions import ALL_JOINTS,JOINT_LIMITS


def q():return {n:max(lo,min(hi,0.)) for n,(lo,hi) in JOINT_LIMITS.items()}


def test_removed_reference_remains_resolvable_and_views_do_not_pollute_list(tmp_path):
    s=ReferenceStore(tmp_path);meta=dict(name='瓶子',object_pose=np.eye(4).tolist());a=dict(cloud=np.ones((30,3)))
    first=s.save(meta,a);child=s.add_view(first,meta,a)
    assert len(s.views(first))==2 and len(s.list())==1
    s.remove(first);assert s.list()==[]
    assert s.load(first)[0]['name']=='瓶子' and s.load(child)[0]['parent_reference']==first


def node():
    pose=q();pose['Joint_Right_Gripper']=180.
    return NS(_current=pose,_details={n:dict(speed=0.,communication_lost=False) for n in pose},
        _is_feedback_fresh=lambda:True,_feedback_monotonic=1.,get_parameter=lambda n:NS(value=.2))


def test_clamp_target_does_not_require_crushing_object_to_requested_width():
    b=node();n='Joint_Right_Gripper';state={}
    for t in (1.,1.1,1.2,1.3,1.41):
        b._feedback_monotonic=t;result=stable_closing(b,state,{n:220.},{n:220.},{n:10.})
    assert result and b._current[n]==180.


@pytest.mark.parametrize('fault',['offline','not_sent','arm_error','opening'])
def test_closing_completion_does_not_hide_other_faults(fault):
    b=node();n='Joint_Right_Gripper';target={n:220.};command={n:220.};start={n:10.}
    if fault=='offline':b._details[n]['communication_lost']=True
    elif fault=='not_sent':command[n]=190.
    elif fault=='opening':target[n]=command[n]=100.;start[n]=250.
    else:
        joint='Joint_Right_Shoulder_Inner';target[joint]=b._current[joint]+10;command[joint]=target[joint];start[joint]=b._current[joint]
    state={}
    for t in np.arange(1.,2.,.05):
        b._feedback_monotonic=t;assert not stable_closing(b,state,target,command,start)


def test_manual_clamp_finishes_but_keeps_target_and_other_hand(monkeypatch):
    clock=[10.];monkeypatch.setattr('time.monotonic',lambda:clock[0])
    b=node();b._hardware_ready=True;b._hardware_enabled=True;b._communication_lost_joints=lambda _:[];b._gripper_pub=object()
    sent=[];receipts=[];b._publish_vendor=lambda pub,packet:sent.append(packet)
    n='Joint_Right_Gripper'
    e=NS(job=dict(targets={n:220.},speed=.5,token='grip'),bridge=b,gripper_targets={'Joint_Left_Gripper':150.},
        panel=NS(setEnabled=lambda v:None),completed=NS(emit=lambda *a:receipts.append(a)))
    for t in np.arange(10.,30.,.05):
        clock[0]=float(t);b._feedback_monotonic=float(t);manual_tick(e)
        if e.job is None:break
    assert e.job is None and receipts[-1][1]
    assert e.gripper_targets[n]==220. and b._current[n]==180.
    assert all(v['left_gripper_target_joints_position']==[150.] for v in sent)
    values=[v['right_gripper_target_joints_position'][0] for v in sent]
    assert values[-1]==220. and all(a<=b for a,b in zip(values,values[1:]))


def test_readonly_refresh_waits_for_ros_after_real_hardware_response(monkeypatch):
    from autolife_shelf_grasp_306.pose_feedback_read import PoseFeedbackRead
    from control_center_joint_control.joint_definitions import group_joint_names
    clock=[1.];monkeypatch.setattr('time.monotonic',lambda:clock[0]);values=q()
    groups={g:group_joint_names(g) for g in ('neck','leg_waist','left_arm','right_arm')}
    groups.update(left_gripper=['Joint_Left_Gripper'],right_gripper=['Joint_Right_Gripper'])
    destroyed=[]
    def create(kind,name):
        g=next(g for g in groups if name.endswith('_'+g))
        response=NS(success=True,message=json.dumps(dict(protocol=2,group=g,positions={n:values[n] for n in groups[g]})))
        return NS(service_is_ready=lambda:True,call_async=lambda _:NS(done=lambda:True,result=lambda:response))
    b=NS(create_client=create,destroy_client=lambda c:destroyed.append(c),_current=dict(values),_feedback_monotonic=0.,_is_feedback_fresh=lambda:True)
    r=PoseFeedbackRead(b);assert r.poll() is None;assert r.poll() is None
    b._feedback_monotonic=.99;assert r.poll() is None
    b._feedback_monotonic=1.1;b._current['Joint_Right_Elbow']+=20;assert r.poll() is None
    b._current=dict(values);assert r.poll()==values;r.close();assert len(destroyed)==6


def test_save_pose_keeps_intended_width_not_loaded_measurement(tmp_path):
    from PyQt5 import QtWidgets as Q
    from autolife_shelf_grasp_306.app import Window
    app=Q.QApplication.instance() or Q.QApplication([])
    root=tmp_path/'root';root.mkdir();(root/'data').mkdir();run=root/'run';run.mkdir()
    w=Window(root,run,start_worker=False);w.timer.stop();w.busy=False;page=w.pose_workspace
    try:
        actual=q();actual['Joint_Right_Gripper']=180.;page.capture=lambda:dict(actual)
        w.studio.engine.gripper_targets={'Joint_Right_Gripper':220.}
        page.pose_name.setText('夹瓶意图');page.save_source.setCurrentIndex(0);page.save_pose()
        saved=page.store.read()['夹瓶意图'];assert saved['Joint_Right_Gripper']==220.
        assert all(saved[n]==v for n,v in actual.items() if n!='Joint_Right_Gripper')
    finally:w.studio.editor.dirty=False;page.dirty=False;w.close();app.processEvents()
