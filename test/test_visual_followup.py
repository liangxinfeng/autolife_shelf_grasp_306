import copy,json
from types import SimpleNamespace as NS
import numpy as np
import pytest
from autolife_shelf_grasp_306.visual_reference import ReferenceStore
from autolife_shelf_grasp_306.visual_multiview import prepare_view,resolve_views
from autolife_shelf_grasp_306.visual_registration import register_views
from autolife_shelf_grasp_306.teaching_trajectory import build,evaluate
from autolife_shelf_grasp_306.pose_synchronized_output import progress,publish_body,before_tick
from control_center_joint_control.joint_definitions import JOINT_LIMITS,GROUP_ORDER


def q():return {n:max(lo,min(hi,0.)) for n,(lo,hi) in JOINT_LIMITS.items()}

def fixture():
    rng=np.random.default_rng(9);pts=rng.uniform(-.07,.07,(40,3));desc=rng.uniform(0,200,(40,128)).astype('float32')
    a=dict(points=pts,cloud=pts.copy(),pixels=rng.uniform(0,300,(40,2)),descriptors=desc)
    m=dict(name='水瓶',object_pose=np.eye(4).tolist(),camera_pose=np.eye(4).tolist(),created=1.)
    return m,a

@pytest.mark.parametrize('shift',[.05,.3,.8])
def test_moved_supplement_keeps_first_anchor_and_original_bytes(tmp_path,shift):
    m,a=fixture();store=ReferenceStore(tmp_path);ident=store.save(m,a);raw=(store.directory/(ident+'.npz')).read_bytes()
    b=copy.deepcopy(a);b['points']+=np.array([shift,.04,.02]);b['cloud']+=np.array([shift,.04,.02]);new=copy.deepcopy(m);new['created']=2.;new['object_pose'][0][3]=shift
    meta,arrays=prepare_view(m,store.views(ident),new,b);assert meta['metric_registered']
    store.add_view(ident,meta,arrays)
    assert (store.directory/(ident+'.npz')).read_bytes()==raw
    assert meta['object_pose']==m['object_pose'] and meta['captured_object_pose'][0][3]==shift
    assert np.max(np.abs(arrays['points']-a['points']))<1e-10
    t,stats,*_=register_views(store.views(ident),b);assert abs(t[0,3]-shift)<1e-8
    assert len(store.list())==1 and store.list()[0]['id']==ident


def test_unknown_surface_saved_without_fake_transform(tmp_path,monkeypatch):
    m,a=fixture();store=ReferenceStore(tmp_path);ident=store.save(m,a)
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_multiview.register_views',lambda *a:(_ for _ in ()).throw(ValueError('no overlap')))
    b=copy.deepcopy(a);b['points']+=1;b['cloud']+=1
    meta,arrays=prepare_view(m,store.views(ident),m,b);store.add_view(ident,meta,arrays)
    assert not meta['metric_registered'];assert np.array_equal(arrays['points'],b['points'])
    assert meta['object_pose']==m['object_pose'] and len(store.views(ident))==2


def test_later_overlap_links_previously_unregistered_view_without_writing(monkeypatch):
    m,a=fixture();pending={**m,'metric_registered':False,'created':2.}
    fake=copy.deepcopy(a);fake['points']+=.5
    calls=[]
    def prepared(original,views,meta,arr):
        calls.append(meta['created']);return dict(meta,metric_registered=True),a
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_multiview.prepare_view',prepared)
    result=resolve_views([(m,a),(pending,fake)])
    assert all(x[0].get('metric_registered',True) for x in result) and not pending['metric_registered']


def state_segment():
    start=q();target=dict(start);target['Joint_Right_Shoulder_Inner']=30.;target['Joint_Right_Elbow']=-20.;target['Joint_Right_Wrist_Lower']=-10.
    record=dict(groups=list(GROUP_ORDER),speed_percent=100.,loop_count=1,keyframes=[dict(joints_deg=target,transition_sec=3.,hold_sec=.3,easing='smooth')])
    record,segments=build(record,start)
    state=dict(record=record,commanded_values=dict(start),stage='transition',frame_index=0,loop_index=0,frame_deadline=100.)
    node=NS(_current=dict(start),get_parameter=lambda _:NS(value=2.),_controller_target=dict(start),_whole_body_pub=None)
    return node,state,segments[0]


def test_slowest_joint_limits_whole_curve_not_individual_axes():
    b,state,s=state_segment()
    desired=progress(b,state,s,.8)
    # Current shoulder limits advancement to 2deg; elbow and wrist cannot race ahead.
    assert 0<=desired['Joint_Right_Shoulder_Inner']<=2.000001
    assert np.isclose((desired['Joint_Right_Elbow']-s['start']['Joint_Right_Elbow'])/(-20-s['start']['Joint_Right_Elbow']),desired['Joint_Right_Shoulder_Inner']/30)
    assert np.isclose((desired['Joint_Right_Wrist_Lower']-s['start']['Joint_Right_Wrist_Lower'])/(-10-s['start']['Joint_Right_Wrist_Lower']),desired['Joint_Right_Shoulder_Inner']/30)
    assert desired==evaluate(s,state['progress'])


def test_synchronized_stream_reaches_exact_target_with_different_motor_lags():
    b,state,s=state_segment();u=0.
    for _ in range(4000):
        desired=progress(b,state,s,min(1.,u+.02/s['seconds']));u=state['progress']
        for n in b._current:b._current[n]+=(desired[n]-b._current[n])*(.06 if 'Shoulder' in n else .3)
        assert desired==evaluate(s,u)
        if u>=1:break
    assert u==1 and desired==evaluate(s,1)


def test_joint_fault_cannot_advance_curve():
    b,state,s=state_segment();b._current['Joint_Right_Elbow']=float('nan')
    with pytest.raises(ValueError):progress(b,state,s,.5)


def test_no_wall_clock_catchup_and_deadline_preserved(monkeypatch):
    b,state,s=state_segment();b._action_state=state;b.teaching_segments=[s];cancel=[];b._cancel_action=lambda reason:cancel.append(reason)
    clock=[10.];monkeypatch.setattr('time.monotonic',lambda:clock[0]);before_tick(b)
    state['synchronized_clock']['progress']=.2;clock[0]=12.;before_tick(b)
    assert (12.-state['frame_started'])/s['seconds']<=.2+.0400001/s['seconds']
    assert state['frame_deadline']==100.
    clock[0]=101.;before_tick(b);assert cancel


def test_recorded_timing_is_not_changed(monkeypatch):
    b,state,s=state_segment();s['recorded_timing']=True;b._action_state=state;b.teaching_segments=[s]
    before_tick(b);assert 'synchronized_clock' not in state


def test_invalid_output_is_rejected_before_any_publish():
    b,state,s=state_segment();sent=[];b._publish_vendor=lambda *a:sent.append(a)
    desired=dict(s['target']);desired['Joint_Right_Elbow']=float('nan')
    with pytest.raises(ValueError):publish_body(b,state,desired,{})
    assert sent==[]


def test_multiaxis_ik_uses_shared_executor_and_single_joint_stays_manual():
    from autolife_shelf_grasp_306.pose_workspace_motion import begin
    calls=[];e=NS(moving=lambda:False,job=None)
    def execute(*args,**kwargs):calls.append((args,kwargs));e.job={}
    e.execute=execute
    start=q();begin(e,'ik',{'Joint_Right_Elbow':-20.,'Joint_Right_Shoulder_Inner':10.},start)
    assert calls[0][1]['keep_session'] and e.job['selected_targets']['Joint_Right_Elbow']==-20.


def test_rotated_and_translated_view_keeps_reference_frame(tmp_path):
    from scipy.spatial.transform import Rotation
    m,a=fixture();r=Rotation.from_euler('z',35,degrees=True).as_matrix();shift=np.array([.35,.1,.02])
    b=copy.deepcopy(a);b['points']=a['points']@r.T+shift;b['cloud']=a['cloud']@r.T+shift
    meta,aligned=prepare_view(m,[(m,a)],m,b)
    assert meta['metric_registered'] and meta['object_pose']==m['object_pose']
    assert np.max(np.abs(aligned['points']-a['points']))<1e-9
    t,stats,*_=register_views([(m,a),(meta,aligned)],b)
    assert np.allclose(t[:3,:3],r) and np.allclose(t[:3,3],shift)


def test_textureless_supplement_cannot_fake_geometric_match():
    m,a=fixture();b=copy.deepcopy(a);b.update(points=np.empty((0,3)),pixels=np.empty((0,2)),descriptors=np.empty((0,128),np.float32))
    meta,arrays=prepare_view(m,[(m,a)],m,b)
    assert not meta['metric_registered'] and len(arrays['cloud'])==40
    with pytest.raises(ValueError):register_views([(meta,arrays)],a)
