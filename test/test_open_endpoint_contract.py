"""Open-end measurement is a region, not an exact repeatable encoder value."""
from copy import deepcopy
import pytest
from test_full_open_arrival import fixture,stable
from autolife_shelf_grasp_306.action_arrival import settled_full_open_arrival

@pytest.mark.parametrize('target',[10.,11.4331,12.8869,14.9])
@pytest.mark.parametrize('actual',[10.,11.4331,12.8869,14.9])
@pytest.mark.parametrize('start',[10.,12.8869,290.])
def test_open_endpoint_bidirectional_feedback_variation(target,actual,start):
 b,goal=fixture();n='Joint_Right_Gripper';goal[n]=target;b._current[n]=actual
 b._action_state['commanded_values'][n]=target;b._action_state['start_values'][n]=start
 before=deepcopy((goal,b._action_state['commanded_values']))
 assert stable(b,goal)
 assert (goal,b._action_state['commanded_values'])==before

@pytest.mark.parametrize('fault',['goal_outside','feedback_outside','closing','not_commanded','body','moving','stale','lost'])
def test_band_does_not_turn_obstruction_or_partial_closing_into_success(fault):
 b,goal=fixture();n='Joint_Right_Gripper';goal[n]=12.8869;b._current[n]=10.
 b._action_state['commanded_values'][n]=12.8869;b._action_state['start_values'][n]=10.
 if fault=='goal_outside':goal[n]=15.01;b._action_state['commanded_values'][n]=15.01
 if fault=='feedback_outside':b._current[n]=15.01
 if fault=='closing':goal[n]=290.;b._current[n]=285.2;b._action_state['commanded_values'][n]=290.
 if fault=='not_commanded':b._action_state['commanded_values'][n]=10.
 if fault=='body':b._current['Joint_Knee']+=1.01
 if fault=='moving':b._details[n]['speed']=5.01
 if fault=='stale':b._is_feedback_fresh=lambda:False
 if fault=='lost':b._details[n]['communication_lost']=True
 assert not stable(b,goal)

def test_no_success_from_reused_sample_or_drifting_encoder():
 b,goal=fixture();n='Joint_Right_Gripper';goal[n]=12.8869;b._current[n]=10.
 b._action_state['commanded_values'][n]=12.8869;b._action_state['start_values'][n]=10.
 for _ in range(1000):assert not settled_full_open_arrival(b,goal)
 for i in range(20):
  b._feedback_monotonic+=.1;b._current[n]=10.+(i%2)*.8
  assert not settled_full_open_arrival(b,goal)

@pytest.mark.parametrize('fault',[None,'drift','lost','body','stale'])
def test_final_confirmation_never_extends_motion_stream(tmp_path,monkeypatch,fault):
 import numpy as np
 from test_recorded_timing import fixture as track_fixture,Controller
 from autolife_shelf_grasp_306.teaching_track import NAMES,make_draft
 from autolife_shelf_grasp_306.teaching_motion import compile_action
 from autolife_shelf_grasp_306.teaching_replay import plan,advance
 d,t,q,entry=track_fixture(tmp_path,seconds=2.)
 q[:]=q[0];q[:,NAMES.index('Joint_Right_Gripper')]=12.8869
 entry=dict(zip(NAMES,map(float,q[0])))
 d=make_draft(tmp_path,'endpoint-generic',t,q);r,segs,_=plan(compile_action(d),entry,tmp_path)
 clock=[0.];monkeypatch.setattr('time.monotonic',lambda:clock[0]);c=Controller(r,segs,entry,clock)
 c._action_state.update(loop_index=0,start_values=entry.copy())
 c._details={n:dict(speed=0.,communication_lost=False) for n in entry};sent=[]
 def publish(state,desired,frame):
  sent.append(clock[0]);state['commanded_values'].update(desired)
 c._publish_action_targets=publish
 while c._action_state and clock[0]<3.:
  c._feedback_monotonic=clock[0]
  if clock[0]>=2.:
   c._current['Joint_Right_Gripper']=10.
   if fault=='drift':c._current['Joint_Right_Gripper']+=.8*(int(clock[0]*50)%2)
   if fault=='lost':c._communication_lost_joints=lambda n:['Joint_Right_Gripper']
   if fault=='body':c._current['Joint_Right_Elbow']+=1.1
   if fault=='stale':c.fresh=False
  advance(c);clock[0]+=.02
 assert max(sent)<=2.021
 if fault in (None,'drift'):
  assert c.outcome and c.outcome[1] and 2.<=clock[0]<=2.05
 else:assert c.reason and not c.outcome
