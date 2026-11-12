from types import SimpleNamespace
import pytest
from autolife_shelf_grasp_306 import reset_collision_scope as mod
from autolife_shelf_grasp_306.grasp_collision import STATIC_NECK_PAIR,GROUPS

HEAD='Link_Waist_Yaw_to_Shoulder_Inner_0 <-> Link_Neck_Yaw_to_Head_0'
NECK=' <-> '.join(sorted(STATIC_NECK_PAIR))
OTHER='left_arm <-> right_arm'

def make_guard(tmp_path,monkeypatch,initial=None,final=None):
 p=tmp_path/'model';p.write_text('model')
 monkeypatch.setattr(mod,'MODEL_SHA256',mod.hashlib.sha256(p.read_bytes()).hexdigest())
 source={n:0. for names in GROUPS.values() for n in names};goal={**source,'Joint_Neck_Pitch':10.}
 def pairs(q):return (final if q['neck'][1]==10 else initial)
 kin=SimpleNamespace(urdf_path=p,q_from_feedback=lambda q,**kw:q,collision_pairs=pairs)
 guard=SimpleNamespace(kin=kin,blocking_pairs=lambda pairs,values:list(pairs),cache=set())
 return guard,source,goal

def test_initial_model_overlap_can_exit_but_not_reenter(tmp_path,monkeypatch):
 g,s,t=make_guard(tmp_path,monkeypatch,[HEAD,NECK],[NECK]);original=g.blocking_pairs
 with mod.reset_path_scope(g,s,t) as notes:
  assert g.blocking_pairs([HEAD,NECK],s)==[]
  mid={**s,'Joint_Neck_Pitch':5.}
  assert g.blocking_pairs([HEAD,NECK,OTHER],mid)==[OTHER]
  assert g.blocking_pairs([NECK],mid)==[]
  assert g.blocking_pairs([HEAD],mid)==[HEAD]
 assert g.blocking_pairs==original and HEAD in notes

@pytest.mark.parametrize('initial,final', [([],[]),([HEAD],[HEAD]),([HEAD,OTHER],[])])
def test_new_collision_or_occupied_endpoint_not_exempt(tmp_path,monkeypatch,initial,final):
 g,s,t=make_guard(tmp_path,monkeypatch,initial,final)
 with mod.reset_path_scope(g,s,t):assert g.blocking_pairs([HEAD],s)==[HEAD]

def test_unknown_model_and_off_path_not_exempt(tmp_path,monkeypatch):
 g,s,t=make_guard(tmp_path,monkeypatch,[HEAD],[])
 with mod.reset_path_scope(g,s,t):assert g.blocking_pairs([HEAD],{**s,'Joint_Neck_Pitch':11.})==[HEAD]
 monkeypatch.setattr(mod,'MODEL_SHA256','different')
 with mod.reset_path_scope(g,s,t):assert g.blocking_pairs([HEAD],s)==[HEAD]
