import copy
from types import SimpleNamespace
import numpy as np
from autolife_shelf_grasp_306.visual_object_memory import memory_prompt,ObjectAppearanceMemory
from autolife_shelf_grasp_306.grasp_geometry import same_nominal_camera

def test_memory_scope_ignores_grasp_filter_and_keeps_identity():
 cfg={'semantic_prompt_overrides':{'all':'纸巾'},'grounded_sam2_prompts':{'all':'bottle. cup. box.'}}
 before=copy.deepcopy(cfg);image=np.zeros((20,20,3),np.uint8);mask=np.ones((20,20),bool);calls=[]
 def proposals(frame,mode,**kw):
  calls.append(kw['prompt']);return [dict(mask=mask)]
 p=SimpleNamespace(segmenter=SimpleNamespace(proposals=proposals),embed=lambda crops:np.array([[1.,0]]*len(crops)))
 views=[({'name':'垃圾盖'},{'image':image,'mask':mask})];store=SimpleNamespace(view_ids=lambda ident:['root'])
 result=ObjectAppearanceMemory(p,cfg).recognize(store,'id',views,{'image':image})
 assert result[1]['recognition_method']=='dinov2_object'
 assert 'trash bin lid' in calls[0] and 'tissue' not in calls[0]
 assert cfg==before

def test_custom_name_preserves_broad_candidates():
 cfg={'semantic_prompt_overrides':{'all':'纸巾'},'grounded_sam2_prompts':{'all':'bottle. box.'}}
 assert memory_prompt(cfg,[({'name':'农夫山泉1bb307'}, {})])=='bottle. box.'
 assert 'bottle' in memory_prompt(cfg,[({'name':'custom 123'}, {})])

def test_camera_reuse_only_identical_nominal_optical_chain():
 a=dict(geometry_source='urdf',geometry_version=2,urdf_sha256='hash',T_mount_camera=np.eye(4).tolist(),config=dict(robot_id=345,head_offsets_deg=[0,0,0],arm='left'))
 b=copy.deepcopy(a);b['config']['arm']='right';b['T_wrist_tool']='different TCP'
 assert same_nominal_camera(a,b)
 for key,val in [('geometry_source','calibrated'),('urdf_sha256','other'),('geometry_version',1),('T_mount_camera',np.zeros((4,4)).tolist())]:
  changed=copy.deepcopy(b);changed[key]=val;assert not same_nominal_camera(a,changed)
 for key,val in [('robot_id',300),('head_offsets_deg',[1,0,0])]:
  changed=copy.deepcopy(b);changed['config'][key]=val;assert not same_nominal_camera(a,changed)
 assert not same_nominal_camera({}, {})

def test_gripper_geometry_frozen_only_inside_one_planning_call(monkeypatch):
 from autolife_shelf_grasp_306.low_object_grasp import grasp_scope,gripper_vertices
 import autolife_shelf_grasp_306.gripper_contact_geometry as module
 calls=[]
 def geometry(p):calls.append(p.cfg['width']);return {'vertices':np.full((3,3),p.cfg['width'])}
 monkeypatch.setattr(module,'gripper_geometry',geometry)
 class P:
  cfg={'arm':'right','width':90};guard=object()
  @grasp_scope
  def plan(self):return [gripper_vertices(self).copy() for _ in range(10)]
 p=P();assert p.plan()[0][0,0]==90
 p.cfg={**p.cfg,'width':80};assert p.plan()[0][0,0]==80
 assert calls==[90,80] and p._request_gripper_vertices is None
