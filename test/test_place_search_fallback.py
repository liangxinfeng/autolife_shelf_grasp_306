import time
from types import SimpleNamespace
import numpy as np
from autolife_shelf_grasp_306.demo_planning import placement_candidates,rank_place_ik


def test_finer_shortlist_retains_local_pockets():
    near=np.array([[x,y,0.] for x in np.arange(0,.10,.005) for y in np.arange(0,.10,.005)])
    far=near+[.25,.25,0.];points=np.vstack([near,far]);current=np.zeros(3)
    old=placement_candidates(points,current)
    fine=placement_candidates(points,current,limit=50,spacing=.015)
    assert len(old)==10 and len(fine)>10
    assert sum(np.linalg.norm(p)<.15 for p in fine)>sum(np.linalg.norm(p)<.15 for p in old)


class FakePlanner:
    names=['arm']
    def __init__(self,good):self.good=good;self.seen=[]
    def solve(self,source,tcp,target,*args,**kw):
        self.seen.append(float(target[0]))
        if float(target[0]) not in self.good:raise ValueError('unreachable')
        return {'arm':float(target[0])}
    def check(self,*args):pass


def rank(p,first,fallback):
    return rank_place_ik(p,{'arm':0.},np.zeros(3),first,np.eye(3),np.array([0,0,1.]),{},
        {'approach_height_m':.1},SimpleNamespace(check=lambda *a:None),time.monotonic()+8,[],fallback)


def test_fallback_only_after_primary_failed():
    p=FakePlanner({2.});r=rank(p,[np.array([1.,0,0])],[np.array([2.,0,0])])
    assert r and r[0]['target'][0]==2 and p.seen[:5]==[1.]*5
    p=FakePlanner({1.,2.});r=rank(p,[np.array([1.,0,0])],[np.array([2.,0,0])])
    assert r and 2. not in p.seen


def test_planning_failure_does_not_invoke_observation():
    import ast
    from pathlib import Path
    text=(Path(__file__).parents[1]/'autolife_shelf_grasp_306/worker.py').read_text();ast.parse(text)
    branch=next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.If)
                and isinstance(n.test,ast.Name) and n.test.id=='surface_ready')
    # A fresh-image retry continues the read-only command loop; otherwise
    # fail here. Neither case may fall through to a physical observation move.
    assert isinstance(branch.body[-1],ast.Raise)
    assert any(isinstance(n,ast.Continue) for n in ast.walk(branch))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                   and n.func.id=='plan_observation' for n in ast.walk(branch))
    assert "diag['observation_reason']" in text
