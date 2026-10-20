import numpy as np
import pytest
from autolife_shelf_grasp_306.unregistered import visible_surface,upper_surface_support
from autolife_shelf_grasp_306.bottle import DepthQualityError


def cloud(angle=0):
    x,y=np.meshgrid(np.linspace(-.03,.03,80),np.linspace(-.045,.045,80))
    r=(x/.03)**2+(y/.045)**2;valid=r<1
    p=np.c_[x[valid],y[valid],.04*np.sqrt(1-r[valid])]
    a=np.radians(angle);R=np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
    return p@R.T


@pytest.mark.parametrize('angle',[-25,-15,0,15,25])
def test_tilted_rounded_surface_has_local_upper_evidence(angle):
    p=cloud(angle);lo,hi=np.percentile(p,[1,99],axis=0)
    result=visible_surface(p,lo,hi,np.eye(3),np.array([0.,0.,.6]))
    assert result['surface_evidence'] in ('irregular_visible_surface','visible_top')
    assert min(result['surface_coverage'])>=.50


def test_noisy_vertical_wall_is_not_upper_surface():
    y,z=np.meshgrid(np.linspace(-.045,.045,80),np.linspace(0,.08,80))
    p=np.c_[np.random.default_rng(42).normal(0,.006,y.size),y.ravel(),z.ravel()]
    lo,hi=np.percentile(p,[1,99],axis=0)
    assert upper_surface_support(p,lo,hi) is None


def test_sparse_and_line_like_patches_not_extent_evidence():
    p=np.c_[np.zeros(100),np.zeros(100),np.linspace(0,.06,100)]
    assert upper_surface_support(p,p.min(axis=0),p.max(axis=0)) is None
    p=cloud()[::100]
    assert upper_surface_support(p,p.min(axis=0),p.max(axis=0)) is None


def test_low_view_and_single_sloped_face_still_rejected():
    p=cloud();lo,hi=np.percentile(p,[1,99],axis=0)
    with pytest.raises(DepthQualityError):visible_surface(p,lo,hi,np.eye(3),np.array([.6,0.,.1]))
    p[:,2]=p[:,1]*.7
    lo,hi=np.percentile(p,[1,99],axis=0)
    with pytest.raises(DepthQualityError):visible_surface(p,lo,hi,np.eye(3),np.array([0.,0.,.6]))
