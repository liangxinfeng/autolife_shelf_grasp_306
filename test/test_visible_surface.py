import numpy as np
import pytest
from autolife_shelf_grasp_306.unregistered import visible_surface
from autolife_shelf_grasp_306.bottle import DepthQualityError


def rounded_surface():
    x,y=np.meshgrid(np.linspace(-.04,.04,70),np.linspace(-.035,.035,70))
    r=(x/.04)**2+(y/.035)**2;keep=r<1
    return np.c_[x[keep],y[keep],.045*np.sqrt(1-r[keep])]


def certify(points,camera=(0,0,.6),basis=None):
    lo,hi=np.percentile(points,[1,99],axis=0)
    return visible_surface(points,lo,hi,np.eye(3) if basis is None else basis,np.array(camera))


def test_curved_surface_needs_no_flat_top():
    result=certify(rounded_surface())
    assert result['surface_evidence']=='irregular_visible_surface'
    assert result['surface_volume_ratio']>.2


def test_flat_box_top_unchanged():
    p=rounded_surface();p[p[:,2]>.02,2]=.045
    assert certify(p)['surface_evidence']=='visible_top'


def test_low_view_does_not_infer_hidden_back():
    with pytest.raises(DepthQualityError):certify(rounded_surface(),camera=(.6,0,.1))


@pytest.mark.parametrize('noise',[0,.0008,.0015])
def test_tilted_front_plane_is_not_a_measured_volume(noise):
    x,y=np.meshgrid(np.linspace(-.04,.04,70),np.linspace(-.035,.035,70))
    z=x*.6+np.random.default_rng(42).normal(0,noise,x.shape)
    with pytest.raises(DepthQualityError):certify(np.c_[x.ravel(),y.ravel(),z.ravel()])


def test_camera_and_extent_use_same_reference_frame():
    p=rounded_surface();angle=.7
    b=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    a=certify(p);c=certify(p,camera=b@np.array([0,0,.6]),basis=b)
    assert c['surface_evidence']==a['surface_evidence']
    assert c['view_elevation_deg']==pytest.approx(a['view_elevation_deg'])


def test_small_depth_noise_cannot_become_thickness(monkeypatch):
    import autolife_shelf_grasp_306.unregistered as module
    p=rounded_surface();p[:,2]*=.04
    pixels=np.c_[np.zeros(len(p)),np.linspace(1,99,len(p))]
    monkeypatch.setattr(module,'object_points',lambda *a:(p,pixels,(0,100)))
    mask=np.zeros((120,120),bool);mask[10:110,10:110]=True
    T=np.eye(4);T[:3,3]=[0,0,.6]
    # The unchanged caller's size guard rejects a near-flat sheet before
    # top-surface certification; that helper alone doesn't enforce size.
    with pytest.raises(DepthQualityError,match='过薄'):
        module.estimate_temporary({},mask,T,[0,0,1],{'geometry':'solid'}, {})


def test_solid_directly_below_camera_needs_no_horizontal_view_vector(monkeypatch):
    import autolife_shelf_grasp_306.unregistered as module
    p=rounded_surface();pixels=np.c_[np.zeros(len(p)),np.linspace(1,99,len(p))]
    monkeypatch.setattr(module,'object_points',lambda *a:(p,pixels,(0,100)))
    mask=np.zeros((120,120),bool);mask[10:110,10:110]=True
    T=np.eye(4);T[:3,3]=[0,0,.6]
    c,s=module.estimate_temporary({},mask,T,[0,0,1],{'geometry':'solid'}, {})
    assert np.isfinite(c).all() and s['surface_evidence']=='irregular_visible_surface'
