import numpy as np
import pytest
from autolife_shelf_grasp_306.bottle import center_from_points,combine,horizontal_basis,DepthQualityError
from autolife_shelf_grasp_306.core import stages,reset_blockers,execution_blockers


def cylinder():
    t,z=np.meshgrid(np.linspace(-1.1,1.1,40),np.linspace(-.109,.109,80))
    p=np.c_[.6-.035*np.cos(t.ravel()),.1+.035*np.sin(t.ravel()),.2+z.ravel()]
    return p,{'diameter_m':.07,'height_m':.22}


@pytest.mark.parametrize('sign',[1,-1])
def test_center_does_not_assume_z_up(sign):
    p,product=cylinder();p[:,2]*=sign
    center,stats,_=center_from_points(p,[0,0,sign],product,[0,.1,.2*sign])
    np.testing.assert_allclose(center,[.6,.1,.2*sign],atol=.002)
    assert stats['height_source']=='visible_bottle_midpoint'


def test_incomplete_height_rejected():
    p,product=cylinder();p=p[p[:,2]>.2]
    with pytest.raises(DepthQualityError,match='高度'):center_from_points(p,[0,0,1],product,[0,.1,.2])


def test_invalid_dimensions_are_not_retryable_depth_error():
    p,product=cylinder();product['diameter_m']=.9
    with pytest.raises(ValueError,match='尺寸') as error:center_from_points(p,[0,0,1],product,[0,.1,.2])
    assert not isinstance(error.value,DepthQualityError)


def test_lift_uses_real_up():
    cfg={'up_torso':[0,0,-1],'pregrasp_m':.12,'lift_m':.035,'withdraw_m':.2}
    out=stages([.6,.1,-.2],0,cfg)
    np.testing.assert_allclose(out[2][1],[.6,.1,-.235])
    f,s,u=horizontal_basis([.1,0,-1]);assert abs(f@u)<1e-10
    assert np.linalg.det(np.column_stack([f,s,u]))==pytest.approx(1)


def test_no_silent_default_up():
    with pytest.raises(KeyError):stages([0,0,0],0,{'lift_m':.03})


def test_reset_does_not_unlock_grasp():
    cfg={'reset_hardware_enabled':True}
    assert reset_blockers(cfg)==[] and len(execution_blockers(cfg))==5
    assert reset_blockers({'reset_hardware_enabled':'true'})


def test_multiframe():
    samples=[(np.array([.6,.1,-.2]),{'observed_height_mm':219}) for _ in range(5)]
    c,stats=combine(samples);assert stats['support_required'] is False
    samples[-1]=(c+[.03,0,0],samples[-1][1])
    with pytest.raises(ValueError,match='10mm'):combine(samples)
