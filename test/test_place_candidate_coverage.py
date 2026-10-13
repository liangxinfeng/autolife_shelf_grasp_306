import itertools
import numpy as np
import pytest
from autolife_shelf_grasp_306.demo_planning import placement_trials,jaw_envelope_bounds


def test_deadline_prefix_covers_headings_positions_and_branches():
    targets=[np.array([i,0.,0.]) for i in range(10)]
    prefix=list(itertools.islice(placement_trials(targets,2),75))
    assert {s for _,_,s in prefix}=={0,1}
    assert len({t[0] for t,_,_ in prefix})>=4
    assert {-25,25,-15,15,0}.issubset({yaw for _,yaw,_ in prefix})


def test_full_queue_is_unique_and_complete():
    entries=[(tuple(t),a,s) for t,a,s in placement_trials([np.zeros(3),np.ones(3)],2)]
    assert len(entries)==2*17*2 and len(set(entries))==len(entries)
    assert list(placement_trials([],2))==[]


def test_jaw_envelope_preserves_model_thickness_and_full_opening():
    low=np.array([-.01871,-.03281,-.09421]);high=np.array([.24219,.03268,.03367])
    tcp=np.array([.22947,.00731,-.01594])
    lo,hi=jaw_envelope_bounds(low,high,tcp,.095)
    # Thickness derives from the model, not the finger separation or a sphere.
    assert lo[1]<low[1] and hi[1]>high[1]
    assert hi[1]-lo[1]==pytest.approx(high[1]-low[1]+.002)
    assert hi[2]-lo[2]>=.095+.08
    assert lo[2]<=low[2] and hi[2]>=high[2]
    assert lo[0]<tcp[0]-.045 and hi[0]>tcp[0]+.045
    np.testing.assert_allclose(low,[-.01871,-.03281,-.09421])


def test_envelope_contains_front_mesh_and_does_not_track_virtual_center():
    low=np.array([-.02,-.033,-.095]);high=np.array([.243,.033,.034])
    physical=np.array([.23,.007,-.016]);offset=np.array([.035,0,.01])
    a=jaw_envelope_bounds(low,high,physical,.095)
    b=jaw_envelope_bounds(low,high,(physical+offset)-offset,.095)
    np.testing.assert_allclose(a,b)
    for corner in itertools.product([physical[0]-.045,high[0]],
                                    [low[1],high[1]],[low[2],high[2]]):
        assert np.all(corner>=a[0]) and np.all(corner<=a[1])


@pytest.mark.parametrize('width',[0,-.1,float('nan')])
def test_invalid_envelope_is_rejected(width):
    with pytest.raises(ValueError):jaw_envelope_bounds([0,0,0],[1,1,1],[.5,.5,.5],width)
