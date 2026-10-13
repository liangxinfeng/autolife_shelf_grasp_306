import pytest
from autolife_shelf_grasp_306.chassis import InputWatchdog


def packet(seq,stamp,values=(.1,0.,0.)):
    return {'sequence':seq,'stamp':stamp,'velocity':values}


def test_no_input_never_drives_and_stale_input_stops():
    w=InputWatchdog();assert w.current(1)==(0,0,0)
    w.update(packet(1,1),1.01);assert w.current(1.1)==(.1,0,0)
    assert w.current(1.3)==(0,0,0)
    w.update(packet(2,1.4,(0,0,0)),1.4);assert w.current(1.4)==(0,0,0)


@pytest.mark.parametrize('values',[(1,0,0),(.15,.15,0),(0,0,1),(float('nan'),0,0),(0,0)])
def test_bad_velocity_rejected(values):
    with pytest.raises(ValueError):InputWatchdog().update(packet(1,1,values),1)


def test_old_future_and_duplicate_frames_rejected():
    w=InputWatchdog();w.update(packet(2,1),1)
    for p in [packet(1,1),packet(2,1),packet(3,.5),packet(3,2)]:
        with pytest.raises(ValueError):w.update(p,1)


def test_fault_needs_fresh_key_release_not_held_key_replay():
    w=InputWatchdog();w.update(packet(1,1),1);w.inhibit()
    w.update(packet(2,1.1),1.1);assert w.current(1.1)==(0,0,0)
    w.update(packet(3,1.2,(0,0,0)),1.2)
    w.update(packet(4,1.3),1.3);assert w.current(1.3)==(.1,0,0)
