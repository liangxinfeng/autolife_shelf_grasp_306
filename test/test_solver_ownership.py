from types import SimpleNamespace as S
from autolife_shelf_grasp_306.planner_cache import transfer_idle_solver


def test_process_moves_without_shared_ownership():
    solver=S(_closed=False,process=S(poll=lambda:None));donor=S(solver=solver);receiver=S(solver=None)
    assert transfer_idle_solver(receiver,('same',donor),'same')
    assert donor.solver is None and receiver.solver is solver
    assert transfer_idle_solver(donor,('same',receiver),'same')
    assert receiver.solver is None and donor.solver is solver


def test_changed_model_cannot_reuse_solver():
    donor=S(solver=S(_closed=False,process=S(poll=lambda:None)));receiver=S(solver=None)
    assert not transfer_idle_solver(receiver,('old',donor),'new')
    assert receiver.solver is None and donor.solver is not None


def test_closed_process_not_reused():
    for solver in (S(_closed=True,process=S(poll=lambda:None)),S(_closed=False,process=S(poll=lambda:1))):
        assert not transfer_idle_solver(S(solver=None),('same',S(solver=solver)),'same')


def test_existing_owner_kept():
    receiver=S(solver=object());old=receiver.solver;donor=S(solver=object())
    assert not transfer_idle_solver(receiver,('same',donor),'same')
    assert receiver.solver is old
