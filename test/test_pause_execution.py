"""Exercise the complete executor with simulated feedback, no ROS or motors."""
import time
import pytest
from test_right_execution import test_execution_finishes_without_stop as execute_case
from autolife_shelf_grasp_306 import task_pause


@pytest.mark.parametrize('mode',['hold_reset','grasp','grasp_place'])
def test_paused_executor_resumes_same_path(tmp_path,monkeypatch,mode):
    original=task_pause.BridgePause;events=[]
    class SimulatedPause(original):
        def __init__(self,node,run,*args,**kwargs):
            super().__init__(node,run,*args,**kwargs)
            node._current=node.q;node._communication_lost_joints=lambda _:[]
            tick=node.tick;start=time.monotonic();paused=False;resumed=False
            def feedback():
                nonlocal paused,resumed
                age=time.monotonic()-start
                if age>.15 and not paused:
                    (run/'task_pause').touch();paused=True;events.append('pause')
                if age>2.15 and not resumed:
                    (run/'task_pause').unlink(missing_ok=True);resumed=True;events.append('resume')
                # Real bridge goals contain current commands as well as targets.
                for goal in node._active_goals.values():goal.setdefault('commands',dict(goal['targets']))
                if not self.tick():tick()
            node.tick=feedback
    monkeypatch.setattr(task_pause,'BridgePause',SimulatedPause)
    execute_case(tmp_path,monkeypatch,mode)
    assert events==['pause','resume']
