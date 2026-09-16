"""Replay saved targets against the current URDF; no ROS or motion calls."""
import argparse
import json
import numpy as np
from autolife_shelf_grasp_306.planning import Planner

parser=argparse.ArgumentParser()
parser.add_argument('plans',nargs='+')
args=parser.parse_args()
for filename in args.plans:
    with open(filename,encoding='utf-8') as stream:old=json.load(stream)
    planner=Planner(old['config'])
    plan=planner.plan(old['source_q'],np.asarray(old['tcp']),np.asarray(old['center']),old['product'])
    assert len(plan['segments'])==4 and plan['self_collision_checked']
    assert np.allclose(plan['segments'][1]['tcp_goal'],old['center'])
    for segment in plan['segments']:
        knots=segment['waypoints']
        for a,b in zip(knots,knots[1:]):
            assert max(abs(a[n]-b[n]) for n in planner.names)<=8.00001
        position,_=planner.pose(knots[-1],np.asarray(old['tcp']))
        assert np.linalg.norm(position-segment['tcp_goal'])<=.004
    print('APPROACH_REPLAY_OK',json.dumps({'product':old['product']['name'],
        'old_yaw_deg':old.get('approach_yaw_deg'),'new_yaw_deg':plan['approach_yaw_deg'],
        'profile':plan['pregrasp_profile'],'orientation':plan['orientation_mode'],
        'seconds':plan['seconds'],'checked_states':plan['checked_states'],
        'failed_candidates':len(plan['candidate_failures'])},ensure_ascii=False),flush=True)
