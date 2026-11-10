"""放置执行契约的无硬件测试：同一控制器、到位松爪、取消保持、固定关节。"""
import copy
import unittest
import numpy as np
from types import SimpleNamespace
from autolife_handeye_calibration_306.core import BODY,ARMS,NECK
from control_center_joint_control.joint_definitions import GROUPS
from autolife_shelf_grasp_306.vendor_place_trajectory import build_blocks,validate_blocks,execute_blocks


def fixture(side='right'):
    source={n:0. for n in BODY+ARMS['left']+ARMS['right']+NECK+['Joint_Left_Gripper','Joint_Right_Gripper']}
    names=BODY[2:]+ARMS[side];points=[]
    for value in (0.,1.,2.,0.):
        q=dict(source);q[names[0]]=value;q[names[2]]=value;points.append(q)
    plan=dict(kind='place',place_protocol_version=4,execution_backend='vendor_place_trajectory_v1',
        ik_backend='autolife_robot_sdk.KdlSolver.nr_jl',controlled_joints=names,source_q=source,
        waist_bounds_deg={n:[-12.,12.] for n in BODY[2:]},config={'arm':side},
        native_checked_states=1,table_height_checked=True,segments=[])
    for i,phase in enumerate(('place_approach','place_lower','place_return')):
        length=abs(points[i+1][names[0]]-points[i][names[0]])
        plan['segments'].append(dict(name=phase,phase=phase,waypoints=points[i:i+2],path_speed_profile=[[0.,4.],[length,4.]]))
    plan['vendor_blocks']=build_blocks(plan,plan['config'])
    return plan


class PlaceTests(unittest.TestCase):
    def test_both_arms(self):
        for side in ('left','right'):validate_blocks(fixture(side))

    def test_slower_slider(self):
        p=fixture();slow={**p['config'],'grasp_speed_percent':50}
        self.assertGreater(sum(b['duration_sec'] for b in build_blocks(p,slow)),sum(b['duration_sec'] for b in p['vendor_blocks']))

    def test_fixed_other_arm(self):
        p=fixture();p['segments'][1]['waypoints'][-1][ARMS['left'][0]]=1
        p['vendor_blocks']=build_blocks(p,p['config'])
        with self.assertRaisesRegex(ValueError,'固定关节'):validate_blocks(p)

    def test_fixed_knee(self):
        p=fixture();p['segments'][0]['waypoints'][0][BODY[0]]=1
        p['vendor_blocks']=build_blocks(p,p['config'])
        with self.assertRaises(ValueError):validate_blocks(p)

    def test_changed_timing(self):
        p=fixture();p['vendor_blocks'][0]['duration_sec']*=.5
        with self.assertRaisesRegex(ValueError,'时序'):validate_blocks(p)

    def test_missing_check(self):
        p=fixture();p['native_checked_states']=0
        with self.assertRaises(ValueError):validate_blocks(p)

    def test_wrong_order(self):
        p=fixture();p['controlled_joints']=p['controlled_joints'][::-1]
        with self.assertRaises(ValueError):validate_blocks(p)

    def test_bad_profile(self):
        p=fixture();p['segments'][0]['path_speed_profile'][0][1]=float('nan')
        with self.assertRaises(ValueError):build_blocks(p,p['config'])

    def run_execution(self,stop=False,arrival=True):
        p=fixture();q=dict(p['source_q']);now=[0.];events=[];writes=[0]
        def checked():
            now[0]+=.05
            if stop and writes[0]>=3:raise ValueError('用户停止')
            return dict(q)
        def publish(pub,payload):
            for group in ('leg_waist','right_arm'):
                for (name,*_),value in zip(GROUPS[group]['joints'],payload[GROUPS[group]['command_key']]):q[name]=value
            writes[0]+=1
        node=SimpleNamespace(_publish_vendor=publish,_whole_body_pub=None)
        def trajectory(adapter,rows,**kwargs):
            for row in rows:adapter._set_target_body_joints_deg(row[:11],row[:4]+row[11:])
            events.append('trajectory')
        def release():
            self.assertEqual(events,['trajectory','trajectory'])
            self.assertAlmostEqual(q[BODY[2]],2.)
            events.append('release')
        try:
            execute_blocks(node,p,checked,lambda:now[0],release,lambda *a:events.append('hold'),
                lambda value:None,lambda *a:arrival,trajectory)
        except ValueError:
            self.assertNotIn('release',events)
            raise
        return events

    def test_release_only_at_landing(self):
        self.assertEqual(self.run_execution(),['trajectory','trajectory','release','trajectory','hold'])

    def test_stop_never_releases(self):
        with self.assertRaisesRegex(ValueError,'用户停止'):self.run_execution(stop=True)

    def test_fk_not_arrived_never_releases(self):
        with self.assertRaisesRegex(ValueError,'尚未到位'):self.run_execution(arrival=False)

if __name__=='__main__':unittest.main()
