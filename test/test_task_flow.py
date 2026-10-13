import unittest
from autolife_shelf_grasp_306.task_flow import Cursor, validate, select_instance, navigation_pose
from autolife_shelf_grasp_306.task_flow import full_record
from control_center_joint_control.joint_definitions import GROUP_ORDER, ALL_JOINTS, JOINT_LIMITS


def action():
    values={n:(JOINT_LIMITS[n][0]+JOINT_LIMITS[n][1])/2 for n in ALL_JOINTS}
    return dict(groups=GROUP_ORDER,loop_count=1,speed_percent=100,keyframes=[dict(
        joints_deg=values,transition_sec=3.,hold_sec=0.,easing='smooth')])


class FlowTests(unittest.TestCase):
    def test_failure_does_not_advance(self):
        c=Cursor({'steps':[dict(kind='navigate',target='前台'),dict(kind='place')]})
        token=c.token;c.stop('失败');self.assertFalse(c.complete(token));self.assertEqual(c.index,0)
    def test_duplicate_receipt_ignored(self):
        c=Cursor({'steps':[dict(kind='navigate',target='前台'),dict(kind='place')]})
        token=c.token;self.assertTrue(c.complete(token));self.assertFalse(c.complete(token));self.assertEqual(c.index,1)
        self.assertTrue(c.complete(c.token));self.assertFalse(c.active)
    def test_snapshot_is_immutable(self):
        r={'steps':[dict(kind='action',target='开门')]};c=Cursor(r);r['steps'][0]['target']='关门'
        self.assertEqual(c.step['target'],'开门')
    def test_invalid_steps(self):
        for r in ({'steps':[]},{'steps':[dict(kind='navigate')]},{'steps':[dict(kind='shell',target='anything')]}):
            with self.assertRaises(ValueError):validate(r)
    def test_instance_ambiguity(self):
        self.assertEqual(select_instance([dict(id=1,name='瓶子'),dict(id=2,name='瓶子')],'瓶子',[]),1)
        self.assertEqual(select_instance([dict(id=3,product_id='x')],'怡宝',[dict(id='x',name='怡宝')]),3)
    def test_fullbody(self):
        result=full_record(action());self.assertEqual(len(result['keyframes'][0]['joints_deg']),23)
    def test_partial_and_nonfinite_fail(self):
        a=action();a['groups']=['left_arm']
        with self.assertRaises(ValueError):full_record(a)
        for value in (float('nan'),float('inf'),10000):
            a=action();a['keyframes'][0]['joints_deg'][ALL_JOINTS[0]]=value
            with self.assertRaises(ValueError):full_record(a)
    def test_limits_not_silently_clamped(self):
        a=action();del a['keyframes'][0]['joints_deg'][ALL_JOINTS[0]]
        with self.assertRaises(ValueError):full_record(a)
    def test_generic_navigation_stationary_snapshot(self):
        c=Cursor({'steps':[dict(kind='navigate',target='前台')]})
        context=dict(run_id=c.run_id,map='地图',record=c.record)
        cmd=dict(leg=c.token,run_id=c.run_id,map='地图',name='前台')
        q=action()['keyframes'][0]['joints_deg'];samples=[(v,dict(q)) for v in (1.,1.2,1.5)]
        self.assertEqual(navigation_pose(cmd,context,samples,1.6),q)
        with self.assertRaises(ValueError):navigation_pose(cmd,context,samples,3.)
        samples[0][1][ALL_JOINTS[0]]+=5
        with self.assertRaises(ValueError):navigation_pose(cmd,context,samples,1.6)
    def test_generic_navigation_rejects_wrong_step(self):
        c=Cursor({'steps':[dict(kind='grasp',target='怡宝')]})
        with self.assertRaises(ValueError):navigation_pose(dict(leg=c.token,run_id=c.run_id,map='m',name='前台'),
            dict(run_id=c.run_id,map='m',record=c.record),[],1.)


if __name__=='__main__':unittest.main()
