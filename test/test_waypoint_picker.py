import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import tempfile, unittest, json
from pathlib import Path
from unittest.mock import patch
from PyQt5 import QtWidgets as Q
from autolife_shelf_grasp_306 import navigation_logic as nav
from autolife_shelf_grasp_306.app import Window
from autolife_shelf_grasp_306.task_flow_ui import FlowDialog
import yaml

app=Q.QApplication([])


class PickerTests(unittest.TestCase):
    def test_current_map_only_radians_unchanged(self):
        root=Path(tempfile.mkdtemp());active=root/'active.yaml';points=root/'points.yaml'
        active.write_text(yaml.safe_dump(dict(active_map=str(root/'a.yaml'),waypoints_file=str(points))))
        points.write_text(yaml.safe_dump(dict(map_yaml=str(root/'a.yaml'),waypoints={'前台':dict(x=1,y=2,yaw=1.57)})))
        (root/'maps_index.json').write_text(json.dumps({'maps':{'wrong':{'prepared_positions':[{'name':'其他地图'}]}}}))
        with patch.object(nav,'ACTIVE',active):
            self.assertEqual(nav.waypoint_poses(),{'前台':[1.,2.,1.57]})
            self.assertEqual(nav.waypoint_poses([]),{})
            self.assertEqual(nav.waypoint_names({}),['前台'])
            points.write_text(yaml.safe_dump(dict(map_yaml=str(root/'b.yaml'),waypoints={'错图':dict(x=1,y=2,yaw=0)})))
            self.assertEqual(nav.waypoint_poses(),{})

    def test_offline_live_and_map_fault(self):
        self.assertEqual(nav.waypoint_names(dict(names=[],library={'前台':[],'货架':[]})),['前台','货架'])
        self.assertEqual(nav.waypoint_names(dict(names=['货架','货架'],library={'旧点':[]})),['货架'])
        self.assertEqual(nav.waypoint_names(dict(map_fault=True,library={'前台':[]})),[])

    def test_refresh_preserves_edit_and_saved_missing_target(self):
        root=Path(tempfile.mkdtemp());(root/'data').mkdir();run=root/'run';run.mkdir()
        w=Window(root,run,start_worker=False);w.timer.stop()
        w.navigation.data=dict(names=[],library={'前台':[],'货架':[]},active_map='/maps/a.yaml',ready=False)
        d=FlowDialog(w);d.add('navigate','货架');box=d.table.cellWidget(0,1)
        self.assertEqual(box.count(),2);self.assertEqual(box.currentText(),'货架')
        w.navigation.data.update(names=['前台','货架','垃圾桶'],ready=True);d.refresh_waypoints()
        self.assertEqual(box.count(),3);self.assertEqual(box.currentText(),'货架')
        box.setCurrentText('人工输入');d.refresh_waypoints();self.assertEqual(box.currentText(),'人工输入')
        w.navigation.data.update(names=['新前台'],active_map='/maps/b.yaml');d.refresh_waypoints()
        self.assertEqual(box.currentText(),'人工输入');self.assertIn('失效',d.nav_note.text())
        self.assertEqual(box.itemText(0),'新前台');self.assertEqual(box.count(),1)
        d.done(0);self.assertFalse(d.waypoint_timer.isActive());w.close()


if __name__=='__main__':unittest.main()
