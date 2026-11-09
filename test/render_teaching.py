import os,time,tempfile,shutil,json
from pathlib import Path
from types import SimpleNamespace
from PyQt5 import QtCore as C,QtWidgets as Q
from autolife_shelf_grasp_306.app import STYLE
from autolife_shelf_grasp_306.teaching_studio import TeachingStudio
from test_teaching import pose,payload,draft

app=Q.QApplication([]);app.setStyleSheet(STYLE)
root=Path(tempfile.mkdtemp(prefix='teaching-ui-'));(root/'config').mkdir();(root/'run').mkdir()
shutil.copy('/home/ubuntu/ros2_ws/src/autolife_shelf_grasp_306/config/grasp.yaml',root/'config/grasp.yaml')
owner=SimpleNamespace(root=root,run=root/'run');w=TeachingStudio(owner);w.runtime.timer.stop();w.ensure();w.resize(1060,930);w.show()
q=pose(Joint_Left_Elbow=-60,Joint_Right_Elbow=-60,Joint_Left_Shoulder_Outer=20,Joint_Right_Shoulder_Outer=-20)
p1=dict(q);p1['Joint_Right_Shoulder_Inner']=20
p2=dict(p1);p2['Joint_Right_Gripper']=180
p3=dict(p2);p3['Joint_Right_Shoulder_Inner']=-20
p4=dict(p3);p4['Joint_Right_Gripper']=10
ed=w.editor;ed.draft=draft([p1,p2,p3,p4]);ed.draft['name']='取饮料并放到前台'
for f,label in zip(ed.draft['frames'],['靠近瓶身','夹住饮料','移动到前台','松开放下']):f['label']=label
ed.name.setText(ed.draft['name']);ed.refresh_frames(1)
def feed():w.feedback.accept(payload(q))
feed_timer=C.QTimer();feed_timer.timeout.connect(feed);feed_timer.start(40)
def check():
    ed.save();assert w.library.load(ed.draft['name'])['frames'][1]['label']=='夹住饮料'
    old=q['Joint_Right_Wrist_Upper'];ed.model_joint('Joint_Right_Wrist_Upper',3.)
    assert w.feedback.values['Joint_Right_Wrist_Upper']==old
    ed.undo();assert ed.selected()['joints_deg']['Joint_Right_Wrist_Upper']==old
    assert not w.engine.bridge and not w.engine.node
    assert not any('target_joints' in topic for topic,_ in w.monitor.get_topic_names_and_types())
    w.grab().save('/tmp/vr_teaching_20261030/editor.png')
    ed.prepare_preview();ed.elapsed=1.;ed.tick()
    w.grab().save('/tmp/vr_teaching_20261030/preview.png')
    print('UI_OK size',w.size().width(),w.size().height(),'capture publisher count',len(list(w.monitor.publishers)),'engine bridge',w.engine.bridge,flush=True)
    w.shutdown();app.quit()
C.QTimer.singleShot(2000,check)
raise SystemExit(app.exec_())
