import os,time,tempfile,shutil
os.environ['ROS_DOMAIN_ID']='87'
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PyQt5 import QtCore as C,QtWidgets as Q
from autolife_shelf_grasp_306.app import STYLE
from autolife_shelf_grasp_306.teaching_studio import TeachingStudio
from autolife_shelf_grasp_306.teaching_track import make_draft,NAMES
from test_teaching import pose,payload
app=Q.QApplication([]);app.setStyleSheet(STYLE)
root=Path(tempfile.mkdtemp(prefix='teaching-modes-ui-'));(root/'config').mkdir();(root/'run').mkdir()
shutil.copy('/home/ubuntu/ros2_ws/src/autolife_shelf_grasp_306/config/grasp.yaml',root/'config/grasp.yaml')
w=TeachingStudio(SimpleNamespace(root=root,run=root/'run'));w.runtime.timer.stop();w.ensure();w.resize(1060,1010);w.show()
q=pose(Joint_Right_Elbow=-70,Joint_Left_Elbow=-70,Joint_Left_Shoulder_Outer=20,Joint_Right_Shoulder_Outer=-20)
t=np.arange(0.,6.001,.02);values=np.array([[q[n] for n in NAMES]]*len(t))
values[:,NAMES.index('Joint_Right_Shoulder_Inner')]+=10*np.sin(t/6*np.pi)
values[:,NAMES.index('Joint_Right_Gripper')]+=60*np.clip((t-2.5)/.8,0,1)
d=make_draft(w.library.directory,'示例：取饮料并收回',t,values,[50,150,250]);w.library.save(d)
ed=w.editor;ed.refresh_library();ed.load_selected(ed.actions.findData(d['name']))
def feed():w.feedback.accept(payload(q))
timer=C.QTimer();timer.timeout.connect(feed);timer.start(40)
def check():
    result=0
    try:
        ed.model_joint('Joint_Right_Wrist_Upper',2.);assert w.feedback.values['Joint_Right_Wrist_Upper']==q['Joint_Right_Wrist_Upper']
        ed.undo();ed.prepare_preview();ed.elapsed=1.;ed.tick()
        app.processEvents();w.grab().save('/tmp/vr_teaching_modes_test_20261030/editor_track.png')
        ed.end_preview();w.recorder.timer.stop()
        data=dict(available=True,enabled=True,mode='trajectory',phase='recording',service_id='render',session='render',revision=1,
                  name=d['name'],count=len(d['frames']),samples=175,elapsed=3.5,message='正在录制',resetting=True)
        w.recorder.data=data;ed.update_recorder(data);ed.tick()
        app.processEvents();w.grab().save('/tmp/vr_teaching_modes_test_20261030/editor_recording.png')
        assert not ed.angle.isEnabled() and ed.record_action.isEnabled()
        assert not w.engine.bridge and not w.engine.node
        print('RENDER_OK: real URDF, recorded curve preview, live recording controls, no motor bridge',flush=True)
    except Exception:
        import traceback
        traceback.print_exc();result=1
    finally:w.shutdown();app.exit(result)
C.QTimer.singleShot(2200,check)
raise SystemExit(app.exec_())
