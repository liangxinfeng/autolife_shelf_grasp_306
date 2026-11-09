"""全身复位专用入口：沿用正式姿态控制接口，不继承标定工具的禁用规则。

只由已检查的全身复位执行进程调用；ROS上的标定动作入口仍不开放。
"""
import json
from control_center_joint_control.joint_bridge import JointBridge
from std_srvs.srv import SetBool
from std_msgs.msg import String
from .pose_execution_bridge import make_bridge

def make_reset_bridge(base,context):
    class WorkspaceBridge(base):
        def enable_workspace_body(self):
            return JointBridge._set_leg_waist_enabled(self,SetBool.Request(data=True),SetBool.Response())
        def start_workspace_reset(self,token,record):
            if not getattr(self,'teaching_sync_verified',False):
                raise ValueError('全身复位位置SYNC尚未就绪')
            if not str(token).startswith('workspace-reset-'):raise ValueError('全身复位调用来源无效')
            JointBridge._on_action_command(self,String(data=json.dumps(dict(command='start',name=token,action=record))))
    return make_bridge(WorkspaceBridge,context)
