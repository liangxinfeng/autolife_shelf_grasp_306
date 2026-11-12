"""按真实碰撞模型关节角采样，不把DM夹爪编码量当作角度。无硬件接口。"""
import math
from control_center_joint_control.joint_definitions import BODY_JOINTS, GRIPPER_JOINTS


def sample_count(segment, model_names):
    # 只有确认模型中没有活动夹爪轴，才排除其编码单位。模型升级自动兼容。
    names = list(BODY_JOINTS)
    if any(n in model_names for n in GRIPPER_JOINTS):
        names += list(GRIPPER_JOINTS)
    # 所有身体关节仍使用原来的0.25度采样上限；腰腿和另一臂也参与。
    return max(2, math.ceil(max(abs(segment['target'][n]-segment['start'][n])
                              for n in names)/.25))
