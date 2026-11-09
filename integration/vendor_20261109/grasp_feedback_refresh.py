"""原厂电机进程内的只读刷新服务；不创建硬件、不使能、不写运动目标。"""
import json
import math
import time
from functools import lru_cache

GROUPS = {
    'neck': ('mod_motor_neck', 'neck_joint_state', 'JOINT_NAMES_GROUP_NECK'),
    'leg_waist': ('mod_motor_waist-leg', 'leg_waist_joint_state', 'JOINT_NAMES_GROUP_WAIST_LEG'),
    'left_arm': ('mod_motor_left_arm', 'left_arm_joint_state', 'JOINT_NAMES_GROUP_LEFT_ARM'),
    'right_arm': ('mod_motor_right_arm', 'right_arm_joint_state', 'JOINT_NAMES_GROUP_RIGHT_ARM'),
    'left_gripper': ('mod_eef_gripper_left', 'left_gripper_state', 'JOINT_NAMES_GROUP_LEFT_GRIPPER'),
    'right_gripper': ('mod_eef_gripper_right', 'right_gripper_state', 'JOINT_NAMES_GROUP_RIGHT_GRIPPER'),
}


@lru_cache(maxsize=1)
def dm_joint_ids():
    from autolife_robot_sdk import ROBOT_CONFIG_PATH
    result = {}
    def walk(value):
        if isinstance(value, dict):
            if str(value.get('extra_configs',{}).get('motor_model','')).startswith('DM') and value.get('joint_name'):
                name=value['joint_name']; motor_id=int(value['can_id'])
                if name in result and result[name] != motor_id:
                    raise ValueError('本机DM配置存在重复关节ID：'+name)
                result[name]=motor_id
            for child in value.values():walk(child)
        elif isinstance(value,list):
            for child in value:walk(child)
    walk(json.loads(ROBOT_CONFIG_PATH.read_text()))
    return result


def read_position(controller, name):
    api=controller.robot_api
    if name not in dm_joint_ids():
        return api.get_joint_position(name, clear_cache=True, allow_blocking_read=True)
    # DM官方refresh_motor_status：0x7FF，ID低字节/高字节/0xCC/0。
    # 这是反馈请求，不是使能、位置目标、写寄存器或清错命令。
    # 复用原厂驱动持有的总线及接收线程，禁止另建PCAN连接。
    import can
    driver, motor_id=api._HWAPI__get_motor_driver_by_joint_name(name)
    if motor_id != dm_joint_ids()[name]:
        raise ValueError('DM驱动与本机配置ID不一致：'+name)
    api.get_joint_position(name, clear_cache=True, allow_blocking_read=False)
    bus=driver.get_device_can()
    bus.send(can.Message(arbitration_id=0x7FF, is_extended_id=False,
                         data=[motor_id & 255, (motor_id >> 8) & 255, 0xCC, 0]))
    deadline=time.monotonic()+.3
    while time.monotonic()<deadline:
        value=api.get_joint_position(name, clear_cache=False, allow_blocking_read=False)
        if value is not None:return value
        time.sleep(.005)
    return None


def refresh_group(controller, group):
    """强制刷新驱动缓存，并在两次读数一致后更新本组共享反馈。"""
    _, state_name, names_attr = GROUPS[group]
    names = tuple(getattr(controller, names_attr))
    if not names:
        raise ValueError('该组没有关节')
    readings = []
    # 首轮仅清理既有缓存；部分驱动的clear_cache会先返回旧值再清空。
    # 后两轮才用于停稳判定，防止第一次刷新把旧值/新值混作实体运动。
    for turn in range(3):
        values = []
        for name in names:
            value = read_position(controller, name)
            if value is None or not math.isfinite(float(value)):
                raise ValueError('未收到关节实时位置：' + name)
            values.append(float(value))
        if turn:
            readings.append(values)
        if turn < 2:
            time.sleep(.05)
    tolerance = 2. if group.endswith('gripper') else .35
    if max(abs(a-b) for a,b in zip(*readings)) > tolerance:
        raise ValueError('读取期间关节仍在变化：' + group)
    values=canonical_positions(controller,group,readings[-1])
    state = getattr(controller.shared_state_manager, state_name)
    state.set_current_joint_positions(values)
    return dict(group=group, positions=dict(zip(names, values)),
                raw_positions=dict(zip(names,readings[-1])),
                refreshed_monotonic=time.monotonic(), protocol=2)


def canonical_positions(controller,group,values):
    if not group.endswith('gripper'):return list(values)
    eef=controller.eef_controllers[group.split('_')[0]]
    if type(eef).__name__ != 'GripperType1JointController':return list(values)
    # 复用Type1实例本机端点：电机轴角度与应用的夹爪位置量不是同一单位。
    lb,rb,minimum,maximum=map(float,(eef.lb,eef.rb,eef.min_position,eef.max_position))
    if not all(math.isfinite(v) for v in (lb,rb,minimum,maximum)) or abs(rb-lb)<1e-9:
        raise ValueError('原厂Type1夹爪反馈映射无效')
    ratio=max(0.,min(1.,(float(values[0])-lb)/(rb-lb)))
    return [ratio*(maximum-minimum)+minimum]


def install(mod_names, suffix):
    """只在已有原厂电机子进程内附加接口；每个进程仅读取自己管理的模块。"""
    from autolife_robot_arm.arm_joint_control_base import ArmJointControlBase
    from std_srvs.srv import Trigger
    if getattr(ArmJointControlBase, '_grasp_refresh_installed', False):
        return
    owned = [group for group,(prefix,_,_) in GROUPS.items()
             if any(str(mod) == prefix or str(mod).startswith(prefix+'_') for mod in mod_names)]
    original = ArmJointControlBase.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self._grasp_refresh_services = []
        for group in owned:
            def callback(request, response, group=group):
                try:
                    response.message = json.dumps(refresh_group(self, group), allow_nan=False)
                    response.success = True
                except Exception as exc:
                    response.success = False
                    response.message = str(exc)
                return response
            self._grasp_refresh_services.append(self.node.create_service(
                Trigger, f'/refresh_grasp_feedback_{suffix}_{group}', callback))
    ArmJointControlBase.__init__ = init
    ArmJointControlBase._grasp_refresh_installed = True
