"""人工支撑确认后的S3颈部抬离流程；不是通用碰撞豁免。

仅复位入口可生成此阶段。保持其他部位，单调减小颈部俯仰到15°；
仅允许初始已存在的两类头颈/躯干接触，清除后不允许再次出现。
随后必须通过严格碰撞检查，才可执行双臂复位和打开夹爪。
"""
import hashlib
import json
import math
from pathlib import Path

PITCH = 'Joint_Neck_Pitch'
ROLL = 'Joint_Neck_Roll'
YAW = 'Joint_Neck_Yaw'
VERSION = 's3_supported_neck_lift_v1'
TARGET = 15.
MODEL_HASH = '118ba602bd1b0a0964d6d149d095f0825231613218faf7b38f08b35742e72e17'
SRDF_HASH = '31d787a5dc2e616c1daf5820c1a75c420a8719cca578e40428e1cfad47c78a04'
TORSO = 'Link_Waist_Yaw_to_Shoulder_Inner_0'
ALLOWED = {frozenset((TORSO, n)) for n in
           ('Link_Neck_Pitch_to_Neck_Yaw_0', 'Link_Neck_Yaw_to_Head_0')}


def pair_key(text):
    return frozenset(part.strip() for part in text.split('<->'))


def validate_source(source):
    if not source or not all(math.isfinite(float(v)) for v in source.values()):
        raise ValueError('颈部恢复反馈无效')
    if not (15. < source[PITCH] <= 40. and abs(source[ROLL]) <= 2. and abs(source[YAW]) <= 10.):
        raise ValueError('头颈姿态超出已验证的支撑恢复范围，请通过关节界面人工处理')


def validate_segment(segment, source):
    meta = segment.get('supported_neck_recovery', {})
    if (meta.get('version') != VERSION or meta.get('operator_supported') is not True
            or meta.get('model_sha256') != MODEL_HASH or meta.get('srdf_sha256') != SRDF_HASH):
        raise ValueError('缺少现场支撑确认或颈部恢复版本不匹配')
    validate_source(source)
    points = segment.get('waypoints', [])
    if len(points) != 2 or points[0] != source:
        raise ValueError('颈部恢复起点不匹配')
    expected = dict(source); expected[PITCH] = TARGET
    if points[1] != expected:
        raise ValueError('颈部恢复只能改变俯仰角到15°，不能联动其他关节')
    pairs = meta.get('initial_pairs', [])
    if not pairs or any(pair_key(p) not in ALLOWED for p in pairs):
        raise ValueError('颈部恢复中包含未经验证的碰撞部件')


def verify_model(kin):
    model = Path(kin.urdf_path)
    srdf = model.parent / 'robot_v3_1.srdf'
    if (hashlib.sha256(model.read_bytes()).hexdigest() != MODEL_HASH
            or hashlib.sha256(srdf.read_bytes()).hexdigest() != SRDF_HASH):
        raise ValueError('颈部恢复模型已改变，须重新核对，不能沿用旧恢复范围')


def verify_installed_model():
    from ament_index_python.packages import get_package_share_directory
    from types import SimpleNamespace
    path=Path(__file__).resolve().parents[1]/'urdf/robot_v3_1_simplified.urdf'
    verify_model(SimpleNamespace(urdf_path=str(path)))


def build_plan(planner, cfg, source, operator_supported=False):
    if cfg.get('supported_neck_recovery_enabled') is not True:
        raise ValueError('本机未配置经验证的颈部支撑恢复')
    validate_source(source)
    guard = planner.guard; kin = guard.kin
    verify_model(kin)
    from autolife_handeye_calibration_306.auto_plan_worker import GROUPS
    def configuration(values):
        return kin.q_from_feedback({g:[values[n] for n in names] for g,names in GROUPS.items()}, clip=False)
    initial = kin.collision_pairs(configuration(source))
    if not initial or any(pair_key(p) not in ALLOWED for p in initial):
        raise ValueError('存在其他自碰撞，不能按颈部支撑恢复处理')
    if not operator_supported:
        raise ValueError('头部低垂贴身：请现场支撑头部、确认净空后，手动点击“复位并准备”；自动任务不能自行执行接触恢复')
    endpoint = dict(source); endpoint[PITCH] = TARGET
    original = kin.collision_pairs
    initial_keys = {pair_key(p) for p in initial}
    cleared = set(); count = 0
    def checked_pairs(q):
        pairs = original(q); keys = {pair_key(p) for p in pairs}
        if keys - initial_keys or keys & cleared:
            raise ValueError('抬头过程中新增碰撞或再次接触，不能自动恢复')
        cleared.update(initial_keys - keys)
        # 只在该单调抬离路径中认可已确认的初始接触；其余检查仍由guard执行。
        return []
    try:
        kin.collision_pairs = checked_pairs
        guard.cache.clear()
        steps = math.ceil((source[PITCH]-TARGET)/.25)
        for i in range(steps+1):
            values = dict(source); values[PITCH] = source[PITCH]+(TARGET-source[PITCH])*i/steps
            guard.check(values); count += 1
    finally:
        kin.collision_pairs = original
        guard.cache.clear()
    guard.check(endpoint)  # 恢复终点必须是严格的无碰撞姿态。
    result = planner.reset_plan(endpoint)  # 后续全身复位完全沿用原有检查。
    segment = {'name':'先低速抬头脱离初始接触（需现场支撑）', 'waypoints':[dict(source),endpoint],
               'supported_neck_recovery':{'version':VERSION, 'operator_supported':True,
                    'model_sha256':MODEL_HASH, 'srdf_sha256':SRDF_HASH, 'initial_pairs':initial}}
    validate_segment(segment, source)
    result['source_q'] = dict(source)
    result['segments'].insert(0, segment)
    result['supported_neck_recovery_checked_states'] = count
    return result


def execute_segment(node, segment, source, checked, hold, now, progress):
    """与本机已验证的低速抬头一致；仅发送颈部，停止后不关电。"""
    validate_segment(segment, source)
    from autolife_handeye_calibration_306.core import NECK
    from control_center_joint_control.joint_definitions import GROUPS
    q = checked()
    if max(abs(q[n]-source[n]) for n in source if 'Gripper' not in n) > .7:
        raise ValueError('使能后姿态变化过大，未开始颈部恢复')
    begin = dict(q); command = [q[n] for n in NECK]
    start = previous = now(); last_progress = -1.; stable = None
    last_improvement = start; best = q[PITCH]
    try:
        while True:
            q = checked(); stamp = now()
            dt = min(.1, max(0., stamp-previous)); previous = stamp
            fixed = [n for n in source if n != PITCH]
            if (max(abs(q[n]-begin[n]) for n in fixed) > .7
                    or q[PITCH] > begin[PITCH]+.5 or q[PITCH] < TARGET-.6):
                raise ValueError('颈部恢复偏离预检范围，已停止，未开始双臂复位')
            if q[PITCH] < best-.1:
                best = q[PITCH]; last_improvement = stamp
            if stamp-start > 100 or (stamp-last_improvement > 5 and abs(q[PITCH]-TARGET) > .4):
                raise ValueError('颈部没有跟随恢复目标，已保持，未开始双臂复位')
            command[1] = max(TARGET, command[1]-dt, q[PITCH]-.35)
            node._publish_vendor(node._whole_body_pub, {GROUPS['neck']['command_key']:list(command)})
            if stamp-last_progress >= .2:
                progress({'stage':segment['name'], 'measured_pitch_deg':q[PITCH],
                          'target_pitch_deg':TARGET, 'command_pitch_deg':command[1],
                          'progress':max(0., min(1., (begin[PITCH]-q[PITCH])/(begin[PITCH]-TARGET)))})
                last_progress = stamp
            if abs(q[PITCH]-TARGET) <= .3:
                stable = stamp if stable is None else stable
                if stamp-stable >= .5:
                    return dict(q)
            else:
                stable = None
    finally:
        hold(node, ['neck'])


def verify_clear_in_worker(cfg, values, checked, now):
    """执行器为系统Python，几何复核使用已有感知环境，不重复安装依赖。"""
    import subprocess
    import time
    p = subprocess.Popen([cfg['python'], '-m', __name__, '--verify-clear'],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True)
    try:
        p.stdin.write(json.dumps({'config':cfg, 'q':values})); p.stdin.close(); p.stdin = None
        deadline = time.monotonic()+10
        while p.poll() is None:
            current = checked()
            if max(abs(current[n]-values[n]) for n in values) > .7:
                raise ValueError('恢复后几何复核期间姿态改变')
            if time.monotonic() > deadline:
                raise ValueError('恢复后几何复核超时，未开始双臂复位')
        out, err = p.communicate()
        if p.returncode:
            raise ValueError('恢复后仍未通过完整碰撞检查：'+(out+err)[-1500:])
        if json.loads(out).get('clear') is not True:
            raise ValueError('恢复后几何复核结果无效')
    finally:
        if p.poll() is None:
            p.kill(); p.communicate()


if __name__ == '__main__':
    import sys
    import contextlib
    if sys.argv[1:] != ['--verify-clear']:
        raise SystemExit('仅支持无动作几何核对')
    request = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        from .grasp_collision import CollisionGuard
        guard = CollisionGuard(request['config'])
        verify_model(guard.kin)
        guard.check(request['q'])
    print(json.dumps({'clear':True}))
