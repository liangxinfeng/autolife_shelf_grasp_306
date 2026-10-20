"""Bounded settled arm arrivals, NOT a stall/obstacle skip mechanism.

Original neck/body/jaw tolerances remain authoritative. Arm residuals may be
accepted only inside a small kinematic envelope, before no jaw-state change.
This is an arrival tolerance check, not an environment collision certificate.
"""
import math

ARM_TOLERANCE_DEG = 2.0
MAX_DISPLACEMENT_M = .005
MAX_TOOL_ROTATION_DEG = 2.0
SETTLE_SEC = .5


class PoseEnvelope:
    """Check joint origins and the existing wrist-relative tool envelope."""
    def __init__(self, urdf_path):
        from autolife_handeye_calibration_306.core import Kinematics, ARMS, TORSO
        self.fk = Kinematics(urdf_path)
        self.arms, self.root = ARMS, TORSO
        children = {joint[0]: child for child, joint in self.fk.joints.items()}
        self.links = {side: [children[name] for name in names]
                      for side, names in ARMS.items()}

    def check(self, actual, target):
        import numpy as np
        largest = 0.
        rotation_max = 0.
        for side, names in self.arms.items():
            if not all(n in actual and n in target for n in names):
                raise ValueError('全身动作手臂反馈不完整')
            for link in self.links[side]:
                a = self.fk.fk(self.root, link, actual)
                b = self.fk.fk(self.root, link, target)
                largest = max(largest, float(np.linalg.norm(a[:3, 3] - b[:3, 3])))
            # Last link is the wrist/gripper mount. Tool center and radius are
            # the same .20m / .09m envelope used by the existing grasp planner.
            cosine = float(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1.) / 2., -1., 1.))
            angle = math.acos(cosine)
            delta = np.linalg.norm((a[:3, 3] + a[:3, :3] @ np.array([.20, 0., 0.])) -
                                   (b[:3, 3] + b[:3, :3] @ np.array([.20, 0., 0.])))
            largest = max(largest, float(delta) + 2. * .09 * math.sin(angle / 2.))
            rotation_max = max(rotation_max, math.degrees(angle))
        if not math.isfinite(largest) or not math.isfinite(rotation_max):
            raise ValueError('运动学包络无效')
        return dict(displacement_m=largest, tool_rotation_deg=rotation_max)


def settled_arm_arrival(bridge, targets, geometry_check):
    """Return receipt or None; never skip a frame or change recorded targets."""
    from control_center_joint_control.joint_definitions import GRIPPER_JOINTS, JOINT_LIMITS
    from autolife_handeye_calibration_306.core import ARMS
    state = bridge._action_state
    if state is None:
        return None

    def reject():
        state.pop('arm_arrival', None)
        return None

    if (not bridge._hardware_enabled or not bridge._hardware_ready or
            not bridge._is_feedback_fresh() or state.get('stage') != 'transition' or
            state.get('progress', 0.) < 1.):
        return reject()
    if 'leg_waist' in state['record']['groups'] and not bridge._leg_waist_enabled:
        return reject()
    body = float(bridge.get_parameter('goal_tolerance_deg').value)
    jaw = float(bridge.get_parameter('gripper_goal_tolerance_deg').value)
    if not (math.isfinite(body) and math.isfinite(jaw) and 0. < body < ARM_TOLERANCE_DEG and jaw > 0.):
        return reject()
    arms = set(ARMS['left'] + ARMS['right'])
    actual = {}
    residuals = {}
    for name, goal in targets.items():
        value = float(bridge._current.get(name, float('nan')))
        command = float(state['commanded_values'].get(name, float('nan')))
        detail = bridge._details.get(name, {})
        speed = float(detail.get('speed', float('nan')))
        if (not all(math.isfinite(v) for v in (goal, value, command, speed)) or
                detail.get('communication_lost', True) or abs(speed) > 2. or
                abs(command-goal) > .05):
            return reject()
        lo, hi = JOINT_LIMITS[name]
        if not lo <= value <= hi or not lo <= goal <= hi:
            return reject()
        error = abs(value-goal)
        limit = ARM_TOLERANCE_DEG if name in arms else jaw if name in GRIPPER_JOINTS else body
        if error > limit + 1e-9:
            return reject()
        if name in arms and error > body + 1e-9:
            residuals[name] = error
        actual[name] = value
    if not residuals:
        return reject()
    # A tolerant arm receipt must not authorize opening/closing the fingers
    # early, either in this frame or in the immediately following frame/loop.
    frames = state['record']['keyframes']
    next_index = state['frame_index'] + 1
    following = frames[next_index] if next_index < len(frames) else (
        frames[0] if state['loop_index'] + 1 < state['record']['loop_count'] else None)
    for name in GRIPPER_JOINTS:
        if name not in targets or name not in state['start_values']:
            return reject()
        if abs(targets[name] - state['start_values'][name]) > jaw:
            return reject()
        if following and abs(following['joints_deg'][name] - targets[name]) > jaw:
            return reject()
    stamp = float(bridge._feedback_monotonic)
    if not math.isfinite(stamp):
        return reject()
    key = (state['loop_index'], state['frame_index'], tuple(sorted(targets.items())))
    stable = state.get('arm_arrival')
    if (stable is None or stable['key'] != key or stamp < stable['last'] or
            stamp - stable['last'] > .25 or
            any(abs(v - stable['anchor'][n]) > .2 for n, v in actual.items())):
        state['arm_arrival'] = dict(key=key, start=stamp, last=stamp, count=1, anchor=actual)
        return None
    if stamp == stable['last']:
        return None
    stable['last'] = stamp
    stable['count'] += 1
    if stable['count'] < 4 or stamp - stable['start'] < SETTLE_SEC:
        return None
    try:
        geometry = geometry_check(actual, targets)
        distance = float(geometry['displacement_m'])
        angle = float(geometry['tool_rotation_deg'])
        if not (math.isfinite(distance) and math.isfinite(angle) and
                0. <= distance <= MAX_DISPLACEMENT_M + 1e-12 and
                0. <= angle <= MAX_TOOL_ROTATION_DEG + 1e-9):
            return reject()
    except (ValueError, KeyError, TypeError, OSError, ImportError):
        return reject()
    return dict(kind='settled_arm_tolerance', errors_deg=residuals,
                displacement_m=distance, tool_rotation_deg=angle,
                measured=actual, frame_index=state['frame_index'] + 1,
                loop_index=state['loop_index'], feedback_monotonic=stamp)
