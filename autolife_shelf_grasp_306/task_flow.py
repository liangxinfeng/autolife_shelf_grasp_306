"""Small serial workflow protocol. A failed step never advances the cursor."""
from copy import deepcopy
import uuid
import math

KINDS = {'navigate': '导航', 'grasp': '抓取', 'place': '放置', 'action': '固定动作', 'base':'底盘定量'}


def full_record(record):
    from control_center_joint_control.action_library import normalize_action_record
    from control_center_joint_control.joint_definitions import ALL_JOINTS, GROUP_ORDER, JOINT_LIMITS
    if set(record.get('groups', [])) != set(GROUP_ORDER):
        raise ValueError('动作必须记录全部关节，请在固定动作页重新记录')
    for frame in record.get('keyframes', []):
        values = frame.get('joints_deg', {})
        if set(values) != set(ALL_JOINTS): raise ValueError('关键帧缺少全身关节，未自动补零')
        for name, value in values.items():
            lo, hi = JOINT_LIMITS[name]
            if not math.isfinite(float(value)) or not lo <= float(value) <= hi:
                raise ValueError('关键帧关节超限：' + name)
    return normalize_action_record(record)


def validate(record):
    rows = record.get('steps', [])
    if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
        raise ValueError('流程需要1～100个步骤')
    result = []
    for row in rows:
        kind = row.get('kind'); target = str(row.get('target', '')).strip()
        if kind not in KINDS: raise ValueError('不支持的任务类型')
        if kind not in ('place','base') and not target: raise ValueError('请填写航点、物品或动作名称')
        if len(target) > 160: raise ValueError('名称过长')
        from .step_posture import normalize
        item=dict(kind=kind,target=target);posture=normalize(row.get('posture'))
        if kind=='base':
            from .base_sequence import normalize as base_normalize
            item['base']=base_normalize(row.get('base'));item['target']='底盘定量动作'
        if posture is not None:item['posture']=posture
        result.append(item)
    return dict(version=1, steps=result)


class Cursor:
    def __init__(self, record):
        self.record = deepcopy(validate(record)); self.run_id = uuid.uuid4().hex
        self.index = 0; self.active = True; self.reason = ''

    @property
    def token(self): return f'{self.run_id}:{self.index}'

    @property
    def step(self): return self.record['steps'][self.index]

    def complete(self, token):
        if not self.active or token != self.token: return False
        self.index += 1
        if self.index == len(self.record['steps']): self.active = False
        return True

    def stop(self, reason):
        self.active = False; self.reason = reason

    def retry(self):
        if self.active or not 0<=self.index<len(self.record['steps']):
            raise ValueError('没有已停止的当前步骤可重试')
        # New attempt identity rejects delayed navigation/action completions.
        self.run_id=uuid.uuid4().hex;self.active=True;self.reason=''


def select_instance(items, target, products):
    return select_candidates(items, target, products)[0]


def select_candidates(items, target, products, semantic=False):
    from .auto_grasp import rank_candidates
    product_ids = {p['id'] for p in products if p['name'] == target}
    matches = items if semantic else [v for v in items if v.get('product_id') in product_ids or v.get('name') == target]
    if not matches: raise ValueError('未识别到任务目标：' + target)
    return rank_candidates(matches)


class NavigationSettling(ValueError):
    """Fresh stationary feedback is not ready yet; retry only before dispatch."""


def navigation_pose(command, context, samples, now):
    """A generic workflow snapshots fresh stationary joints, not a fake grasp receipt."""
    run_id = context.get('run_id', '')
    try:
        prefix, index = command['leg'].split(':'); index = int(index)
        rows = validate(context['record'])['steps']
        if not 0 <= index < len(rows): raise ValueError()
        row = rows[index]
        if (prefix != run_id or command.get('run_id') != run_id or
                command.get('map') != context.get('map') or row['kind'] != 'navigate' or
                row['target'] != command.get('name')): raise ValueError()
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError('导航请求与本轮任务快照不一致') from exc
    recent = [(t,q) for t,q in samples if 0 <= now-t <= .8]
    if len(recent)<3 or now-recent[-1][0]>.3 or recent[-1][0]-recent[0][0]<.4:
        raise NavigationSettling('等待关节连续反馈与停稳后再开始任务')
    q = recent[-1][1]
    from control_center_joint_control.joint_definitions import BODY_JOINTS
    if not set(BODY_JOINTS).issubset(q): raise ValueError('全身关节反馈不完整')
    for _, sample in recent:
        if set(sample)!=set(q): raise ValueError('关节反馈不完整')
        for n,v in sample.items():
            if not math.isfinite(v) or abs(v-q[n])>(3. if 'Gripper' in n else 1.):
                if not math.isfinite(v):raise ValueError('关节反馈含无效数值')
                raise NavigationSettling('身体或手臂尚未停稳，未开始导航')
    return dict(q)
