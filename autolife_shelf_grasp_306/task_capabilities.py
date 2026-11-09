"""Perception is optional for saved actions, posture and waypoint tasks."""
VISUAL_KINDS = frozenset(('grasp', 'place'))


def uses_perception(record):
    return any(row['kind'] in VISUAL_KINDS for row in (record or {}).get('steps', []))


def require_perception(owner, record):
    if not uses_perception(record) or owner.ready:
        return
    labels = {'grasp': '视觉抓取', 'place': '视觉放置'}
    steps = '、'.join(f"第{i+1}步{labels[row['kind']]}" for i, row in enumerate(record['steps'])
                     if row['kind'] in VISUAL_KINDS)
    reason = getattr(owner, 'vision_error', '') or '识别后台尚未就绪'
    raise ValueError(f'{steps}需要视觉能力：{reason}。预录制动作、航点导航可独立使用。')


def needs_grasp_preparation(record):
    return any(row['kind'] == 'grasp' for row in (record or {}).get('steps', []))
