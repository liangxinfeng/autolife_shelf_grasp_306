"""Explicit standalone placement provenance, without fabricated navigation receipts."""
import json
import math
import uuid
from .core import atomic_json
from .pose_handoff import HELD_JAW_TOLERANCE_DEG
from .arm_selection import gripper_name,require_same_arm


def held_grasp(run):
    try:
        record=json.loads((run/'held_object.json').read_text())
        pid=record.get('plan_id','')
        if len(pid)!=32 or any(c not in '0123456789abcdef' for c in pid):raise ValueError()
        grasp=json.loads((run/('plan_'+pid+'.json')).read_text())
        if record.get('status')!='completed' or grasp.get('run_id')!=run.name or grasp.get('plan_id')!=pid:
            raise ValueError()
        arm=require_same_arm(grasp,record)
        if not math.isfinite(float(record['q'][gripper_name(arm)])):raise ValueError()
        return grasp,record
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise ValueError('没有有效持物记录，请在本窗口先完成一次抓取；复位后需重新抓取') from exc


def new_standalone_request(run,settings):
    from .demo_config import require_placeable
    grasp,record=held_grasp(run);require_placeable(grasp['product'])
    if grasp.get('geometry',{}).get('image_cropped'):
        raise ValueError('本次抓取未完整看到瓶身，无法确认瓶底位置，不能自动放置')
    context=dict(mode='standalone',run_id=uuid.uuid4().hex,session_id=run.name,
                 grasp_plan_id=grasp['plan_id'],product_id=grasp['product']['id'],settings=settings)
    atomic_json(run/'standalone_place.json',context)
    for name in ('observation_attempt.json','demo_observe_result.json','stop'):
        (run/name).unlink(missing_ok=True)
    return context


def standalone_source(run):
    context=json.loads((run/'standalone_place.json').read_text())
    grasp,record=held_grasp(run)
    if (context.get('mode')!='standalone' or context.get('session_id')!=run.name or
            context.get('grasp_plan_id')!=grasp['plan_id'] or
            context.get('product_id')!=grasp['product']['id']):
        raise ValueError('独立放置请求与当前持物记录不一致')
    return context,grasp,record


def check_held_gripper(q,record):
    jaw=gripper_name(record.get('arm','left'))
    held=float(record.get('q',{}).get(jaw,float('nan')))
    current=float(q.get(jaw,float('nan')))
    if not math.isfinite(held) or not math.isfinite(current) or abs(current-held)>HELD_JAW_TOLERANCE_DEG:
        raise ValueError('夹爪状态与抓取结束时不同，请确认物品，不能自动放置')
