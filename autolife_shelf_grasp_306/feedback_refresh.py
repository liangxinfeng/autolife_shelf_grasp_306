"""规划/使能前主动读取各组关节；新消息时间不能代替真实硬件刷新。"""
import json
import math
import time

GROUPS = ('neck','leg_waist','left_arm','right_arm','left_gripper','right_gripper')


def await_snapshot(snapshot, spin, positions, timeout=1.):
    """确认ROS反馈已跟上主动读数，避免随后仍锁存刷新前的消息。"""
    deadline = time.monotonic()+timeout
    while True:
        current = snapshot()
        if all(name in current and math.isfinite(float(current[name])) and
               abs(float(current[name])-value) <= (2. if 'Gripper' in name else .5)
               for name,value in positions.items()):
            return
        if time.monotonic() >= deadline:
            raise ValueError('主动读取与关节反馈尚未一致，未发送使能目标')
        spin()


def refresh(node, spin, suffix='0_345', timeout=5.):
    from std_srvs.srv import Trigger
    clients = {g: node.create_client(Trigger, f'/refresh_grasp_feedback_{suffix}_{g}') for g in GROUPS}
    deadline = time.monotonic()+timeout
    try:
        while not all(c.service_is_ready() for c in clients.values()):
            if time.monotonic() >= deadline:
                missing = [g for g,c in clients.items() if not c.service_is_ready()]
                raise ValueError('原厂只读反馈刷新接口未就绪：'+'、'.join(missing))
            spin()
        futures = {g:c.call_async(Trigger.Request()) for g,c in clients.items()}
        while not all(f.done() for f in futures.values()):
            if time.monotonic() >= deadline:
                raise ValueError('读取真实关节位置超时，尚未使能或开始动作')
            spin()
        positions = {}
        for group, future in futures.items():
            result = future.result()
            if result is None or not result.success:
                raise ValueError('真实关节位置读取失败：'+group+' / '+str(getattr(result,'message','无响应')))
            data = json.loads(result.message)
            values = data.get('positions',{})
            if data.get('protocol') != 2 or data.get('group') != group or not values:
                raise ValueError('关节刷新响应无效：'+group)
            if not all(math.isfinite(float(v)) for v in values.values()):
                raise ValueError('关节刷新数据非有限值：'+group)
            if positions.keys() & values.keys():
                raise ValueError('关节刷新响应重复')
            positions.update({k:float(v) for k,v in values.items()})
        return positions
    finally:
        for client in clients.values():
            node.destroy_client(client)
