"""Waypoint demonstration sequencing. Every handoff needs a successful receipt."""
import json,uuid
from PyQt5 import QtWidgets as Q
from .core import atomic_json
from .demo_config import load,require_drink
from .navigation_logic import select_product


class DemoNavigation:
    def start_navigation_demo(self):
        if self.busy or self.motion or self.demo_active:return
        try:
            route=self.navigation.selection();settings=load(self.root)
        except Exception as exc:self.status.setText(str(exc));return
        if Q.QMessageBox.question(self,'确认演示抓取',
            f'复位并张开夹爪 → {route["shelf"]} → 抓取{route["product_name"]} → {route["front"]} → 放置到桌面。\n'
            '请先取出夹爪中的物品，机器人应已在前台附近。到货架后重新识别规划；导航失败不会继续抓放。\n'
            '底盘使用控制中心避障；机械臂抓取仍需您确认货架及周围空间安全，全程准备实体急停。',
            defaultButton=Q.QMessageBox.No)!=Q.QMessageBox.Yes:return
        self.demo_context={**route,'settings':settings,'run_id':uuid.uuid4().hex}
        atomic_json(self.run/'demo_run.json',self.demo_context)
        for name in ('stop','demo_base_result.json','demo_grasp_result.json','demo_shelf_result.json','observation_attempt.json','demo_observe_result.json'):(self.run/name).unlink(missing_ok=True)
        self.demo_active=True;self.demo_phase='prepare';self.send('prepare')

    def navigate_demo(self,leg):
        if not self.demo_active or (self.run/'stop').exists():return
        self.demo_phase=leg;self.live();self.busy=True;self.controls()
        context=self.demo_context
        self.navigation.command('navigate',name=context['shelf' if leg=='to_shelf' else 'front'],
                                map=context['map'],run_id=context['run_id'],leg=leg)
        self.status.setText('正在导航到货架…' if leg=='to_shelf' else '持物导航回前台…')

    def navigation_done(self,data):
        if not self.demo_context or data.get('run_id')!=self.demo_context['run_id']:return
        if not self.demo_active:
            self.busy=False;self.cancel_pending=False;self.controls();return
        if data.get('leg')!=self.demo_phase:return
        self.busy=False
        if not data.get('success'):self.demo_failure(data.get('text','导航未成功'));return
        if data.get('map')!=self.demo_context['map']:self.demo_failure('导航地图不一致');return
        if self.demo_phase=='to_shelf':
            atomic_json(self.run/'demo_shelf_result.json',{**data,'status':'completed'})
            self.demo_phase='scan';self.send('scan')
        else:
            atomic_json(self.run/'demo_base_result.json',{**data,'status':'completed'})
            self.demo_phase='planning';self.send('demo_place')

    def demo_scene_ready(self):
        try:
            instance=select_product(self.items,self.demo_context['product_id'])
            row=next(i for i,v in enumerate(self.items) if v['id']==instance)
            self.list.setCurrentRow(row);self.demo_phase='grasp_plan';self.send('plan',id=instance)
        except Exception as exc:self.demo_failure(str(exc))

    def demo_plan_ready(self,data):
        try:
            if data['blockers']:raise ValueError('；'.join(data['blockers']))
            if data['target']['product_id']!=self.demo_context['product_id']:raise ValueError('商品与演示选择不一致')
            full=json.loads((self.run/data['filename']).read_text());require_drink(full['product'])
            if full.get('geometry',{}).get('image_cropped'):raise ValueError('需完整看到饮料，才能确定放置高度')
            self.demo_context['grasp_plan_id']=data['plan_id'];atomic_json(self.run/'demo_run.json',self.demo_context)
            self.demo_phase='grasp';self.launch_motion(data['filename'])
        except Exception as exc:self.demo_failure(str(exc))

    def demo_failure(self,text):
        self.stop_motion();self.busy=False;self.cancel_pending=False
        self.status.setText('演示已停止：'+text);self.controls()
