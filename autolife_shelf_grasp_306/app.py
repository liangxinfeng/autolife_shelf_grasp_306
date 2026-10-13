"""Native Qt operator UI. CPU/GPU work never runs in the GUI event loop."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from PyQt5 import QtCore as C, QtGui as G, QtWidgets as Q
from .no_wheel import NoWheelDoubleSpinBox
from .demo_navigation import DemoNavigation
from .core import atomic_json
from .live_overlay import decode_packet

STYLE='''
QWidget {background:#101821;color:#e8eff6;font-family:"Noto Sans CJK SC";font-size:15px;}
QLabel#title {font-size:26px;font-weight:700;} QLabel#muted{color:#91a4b5;}
QFrame#panel {background:#172330;border:1px solid #2d4050;border-radius:14px;}
QPushButton {background:#273c4d;border:1px solid #3c5566;border-radius:9px;padding:12px;min-height:24px;}
QPushButton:hover{background:#36596b;} QPushButton:disabled{color:#647787;background:#1b2936;border-color:#263745;}
QPushButton#primary{background:#147e75;border-color:#21ac98;font-weight:700;}
QPushButton#stop{background:#80393d;border-color:#b45660;}
QListWidget {background:#101b25;border:0;border-radius:8px;}
QListWidget::item {padding:16px;border-bottom:1px solid #243545;}
QListWidget::item:selected{background:#245867;border-radius:6px;}
QLineEdit,QDoubleSpinBox,QComboBox {padding:8px;background:#223544;border:1px solid #385264;border-radius:6px;}
QToolTip {background:#243544;color:white;border:1px solid #507080;}
'''


class ImageView(Q.QWidget):
    clicked=C.pyqtSignal(float,float)
    def __init__(self):
        super().__init__();self.pix=G.QPixmap();self.boxes=[];self.path=[];self.shape=[720,1280,3];self.selected=None
        self.show_masks=True
        self.notice=''
        self.setMinimumSize(320,220);self.rect_image=C.QRectF()

    def paintEvent(self,event):
        painter=G.QPainter(self);painter.fillRect(self.rect(),G.QColor('#090e14'))
        if self.pix.isNull():
            painter.setPen(G.QColor('#8ea3b2'));painter.drawText(self.rect(),C.Qt.AlignCenter,'等待头部 RGB 图像');return
        fitted=self.pix.size().scaled(self.size(),C.Qt.KeepAspectRatio)
        r=C.QRectF((self.width()-fitted.width())/2,(self.height()-fitted.height())/2,fitted.width(),fitted.height());self.rect_image=r
        painter.drawPixmap(r,self.pix,C.QRectF(self.pix.rect()))
        sx,sy=r.width()/self.shape[1],r.height()/self.shape[0]
        def point(x,y):return C.QPointF(r.x()+x*sx,r.y()+y*sy)
        if self.show_masks:
            painter.save();painter.setClipRect(r);painter.setRenderHint(G.QPainter.Antialiasing)
            palette=['#45d9bd','#65b9ff','#d491ff','#ff9e80','#afd76c','#ee88be']
            # Selected object drawn last, matching the yellow selection box.
            for item in sorted(self.boxes,key=lambda v:v['id']==self.selected):
                outline=G.QPainterPath();outline.setFillRule(C.Qt.OddEvenFill)
                for polygon in item.get('mask_contours',[]):
                    if len(polygon)<3:continue
                    outline.moveTo(point(*polygon[0]))
                    for xy in polygon[1:]:outline.lineTo(point(*xy))
                    outline.closeSubpath()
                selected=item['id']==self.selected
                color=G.QColor('#ffc86c' if selected else palette[int(item['id'])%len(palette)])
                painter.setPen(G.QPen(color,2.5 if selected else 1.5))
                color.setAlpha(90 if selected else 65);painter.setBrush(color);painter.drawPath(outline)
            painter.restore()
        for box in self.boxes:
            x0,y0,x1,y1=box['box'];painter.setPen(G.QPen(G.QColor('#ffc86c' if box['id']==self.selected else '#45d9bd'),3))
            painter.drawRect(C.QRectF(point(x0,y0),point(x1,y1)));painter.drawText(point(x0,y0-8),f'#{box["id"]} '+box['name'])
        if self.path:
            painter.setPen(G.QPen(G.QColor('#f7c475'),3));points=[point(*p) for p in self.path]
            for a,b in zip(points,points[1:]):painter.drawLine(a,b)
            for i,p in enumerate(points):
                painter.setBrush(G.QColor('#f7c475'));painter.drawEllipse(p,5,5);painter.drawText(p+C.QPointF(9,-9),str(i+1))
        if self.notice:
            painter.setPen(G.QColor('#ffc86c'));painter.drawText(r.adjusted(10,8,-10,-8),C.Qt.AlignTop|C.Qt.AlignLeft,self.notice)

    def mousePressEvent(self,event):
        if self.rect_image.contains(event.localPos()):
            self.clicked.emit((event.x()-self.rect_image.x())/self.rect_image.width(),(event.y()-self.rect_image.y())/self.rect_image.height())


class ProductEditor(Q.QDialog):
    """One labelled form for enrollment and parameter-only edits."""
    def __init__(self,parent,product=None,catalog=None,category='general',editing=False):
        super().__init__(parent);self.setWindowTitle('修改参数' if editing else '录入 / 追加样本')
        self.resize(460,400);form=Q.QFormLayout(self)
        self.name=Q.QComboBox();self.name.setEditable(True);self.name.addItem('')
        for p in catalog or []:self.name.addItem(p['name'])
        self.shape=Q.QComboBox()
        for label,value in [('圆柱 / 瓶罐','cylinder'),('非圆柱实体 / 模型','solid'),('袋子 / 夹袋口','bag_edge')]:self.shape.addItem(label,value)
        self.category=category # legacy metadata only; both modes share all SKUs
        self.width=Q.QDoubleSpinBox();self.width.setRange(15,350)
        self.height=Q.QDoubleSpinBox();self.height.setRange(40,400)
        self.depth=Q.QDoubleSpinBox();self.depth.setRange(1,150)
        self.width.valueChanged.connect(lambda value:self.depth.setValue(value) if self.shape.currentData()=='cylinder' else None)
        for spin in (self.width,self.height,self.depth):spin.setSuffix(' mm');spin.setDecimals(1)
        self.width_label=Q.QLabel();self.depth_label=Q.QLabel()
        self.shape.currentIndexChanged.connect(self.fields)
        form.addRow('名称',self.name);form.addRow('物品类型',self.shape)
        form.addRow(self.width_label,self.width);form.addRow('整体高度（参考）',self.height);form.addRow(self.depth_label,self.depth)
        hint=Q.QLabel('两种模式共用 · 只修改参数，不新增图片' if editing else '两种模式共用。保存同名商品：更新参数并追加当前图像');hint.setWordWrap(True);form.addRow(hint)
        self.load(product or dict(name='',diameter_m=.07,height_m=.12,depth_m=.04,geometry='solid',category=category))
        def picked(text):
            p=next((p for p in catalog or [] if p['name']==text),None)
            if p:self.load(p)
        if not editing:self.name.textActivated.connect(picked)
        buttons=Q.QDialogButtonBox(Q.QDialogButtonBox.Save|Q.QDialogButtonBox.Cancel)
        buttons.button(Q.QDialogButtonBox.Save).setText('保存');buttons.button(Q.QDialogButtonBox.Cancel).setText('取消')
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);form.addRow(buttons)

    def fields(self):
        kind=self.shape.currentData();self.width.setMaximum(350 if kind=='bag_edge' else 90)
        self.width_label.setText('袋体整体宽度' if kind=='bag_edge' else '夹取宽度 / 直径（左右）')
        self.depth_label.setText('袋口局部厚度' if kind=='bag_edge' else '可见面至背面厚度')
        self.depth.setEnabled(kind!='cylinder');self.depth.setRange(1 if kind=='bag_edge' else 5,20 if kind=='bag_edge' else 150)
        if kind=='cylinder':self.depth.setValue(self.width.value())

    def load(self,p):
        self.name.setEditText(p['name']);self.shape.setCurrentIndex(max(0,self.shape.findData(p.get('geometry','cylinder'))));self.fields()
        self.width.setValue(p['diameter_m']*1000);self.height.setValue(p['height_m']*1000)
        self.depth.setValue(p.get('depth_m',p['diameter_m'])*1000)
        category=p.get('category','beverage' if 'geometry' not in p else 'general')
        self.category=category

    def values(self):
        return dict(name=self.name.currentText(),diameter_mm=self.width.value(),height_mm=self.height.value(),
                    geometry=self.shape.currentData(),depth_mm=self.depth.value(),category=self.category)


class CatalogDialog(Q.QDialog):
    def __init__(self,owner):
        super().__init__(owner);self.owner=owner;self.products=[];self.setWindowTitle('商品库 · 参数和样本')
        self.setWindowModality(C.Qt.WindowModal);self.resize(800,530)
        layout=Q.QVBoxLayout(self);self.names=Q.QComboBox();layout.addWidget(self.names)
        self.info=Q.QLabel();layout.addWidget(self.info)
        self.images=Q.QListWidget();self.images.setViewMode(Q.QListView.IconMode);self.images.setIconSize(C.QSize(128,128))
        self.images.setResizeMode(Q.QListView.Adjust);layout.addWidget(self.images,1)
        row=Q.QHBoxLayout();layout.addLayout(row);self.buttons=[]
        for label,fn in [('修改参数',self.edit),('追加当前图像',self.add),('删除选中样本',self.remove),('删除整个商品',self.remove_product)]:
            b=Q.QPushButton(label);b.clicked.connect(fn);row.addWidget(b);self.buttons.append(b)
        note=Q.QLabel('追加前请先在主界面选中物品。删除会移除匹配特征，原图保留用于恢复。');note.setWordWrap(True);layout.addWidget(note)
        close=Q.QPushButton('关闭');close.clicked.connect(self.close);layout.addWidget(close)
        self.names.currentIndexChanged.connect(self.show_product)

    def current(self):return next((p for p in self.products if p['id']==self.names.currentData()),None)
    def refresh(self,products,selected=None):
        selected=selected or self.names.currentData();self.products=products;self.names.blockSignals(True);self.names.clear()
        for p in products:self.names.addItem(p['name'],p['id'])
        self.names.setCurrentIndex(max(0,self.names.findData(selected)));self.names.blockSignals(False);self.show_product()
    def show_product(self):
        self.images.clear();p=self.current()
        for b in self.buttons:b.setEnabled(bool(p) and not self.owner.busy)
        if not p:self.info.setText('商品库为空');return
        self.info.setText(f'宽 {p["diameter_m"]*1000:g} mm · 高 {p["height_m"]*1000:g} mm · {len(p["examples"])} 张样本')
        root=(self.owner.root/'data').resolve()
        for i,relative in enumerate(p['examples']):
            path=(root/relative).resolve()
            if not path.is_relative_to(root):continue
            item=Q.QListWidgetItem(G.QIcon(str(path)),f'样本 {i+1}');item.setData(C.Qt.UserRole,i);self.images.addItem(item)
    def edit(self):
        p=self.current()
        if p:self.owner.enrollment(p,edit_only=True)
    def add(self):
        if self.owner.selected_id() is None or not self.owner.frozen:
            Q.QMessageBox.information(self,'请先选物品','关闭商品库，识别并选中要追加的物品，再回来追加图像。');return
        p=self.current();self.close();self.owner.enrollment(p)
    def remove(self):
        p=self.current();item=self.images.currentItem()
        if not p or item is None:return
        if Q.QMessageBox.question(self,'删除样本','删除这张样本及对应匹配特征？\n全部删除后，该商品将暂时无法自动匹配。',defaultButton=Q.QMessageBox.No)==Q.QMessageBox.Yes:
            self.owner.send('remove_sample',product_id=p['id'],index=item.data(C.Qt.UserRole))

    def remove_product(self):
        p=self.current()
        if not p or self.owner.busy or self.owner.motion:return
        if Q.QMessageBox.question(self,'删除整个商品',
            f'删除“{p["name"]}”及全部匹配样本？\n两种识别模式同步移除。历史商品库和样本原图保留，可恢复。',
            defaultButton=Q.QMessageBox.No)==Q.QMessageBox.Yes:
            self.owner.send('remove_product',product_id=p['id'])


class Window(DemoNavigation,Q.QWidget):
    def __init__(self,root,run,start_worker=True):
        super().__init__();self.root=Path(root);self.run=Path(run);self.buffer=b'';self.request=0;self.busy=True
        self.ready=False;self.frozen=False;self.items=[];self.blockers=[];self.process=None;self.cancel_pending=False
        # frozen retains the legacy "has selectable recognition" contract, but
        # never freezes RGB. Display tracking is independent of executable plans.
        self.overlay_revision=None;self.path_revision=None
        self.pending_plan=None;self.expected_target=None;self.picking_rim=False
        self.catalog_dialog=None;self.chassis_dialog=None
        self.demo_active=False;self.placement_active=False;self.demo_phase=None;self.demo_context=None
        self.setWindowTitle('云蝶 · 抓取与任务流程')
        available=Q.QApplication.primaryScreen().availableGeometry()
        self.resize(min(1480,available.width()-40),min(900,available.height()-60))
        shell=Q.QVBoxLayout(self);self.module_tabs=Q.QTabWidget();shell.addWidget(self.module_tabs)
        self.grasp_page=Q.QWidget();self.module_tabs.addTab(self.grasp_page,'抓取与任务流程')
        outer=Q.QVBoxLayout(self.grasp_page);outer.setContentsMargins(18,14,18,14);outer.setSpacing(12)
        header=Q.QHBoxLayout();title=Q.QLabel('云蝶  /  物品抓取');title.setObjectName('title');header.addWidget(title);header.addStretch()
        self.mode=Q.QComboBox();self.mode.addItem('仅饮料','beverages');self.mode.addItem('通用物品','all');self.mode.setCurrentIndex(1)
        self.mode.currentIndexChanged.connect(self.change_mode);header.addWidget(self.mode)
        self.demo=Q.QCheckBox('演示模式');header.addWidget(self.demo)
        self.proximity_voice=Q.QCheckBox('避障提示语音');header.addWidget(self.proximity_voice)
        self.proximity_voice.setToolTip('仅控制“前方有人或障碍物”语音；不影响前往、到达播报和实际避障。')
        from .voice_settings import read_proximity_voice
        self.proximity_voice.setChecked(read_proximity_voice(self.root/'data/voice_settings.json'))
        atomic_json(self.run/'voice_settings.json',{'proximity_voice':self.proximity_voice.isChecked()})
        self.proximity_voice.toggled.connect(self.proximity_voice_changed)
        self.demo.toggled.connect(lambda _:self.controls())
        self.demo_settings=Q.QPushButton('演示参数');header.addWidget(self.demo_settings)
        self.demo_settings.clicked.connect(self.edit_demo_settings)
        self.badge=Q.QLabel('306 · 双臂自动选手');self.badge.setObjectName('muted');header.addWidget(self.badge);outer.addLayout(header)
        flow_row=Q.QHBoxLayout();outer.addLayout(flow_row)
        self.flow_button=Q.QPushButton('编排任务流程');flow_row.addWidget(self.flow_button)
        self.flow_label=Q.QLabel('导航 → 抓取 → 导航 → 放置 / 固定动作');flow_row.addWidget(self.flow_label,1)
        self.flow_label.setWordWrap(True)
        self.flow_retry=Q.QPushButton('重试当前步骤');self.flow_retry.setEnabled(False);flow_row.addWidget(self.flow_retry)
        self.flow_retry.clicked.connect(lambda:self.workflow.retry_current())
        content=Q.QHBoxLayout();content.setSpacing(18);outer.addLayout(content,1)
        left=Q.QVBoxLayout();content.addLayout(left,3)
        self.video=ImageView();self.video.clicked.connect(self.image_clicked);left.addWidget(self.video,1)
        line=Q.QHBoxLayout();self.view_text=Q.QLabel('实时 RGB');self.view_text.setObjectName('muted');line.addWidget(self.view_text);line.addStretch()
        live=Q.QPushButton('清除标注');self.live_button=live;live.clicked.connect(self.live);line.addWidget(live);left.addLayout(line)
        self.chassis_button=Q.QPushButton('底盘调整');self.chassis_button.clicked.connect(self.adjust_chassis);line.addWidget(self.chassis_button)
        panel=Q.QFrame();panel.setObjectName('panel');right=Q.QVBoxLayout(panel);right.setContentsMargins(14,14,14,14)
        self.object_scroll=Q.QScrollArea();self.object_scroll.setWidgetResizable(True)
        self.object_scroll.setHorizontalScrollBarPolicy(C.Qt.ScrollBarAlwaysOff)
        self.object_scroll.setMinimumWidth(300);self.object_scroll.setMaximumWidth(360)
        self.object_scroll.setWidget(panel);content.addWidget(self.object_scroll,1)
        # Keep execution controls outside both scrolling columns. Their enable
        # policy and confirmation callbacks remain unchanged.
        self.action_bar=Q.QFrame();actions=Q.QHBoxLayout(self.action_bar);actions.setContentsMargins(0,0,0,0)
        outer.addWidget(self.action_bar)
        heading=Q.QHBoxLayout();right.addLayout(heading)
        label=Q.QLabel('01   选择物品');label.setStyleSheet('font-size:19px;font-weight:600');heading.addWidget(label)
        self.mask_toggle=Q.QCheckBox('分割轮廓');self.mask_toggle.setChecked(True)
        self.mask_toggle.setToolTip('显示模型的物体分割结果；取消勾选查看原图。不改变深度定位或规划使用的掩膜。')
        self.mask_toggle.toggled.connect(self.toggle_masks);heading.addWidget(self.mask_toggle)
        self.semantic=Q.QLineEdit();self.semantic.setPlaceholderText('物品名称：盒子 / 瓶子 / 纸巾（颜色可选）')
        self.semantic.setToolTip('只输入名称识别同类不同颜色物品；加颜色可缩小范围。留空使用当前模式默认范围，品牌仍用商品库匹配。')
        right.addWidget(self.semantic)
        self.semantic_apply=Q.QPushButton('按描述识别');self.semantic_apply.clicked.connect(self.change_prompt)
        self.semantic.returnPressed.connect(self.change_prompt);right.addWidget(self.semantic_apply)
        grip_row=Q.QHBoxLayout();right.addLayout(grip_row);grip_row.addWidget(Q.QLabel('抓取收紧量'))
        self.grip_margin=NoWheelDoubleSpinBox();self.grip_margin.setRange(0,50);self.grip_margin.setDecimals(1);self.grip_margin.setSuffix(' mm')
        self.grip_margin.setToolTip('闭合目标＝估计夹取宽度－收紧量；名义开度换算，非力度控制。固定动作记录不受影响。')
        settings_path=self.root/'data/grip_settings.json'
        try:
            from .grip_width import checked_margin
            value=checked_margin(json.loads(settings_path.read_text('utf-8'))['tightening_mm']) if settings_path.exists() else 5.
        except (ValueError,KeyError,OSError):value=5.
        self.grip_margin.setValue(value);self.grip_margin.valueChanged.connect(self.grip_setting_changed);grip_row.addWidget(self.grip_margin)
        from .target_offsets_ui import TargetOffsets
        self.target_offsets=TargetOffsets(self.root,self);right.addWidget(self.target_offsets)
        self.target_offsets.changed.connect(self.target_offsets_changed)
        self.list=Q.QListWidget();self.list.setMinimumHeight(130);self.list.setMaximumHeight(260)
        self.list.currentItemChanged.connect(self.select);right.addWidget(self.list,1)
        self.prepare=Q.QPushButton('复位并准备');self.prepare.clicked.connect(self.prepare_robot);actions.addWidget(self.prepare)
        self.prepare.setToolTip('点击即执行：双臂和头部复位、双夹爪全开。请先取出物品并清空运动空间。腰腿不动。')
        self.scan=Q.QPushButton('识别物品');self.scan.setObjectName('primary');self.scan.clicked.connect(lambda:self.send('scan'));right.addWidget(self.scan)
        tools=Q.QHBoxLayout();right.addLayout(tools)
        self.enroll=Q.QPushButton('录入 / 追加');self.enroll.clicked.connect(lambda:self.enrollment());tools.addWidget(self.enroll)
        self.manage=Q.QPushButton('商品库');self.manage.clicked.connect(lambda:self.send('catalog'));tools.addWidget(self.manage)
        self.rim=Q.QPushButton('选择袋口夹取点');self.rim.clicked.connect(self.choose_rim);right.addWidget(self.rim)
        right.addSpacing(10);self.flow=Q.QLabel('操作顺序：选择物品 → 规划预览 → 确认抓取');self.flow.setWordWrap(True);right.addWidget(self.flow)
        self.plan=Q.QPushButton('规划预览');self.plan.clicked.connect(lambda:self.send('plan',id=self.selected_id()));actions.addWidget(self.plan)
        from .auto_grasp import AutoExecutionGate
        self.auto_grasp_gate=AutoExecutionGate()
        self.auto_grasp=Q.QCheckBox('自动执行抓取');actions.addWidget(self.auto_grasp)
        self.auto_grasp.setToolTip('默认关闭。勾选后，新发起的抓取规划通过检查即执行，不再逐次确认；适用于单次抓取及任务流程。请先清空运动范围。取消勾选只取消尚未启动的自动抓取，已启动请用停止运动。')
        self.auto_grasp.toggled.connect(lambda _:self.auto_grasp_gate.revoke())
        self.execute=Q.QPushButton('确认抓取');self.execute.setObjectName('primary');self.execute.setEnabled(False)
        self.execute.setToolTip('先选择物品并完成规划预览；路径通过检查后可确认抓取。')
        self.execute.clicked.connect(self.start_execution);actions.addWidget(self.execute,1)
        self.place=Q.QPushButton('确认放置');self.place.clicked.connect(self.start_placement);actions.addWidget(self.place)
        self.place.setToolTip('本窗口完成抓取后，识别当前桌面并自动放置；不需要演示导航。请先停稳并确认仍夹着物品。')
        self.stop=Q.QPushButton('停止运动');self.stop.setObjectName('stop');self.stop.clicked.connect(self.stop_motion);actions.addWidget(self.stop)
        self.status=Q.QLabel('加载模型中…');self.status.setWordWrap(True);self.status.setStyleSheet('font-size:13px')
        self.status.setMaximumHeight(64);outer.addWidget(self.status)
        self.footer=Q.QLabel(self);self.footer.hide()
        self.timer=C.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(80)
        self.preview_stamp=0;self.motion=None
        from .navigation_ui import NavigationPanel
        (self.run/'heartbeat').touch()
        self.navigation=NavigationPanel(self,start=start_worker)
        self.navigation_scroll=Q.QScrollArea();self.navigation_scroll.setWidgetResizable(True)
        self.navigation_scroll.setHorizontalScrollBarPolicy(C.Qt.ScrollBarAlwaysOff)
        self.navigation_scroll.setMinimumWidth(245);self.navigation_scroll.setMaximumWidth(290)
        self.navigation_scroll.setWidget(self.navigation)
        left.removeWidget(self.video);view_row=Q.QHBoxLayout();view_row.addWidget(self.video,1);view_row.addWidget(self.navigation_scroll)
        left.insertLayout(0,view_row,1)
        from .action_studio import ActionStudio
        from .task_flow_ui import Workflow
        self.studio=ActionStudio(self);self.module_tabs.addTab(self.studio,'固定动作编排')
        self.workflow=Workflow(self);self._module_index=0
        from .auto_task_ui import AutomaticTaskPage
        self.automatic=AutomaticTaskPage(self);self.module_tabs.addTab(self.automatic,'自动任务')
        self.module_tabs.currentChanged.connect(self.change_module)
        self.flow_button.clicked.connect(self.edit_flow)
        self.global_stop=Q.QPushButton('停止全部任务 / 动作');self.global_stop.setObjectName('stop')
        self.global_stop.clicked.connect(self.stop_motion);shell.addWidget(self.global_stop)
        if start_worker:
            import yaml
            cfg=yaml.safe_load((self.root/'config/grasp.yaml').read_text('utf-8'))
            proc=C.QProcess(self);self.process=proc
            env=C.QProcessEnvironment.systemEnvironment()
            env.insert('LD_LIBRARY_PATH',str(Path(cfg['python']).parent.parent/'lib')+':'+env.value('LD_LIBRARY_PATH'))
            # IK uses tiny 7-variable matrices; extra BLAS threads add scheduling
            # jitter. Scope this to the read-only planning/perception worker.
            env.insert('OPENBLAS_NUM_THREADS','1');env.insert('MKL_NUM_THREADS','1')
            proc.setProcessEnvironment(env)
            proc.readyReadStandardOutput.connect(self.read);proc.readyReadStandardError.connect(self.log_errors)
            proc.finished.connect(self.finished);proc.errorOccurred.connect(lambda _:self.status.setText('后台启动失败，请查看运行日志'))
            proc.start(cfg['python'],['-m','autolife_shelf_grasp_306.worker','--root',str(self.root),'--run',str(self.run)])
        self.controls()

    def edit_flow(self):
        from .task_flow_ui import FlowDialog
        try:FlowDialog(self).exec_()
        except Exception as exc:Q.QMessageBox.warning(self,'任务库不可用',str(exc))

    def change_module(self,index):
        try:
            if self.automatic.active or self.busy or self.motion or self.demo_active or self.placement_active or self.workflow.active or self.navigation.inflight:
                raise ValueError('当前操作未结束，请先停止或等待完成')
            if index==1:
                self.invalidate_plan();self.studio.ensure();self.studio.panel.setEnabled(True)
            else:self.studio.release()
            self._module_index=index
        except Exception as exc:
            self.module_tabs.blockSignals(True);self.module_tabs.setCurrentIndex(self._module_index);self.module_tabs.blockSignals(False)
            Q.QMessageBox.warning(self,'暂不能切换',str(exc))
        self.controls()

    def navigation_done(self,data):
        if self.workflow.active:self.workflow.navigation_done(data)
        else:super().navigation_done(data)

    def selected_id(self):
        item=self.list.currentItem();return item.data(C.Qt.UserRole) if item else None

    def toggle_masks(self,enabled):
        self.video.show_masks=bool(enabled);self.video.update()

    def proximity_voice_changed(self,checked):
        value={'proximity_voice':bool(checked)}
        try:
            atomic_json(self.root/'data/voice_settings.json',value)
            atomic_json(self.run/'voice_settings.json',value)
        except OSError as exc:
            Q.QMessageBox.warning(self,'语音设置保存失败',str(exc))

    def target_offsets_changed(self):
        self.invalidate_plan();self.path_revision=None;self.video.path=[];self.video.update()
        self.status.setText('夹取中心补偿已保存，请重新规划；持物后的放置沿用抓取时的补偿。')

    def grip_setting_changed(self,value):
        from .core import atomic_json
        self.invalidate_plan();self.path_revision=None;self.video.path=[];self.video.update()
        try:atomic_json(self.root/'data/grip_settings.json',{'tightening_mm':float(value)})
        except OSError as exc:self.status.setText('收紧量本次有效，但保存失败：'+str(exc))

    def edit_demo_settings(self):
        if self.demo_active or self.motion or self.busy:return
        from .demo_ui import edit_settings
        edit_settings(self)

    def adjust_chassis(self):
        if self.busy or self.motion:return
        from .chassis_ui import ChassisDialog
        self.invalidate_plan();self.items=[];self.list.clear();self.live()
        self.chassis_dialog=ChassisDialog(self)
        self.chassis_dialog.move(self.mapToGlobal(C.QPoint(max(0,self.width()-self.chassis_dialog.width()-20),80)))
        self.chassis_dialog.exec_()
        self.chassis_dialog.shutdown();self.chassis_dialog.deleteLater();self.chassis_dialog=None
        self.status.setText('底盘位置已调整，请重新识别物品并规划。');self.controls()

    def controls(self):
        idle=self.ready and not self.busy and self.motion is None and not self.demo_active and not self.placement_active
        if hasattr(self,'workflow'):
            idle=idle and not self.workflow.active and not self.workflow.base.moving() and self.module_tabs.currentIndex()==0 and not self.automatic_active()
            self.flow_button.setEnabled(idle)
            self.flow_retry.setEnabled(idle and self.workflow.retry_available and not self.cancel_pending
                and not self.navigation.inflight and not self.studio.moving())
            self.flow_retry.setToolTip('只重试失败的当前步骤，重新感知和规划；已完成步骤不重复执行')
            if self.workflow.active:
                # Recompute after every workflow transition. The planned event
                # first refreshes controls while phase is still 'plan', then
                # worker_event enters 'confirm'; a one-way disable stays stuck.
                plan=self.pending_plan or {}
                self.execute.setEnabled(
                    self.workflow.phase=='confirm' and self.ready and not self.busy
                    and self.motion is None and not self.cancel_pending
                    and not self.demo_active and not self.placement_active
                    and not self.navigation.inflight and self.grasp_page_allowed()
                    and bool(plan.get('plan_id')) and plan.get('plan_id')==self.workflow.plan_id
                    and plan.get('target',{}).get('instance_id')==self.selected_id()
                    and not self.blockers and not plan.get('blockers'))
        if hasattr(self,'navigation'):idle=idle and not self.navigation.inflight
        self.demo.setEnabled(idle);self.demo_settings.setEnabled(idle)
        self.live_button.setEnabled(idle)
        if hasattr(self,'navigation'):self.navigation.controls(idle)
        self.chassis_button.setEnabled(idle and self.chassis_dialog is None)
        self.scan.setEnabled(idle);self.list.setEnabled(idle)
        self.mode.setEnabled(idle);self.manage.setEnabled(idle)
        self.semantic.setEnabled(idle);self.semantic_apply.setEnabled(idle)
        self.grip_margin.setEnabled(idle)
        self.target_offsets.setEnabled(idle)
        if self.catalog_dialog:
            for b in self.catalog_dialog.buttons:b.setEnabled(idle and bool(self.catalog_dialog.current()))
        self.prepare.setEnabled(idle)
        self.place.setVisible(not self.demo.isChecked());self.place.setEnabled(idle and not self.demo.isChecked())
        chosen=self.selected_id() is not None
        self.enroll.setEnabled(idle and chosen);self.plan.setEnabled(idle and chosen)
        item=next((v for v in self.items if v['id']==self.selected_id()),{})
        self.rim.setVisible(item.get('geometry')=='bag_edge')
        self.rim.setEnabled(idle and self.frozen and item.get('geometry')=='bag_edge')
        if self.demo.isChecked():
            self.execute.setText('确认演示抓取');self.execute.setEnabled(idle)

    def send(self,action,**values):
        if not self.process:return
        if action=='plan':
            values['grip_tightening_mm']=self.grip_margin.value()
            try:values['target_offsets_by_arm_torso_mm']=self.target_offsets.profiles()
            except ValueError as exc:self.status.setText(str(exc));return
        self.invalidate_plan()
        if action=='plan':
            chosen=next((v for v in self.items if v['id']==values.get('id')),None)
            self.expected_target=({'instance_id':chosen['id'],'product_id':chosen.get('product_id')}
                                  if chosen else None)
        self.last_worker_request=(action,dict(values))
        self.request+=1;self.busy=True;self.cancel_pending=False;self.execute.setEnabled(False);self.video.path=[]
        self.overlay_revision=self.request;self.path_revision=None
        if action=='plan':
            self.auto_grasp_gate.arm(self.request,self.grasp_context(),
                self.automatic_grasp_enabled() and not self.demo.isChecked() and not self.demo_active)
        self.process.write((json.dumps({'action':action,'request_id':self.request,**values},ensure_ascii=False)+'\n').encode())
        self.status.setText({'scan':'正在识别物品…','scan_task':'正在按任务物品识别并分割…','set_prompt':'按文字定位并分割物品…','plan':'正在定位和规划…','enroll':'保存商品样本…','prepare':'检查原VR复位路径…','remove_product':'正在删除商品…'}.get(action,'处理中…'))
        self.controls()

    def prepare_robot(self):
        if self.busy or self.motion:return
        if Q.QMessageBox.question(self,'复位到原VR准备位姿',
            '保持当前位置使能，再直接到原VR准备位姿并张开双夹爪；腰腿保持不动。\n'
            '请清空夹爪及机械臂运动范围，并准备好实体急停。',
            defaultButton=Q.QMessageBox.No)==Q.QMessageBox.Yes:self.send('prepare')

    def change_mode(self):
        if not self.ready:return
        self.live();self.items=[];self.list.clear();self.send('set_mode',mode=self.mode.currentData())

    def change_prompt(self):
        if not self.semantic_apply.isEnabled():return
        self.live();self.items=[];self.list.clear();self.send('set_prompt',text=self.semantic.text())

    def read(self):
        self.buffer+=bytes(self.process.readAllStandardOutput())
        while b'\n' in self.buffer:
            line,self.buffer=self.buffer.split(b'\n',1)
            if not line.lstrip().startswith(b'{'):
                with (self.run/'worker.log').open('ab') as stream:stream.write(line+b'\n')
                continue
            try:self.receive_event(json.loads(line))
            except (ValueError,KeyError) as exc:self.status.setText('界面数据异常：'+str(exc))

    def log_errors(self):
        with (self.run/'worker.log').open('ab') as stream:stream.write(bytes(self.process.readAllStandardError()))

    def receive_event(self,data):
        kind=data['event']
        if 'request_id' in data and data['request_id']!=self.request:return
        if self.cancel_pending:
            if 'request_id' in data:
                self.busy=False;self.cancel_pending=False;self.status.setText('已取消本次操作，未开始新运动。');self.controls()
            return
        if self.automatic.defer_worker_event(data):return
        if kind=='status':self.status.setText(data['text']);return
        self.busy=False
        if 'mode' in data:
            self.mode.blockSignals(True);self.mode.setCurrentIndex(max(0,self.mode.findData(data['mode'])));self.mode.blockSignals(False)
        if 'semantic_prompt' in data:self.semantic.setText(data['semantic_prompt'])
        if 'semantic_prompt_resolved' in data:
            self.semantic.setToolTip('实际识别描述：'+data['semantic_prompt_resolved']+'\n只输入名称即可；颜色是可选筛选条件，品牌由商品库匹配。')
        if kind=='ready':
            self.ready=True;self.blockers=data['blockers'];self.status.setText('识别已就绪，可直接识别，或输入物品描述。')
            self.execute.setToolTip('；'.join(self.blockers))
        elif kind=='reset_planned':
            self.live();self.items=[];self.list.clear()
            self.status.setText('保持式使能，复位到原VR位姿并张开双爪…')
            if data['blockers']:
                self.status.setText('；'.join(data['blockers']))
                if self.demo_active:self.demo_failure('；'.join(data['blockers']))
                if self.automatic.active:self.stop_motion(reason='；'.join(data['blockers']),source='automatic_task')
            else:self.launch_motion('last_reset_plan.json',preparing=True)
        elif kind=='observe_planned':
            if not (self.demo_active or self.placement_active) or self.demo_phase!='planning':return
            self.demo_context['observation_plan_id']=data['plan_id']
            self.live();self.status.setText('保持夹紧，轻抬手臂后重新观察桌面…')
            self.demo_phase='observe';self.launch_motion(data['filename'],placement=data)
        elif kind=='place_planned':
            if not (self.demo_active or self.placement_active) or self.demo_phase!='planning':return
            self.demo_context['placement_plan_id']=data['plan_id']
            self.video.pix=G.QPixmap(str(self.run/'scene.jpg'));self.video.shape=data['shape']
            self.video.path=data['pixels'];self.video.boxes=[data['table_segmentation']] if data.get('table_segmentation') else [];self.frozen=True
            self.overlay_revision=data.get('request_id',self.request);self.path_revision=self.overlay_revision
            self.view_text.setText('自动放置点 · 桌面与轨迹检查已通过')
            self.demo_phase='place';self.launch_motion(data['filename'],placement=data)
        elif kind=='auto_prepared':
            if not self.automatic.prepared({'status':'prepared'}):
                if self.workflow.active and self.workflow.phase=='prepare':self.workflow.motion_done({'status':'prepared'})
        elif kind=='prepared':
            self.live();self.send('scan')
        elif kind in ('scene','enrolled','catalog'):
            self.overlay_revision=data.get('request_id',self.request);self.path_revision=None
            previous_id=self.selected_id();self.items=data['items'];self.list.clear()
            for v in self.items:
                dims=v.get('dimensions_mm')
                text=f'#{v["id"]}  {v["name"]}\n'+(f'宽 {dims[0]:g} · 高 {dims[1]:g} mm' if dims else '可直接规划 · 自动估计尺寸')
                item=Q.QListWidgetItem(text);item.setData(C.Qt.UserRole,v['id']);self.list.addItem(item)
            if kind=='scene':
                self.video.shape=data['shape'];self.video.pix=G.QPixmap(str(self.run/'scene.jpg'))
                self.frozen=True;self.video.boxes=self.items;self.view_text.setText('实时 RGB · 分割标注跟踪 · 规划前重新识别')
                self.status.setText(f'找到 {len(self.items)} 个候选 · 识别 {data["elapsed_ms"]:.0f} ms')
                if data.get('semantic_prompt') and data.get('semantic_prompt_resolved'):
                    self.status.setText(self.status.text()+'\n'+data['semantic_prompt_resolved'].rstrip('.'))
            else:self.video.boxes=self.items;self.status.setText(data.get('text','商品库已同步，修改后请重新规划'))
            if self.items:
                keep=data.get('select_id',previous_id if kind!='scene' else None)
                row=next((i for i,v in enumerate(self.items) if v['id']==keep),0)
                self.list.setCurrentRow(row)
            if kind=='catalog':
                if self.catalog_dialog is None:self.catalog_dialog=CatalogDialog(self)
                self.catalog_dialog.refresh(data['products'],data.get('product_id'));self.catalog_dialog.show()
            if kind=='scene' and self.demo_active and self.demo_phase=='scan':self.demo_scene_ready()
        elif kind=='planned':
            target=data.get('target',{});pid=data.get('plan_id','')
            if (not self.expected_target or target.get('instance_id')!=self.selected_id() or
                any(target.get(k)!=v for k,v in self.expected_target.items()) or
                len(pid)!=32 or any(c not in '0123456789abcdef' for c in pid) or
                data.get('filename')!='plan_'+pid+'.json'):
                self.invalidate_plan();self.status.setText('规划目标与所选饮料不一致，请重新规划。')
                if self.demo_active:self.demo_failure('规划商品与选择不一致')
                if self.workflow.active:self.workflow.fail('规划目标校验失败，请重新识别')
                self.controls();return
            self.pending_plan=data
            self.overlay_revision=data.get('request_id',self.request);self.path_revision=self.overlay_revision
            self.frozen=True;self.video.pix=G.QPixmap(str(self.run/'scene.jpg'));self.video.shape=data['shape'];self.video.boxes=[];self.video.path=data['pixels']
            mode='柔性夹取' if data.get('orientation_mode')=='flexible' else '水平夹取'
            retreat=data.get('withdraw_yaw_deg',0.)
            retreat_text=f' · 收回转向 {retreat:+.0f}°' if retreat else ''
            if data.get('withdraw_orientation_mode')=='adaptive':retreat_text=' · 自然收回'
            if data.get('withdraw_orientation_mode')=='fixed_carry':retreat_text=' · 固定持物姿态'
            self.status.setText(f'{target["name"]} · {mode} · 接近 {data.get("approach_yaw_deg",0):.0f}°{retreat_text} · {data["seconds"]:.1f}s')
            hand='右手' if data.get('arm')=='right' else '左手'
            provisional=bool(data.get('tool_geometry',{}).get('provisional'))
            test_calibration=bool(data.get('tool_geometry',{}).get('calibration_profile',{}).get('test_only'))
            self.badge.setText('306 · '+hand+(' · 测试标定（待验证）' if test_calibration else (' · TCP临时估计' if provisional else ' · 标定TCP')))
            self.status.setText(hand+'抓取 · '+self.status.text())
            if data.get('grasp_contact_mode')=='front_local':
                self.status.setText(self.status.text()+' · 前端局部夹取（非物体中心）')
            if data.get('partial_view'):
                self.status.setText(self.status.text()+' · 按可见部分夹取（底部未确认，不能自动放置）')
            if data.get('estimated_dimensions_mm'):
                w,h,d=data['estimated_dimensions_mm']
                self.status.setText(self.status.text()+f'\n估计宽×高×厚：{w:.0f}×{h:.0f}×{d:.0f} mm（未保存商品库）')
            self.view_text.setText('路径预览 · 1预抓取  2抓取  3连续起瓶并收回' if data.get('carry_protocol_version')==2
                                  else '路径预览 · 1预抓取  2抓取  3抬起  4退离'+('  5固定持物' if data.get('carry_protocol_version') else ''))
            self.blockers=data['blockers'];self.execute.setToolTip('；'.join(self.blockers));self.execute.setEnabled(not self.blockers)
            self.execute.setText('确认抓取 · '+target['name'])
            if self.blockers:self.status.setText(target['name']+'路径已预览；'+'；'.join(self.blockers))
            self.footer.setText('仅预览 · '+('；'.join(self.blockers[1:3]) if self.blockers else '请确认操作空间清空'))
            if self.demo_active and self.demo_phase=='grasp_plan':self.demo_plan_ready(data)
        elif kind in ('error','fatal'):
            self.path_revision=None
            self.demo_active=False;self.placement_active=False;self.demo_phase=None
            self.invalidate_plan();self.status.setText(data['text'])
            if self.catalog_dialog and self.catalog_dialog.isVisible():self.catalog_dialog.info.setText(data['text'])
            if kind=='fatal':self.ready=False
            if self.automatic.active and self.automatic.phase=='preparing':self.stop_motion(reason=data['text'],source='automatic_task')
        self.video.update();self.controls()
        self.workflow.worker_event(data)
        if kind=='planned' and self.pending_plan is data and self.auto_grasp_gate.intent:
            intent=self.auto_grasp_gate.intent;plan_id=data['plan_id']
            C.QTimer.singleShot(0,lambda:self.execute_automatic_grasp(intent,plan_id))

    def automatic_active(self):
        return bool(getattr(self,'automatic',None) and self.automatic.active)

    def automatic_grasp_enabled(self):
        return self.auto_grasp.isChecked() or (self.automatic_active() and self.workflow.active)

    def grasp_page_allowed(self):
        return self.module_tabs.currentIndex()==0 or (self.module_tabs.currentIndex()==2 and self.automatic_active())

    def grasp_context(self):
        return self.workflow.cursor.token if self.workflow.active else None

    def execute_automatic_grasp(self,intent,plan_id):
        if self.automatic.active and self.automatic.paused:
            self.automatic.deferred_event={'event':'planned'};return
        data=self.pending_plan or {}
        if (not self.ready or self.busy or self.motion is not None or self.cancel_pending
            or self.demo.isChecked() or self.demo_active or self.placement_active
            or self.navigation.inflight or self.studio.moving() or not self.grasp_page_allowed()
            or data.get('plan_id')!=plan_id or self.blockers or data.get('blockers')
            or data.get('target',{}).get('instance_id')!=self.selected_id()):return
        if self.workflow.active and (self.workflow.phase!='confirm' or self.workflow.plan_id!=plan_id):return
        if not self.auto_grasp_gate.consume(intent,self.request,self.grasp_context(),self.automatic_grasp_enabled()):return
        self.start_execution(automatic=True)
        if self.workflow.active and self.workflow.phase=='confirm' and not self.pending_plan:
            self.workflow.fail(self.status.text())

    def invalidate_plan(self):
        if hasattr(self,'auto_grasp_gate'):self.auto_grasp_gate.revoke()
        self.pending_plan=None;self.expected_target=None;self.execute.setEnabled(False);self.execute.setText('确认抓取')

    def select(self,*_):
        self.path_revision=None
        self.picking_rim=False;self.video.selected=self.selected_id();self.invalidate_plan();self.video.path=[]
        item=next((v for v in self.items if v['id']==self.selected_id()),{})
        if item.get('grasp_fraction'):
            x0,y0,x1,y1=item['box'];u,v=item['grasp_fraction'];self.video.path=[[x0+u*(x1-x0),y0+v*(y1-y0)]]
        self.video.update();self.controls()

    def choose_rim(self):
        self.invalidate_plan();self.picking_rim=True
        self.status.setText('点击袋口的实体边缘（避开细提绳）')

    def live(self):
        self.invalidate_plan();self.picking_rim=False
        self.overlay_revision=None;self.path_revision=None;self.video.notice=''
        self.frozen=False;self.video.boxes=[];self.video.path=[];self.execute.setEnabled(False);self.view_text.setText('实时 RGB');self.preview_stamp=0

    def image_clicked(self,x,y):
        if self.busy or self.motion:return
        if self.picking_rim:
            shown=next((v for v in self.video.boxes if v['id']==self.selected_id()),None)
            if shown is None:self.status.setText('目标标注已丢失，请重新识别后选取袋口');return
            matrix=shown.get('display_to_reference')
            if matrix is not None:
                px=x*self.video.shape[1];py=y*self.video.shape[0]
                x=(matrix[0][0]*px+matrix[0][1]*py+matrix[0][2])/self.video.shape[1]
                y=(matrix[1][0]*px+matrix[1][1]*py+matrix[1][2])/self.video.shape[0]
            self.picking_rim=False;self.send('grasp_point',id=self.selected_id(),x=x,y=y);return
        for v in self.video.boxes:
            a,b,c,d=v['box'];px=x*self.video.shape[1];py=y*self.video.shape[0]
            if self.frozen and a<=px<=c and b<=py<=d:
                i=next((i for i,item in enumerate(self.items) if item['id']==v['id']),None)
                if i is not None:self.list.setCurrentRow(i)
                break

    def enrollment(self,product=None,edit_only=False):
        if not edit_only and (self.selected_id() is None or not self.frozen):return
        path=self.root/'data/products.json'
        try:catalog=json.loads(path.read_text('utf-8'))['products'] if path.exists() else []
        except (ValueError,OSError) as exc:self.status.setText('无法读取商品库：'+str(exc));return
        item=next((v for v in self.items if v['id']==self.selected_id()),{})
        product=product or next((p for p in catalog if p['id']==item.get('product_id')),None)
        dialog=ProductEditor(self,product,catalog,'beverage' if self.mode.currentData()=='beverages' else 'general',edit_only)
        if dialog.exec_()==Q.QDialog.Accepted:
            if edit_only:self.send('update_product',product_id=product['id'],values=dialog.values())
            else:self.send('enroll',id=self.selected_id(),**dialog.values())

    def start_placement(self):
        if self.busy or self.motion or self.demo_active or self.placement_active or self.demo.isChecked():return
        if self.navigation.inflight:return
        try:
            from .placement_request import held_grasp,new_standalone_request
            from .demo_config import load,require_placeable
            grasp,_=held_grasp(self.run);require_placeable(grasp['product'])
            if self.blockers:raise ValueError('；'.join(self.blockers))
            if Q.QMessageBox.question(self,'确认放置',
                '请确认'+('右' if grasp.get('config',{}).get('arm')=='right' else '左')+'夹爪仍稳固夹着“'+grasp['product']['name']+'”，底盘已停稳。\n'
                '将识别当前桌面，必要时保持夹紧轻抬观察，再自动放置并松爪收回。\n'
                '清空运动范围并准备实体急停，开始？',defaultButton=Q.QMessageBox.No)!=Q.QMessageBox.Yes:return
            self.demo_context=new_standalone_request(self.run,load(self.root))
        except (ValueError,OSError,KeyError) as exc:
            self.status.setText('未开始放置：'+str(exc));return
        self.placement_active=True;self.demo_phase='planning';self.send('standalone_place')
        self.status.setText('正在识别当前桌面并规划放置…')

    def start_execution(self,checked=False,*,automatic=False):
        if self.motion:return
        if not automatic:self.auto_grasp_gate.revoke()
        if self.workflow.active:
            self.controls()
            if self.workflow.phase!='confirm' or not self.execute.isEnabled():return
        if self.demo.isChecked():self.start_navigation_demo();return
        if self.blockers:Q.QMessageBox.information(self,'尚未开放运动','\n'.join(self.blockers));return
        data=self.pending_plan
        if not data or data['target']['instance_id']!=self.selected_id():
            self.invalidate_plan();self.status.setText('请为当前选中的饮料重新规划。');return
        try:
            receipt=json.loads((self.run/'prepare_result.json').read_text('utf-8'))
            if receipt.get('status')!='prepared' or not data.get('prepare_token') or receipt.get('token')!=data['prepare_token']:
                raise ValueError('本轮复位尚未完成，请先复位并准备')
        except (ValueError,OSError) as exc:
            self.invalidate_plan();self.status.setText('不能开始抓取：'+str(exc));return
        estimated=data.get('estimated_dimensions_mm')
        notice=(f'本次自动估计宽×高×厚：{estimated[0]:.0f}×{estimated[1]:.0f}×{estimated[2]:.0f} mm，请确认尺寸与实物相符。\n' if estimated else '')
        if data.get('grasp_contact_mode')=='front_local':notice+='本次夹取较窄前端，非物体中心；偏心夹取的夹持稳定性尚未实测。\n'
        from .target_offsets import description
        notice+='本次'+('左手' if data.get('arm','left')=='left' else '右手')+'位置补偿：'+description(data.get('config',{}).get('target_offset_torso_mm',[0,0,0]))+'。\n'
        if not automatic and Q.QMessageBox.question(self,'确认抓取：'+data['target']['name'],notice+'使用当前手眼标定及独立TCP修正；实物精度尚未验收。\n抓取段不检查桌面、货架及邻物，请确认预览路径空旷。\n闭爪稳定后自动抬起并收回；尚无夹稳检测。\n人员离开运动范围并准备实体急停，开始？',defaultButton=Q.QMessageBox.No)!=Q.QMessageBox.Yes:return
        if self.motion or self.cancel_pending or self.busy or self.pending_plan is not data or data['target']['instance_id']!=self.selected_id():return
        self.auto_grasp_gate.revoke()
        if self.workflow.active:self.workflow.phase='grasp'
        self.launch_motion(data['filename'])

    def launch_motion(self,filename,preparing=False,placement=None):
        if self.cancel_pending:return
        extra=[]
        if placement is not None:
            if not (self.demo_active or self.placement_active) or self.demo_phase not in ('place','observe'):return
            path=self.run/filename;extra=['--expected-plan-id',placement['plan_id'],'--expected-product-id',placement['product_id']]
        elif preparing:path=self.root/'data'/filename
        else:
            data=self.pending_plan
            if not data or filename!=data['filename'] or data['target']['instance_id']!=self.selected_id():
                self.status.setText('当前规划已失效，请重新规划');return
            path=self.run/filename
            extra=['--expected-plan-id',data['plan_id'],'--expected-product-id',data['target']['product_id']]
        self.preparing=preparing;self.motion=C.QProcess(self);self.motion.finished.connect(self.motion_finished)
        self.motion.errorOccurred.connect(self.motion_error)
        self.motion.setProcessChannelMode(C.QProcess.MergedChannels)
        self.motion.readyReadStandardOutput.connect(self.motion_output)
        (self.run/'stop').unlink(missing_ok=True)
        (self.run/'execution_result.json').unlink(missing_ok=True)
        if preparing:(self.run/'prepare_result.json').unlink(missing_ok=True)
        self.motion.start('/usr/bin/python3',['-m','autolife_shelf_grasp_306.execution','--plan',str(path),'--run',str(self.run),*extra])
        self.execute.setEnabled(False);self.controls()

    def motion_error(self,error):
        if error==C.QProcess.FailedToStart:
            self.demo_active=False;self.placement_active=False;self.demo_phase=None
            self.motion=None;self.status.setText('运动进程启动失败，未执行运动。');self.controls()
            if self.workflow.active:self.workflow.fail('运动进程启动失败')
            elif self.automatic.active:self.stop_motion(reason='运动进程启动失败',source='automatic_task')

    def motion_output(self):
        if self.motion:
            raw=bytes(self.motion.readAllStandardOutput())
            with (self.run/'execution.log').open('ab') as stream:stream.write(raw)
            text=raw.decode(errors='replace').strip()
            if text:self.status.setText(text[-250:])

    def motion_finished(self,*args):
        phase=self.demo_phase
        self.demo_phase=None
        self.invalidate_plan()
        self.motion=None;self.execute.setText('确认抓取');self.execute.setEnabled(False)
        result_path=self.run/'execution_result.json'
        try:result=json.loads(result_path.read_text('utf-8')) if result_path.exists() else {}
        except (ValueError,OSError):result={}
        if self.automatic.prepared(result):return
        if self.workflow.motion_done(result):return
        if self.placement_active:
            if phase=='observe' and result.get('status')=='observed' and result.get('plan_id')==self.demo_context.get('observation_plan_id'):
                from .core import atomic_json
                atomic_json(self.run/'demo_observe_result.json',result)
                self.demo_phase='planning';self.send('standalone_place');return
            self.placement_active=False;self.demo_phase=None
            if phase=='place' and result.get('status')=='placed' and result.get('plan_id')==self.demo_context.get('placement_plan_id'):
                self.finish_placement(result);return
            else:
                self.status.setText('放置已停止：'+result.get('reason','前一阶段未成功，未继续动作。'))
                if self.workflow.active:self.workflow.fail(self.status.text())
            self.controls();return
        if self.demo_active:
            if phase=='prepare' and result.get('status')=='prepared':
                self.navigate_demo('to_shelf');return
            if phase=='grasp' and result.get('status')=='completed' and result.get('plan_id')==self.demo_context['grasp_plan_id']:
                from .core import atomic_json
                atomic_json(self.run/'demo_grasp_result.json',result)
                self.navigate_demo('to_front');return
            if phase=='observe' and result.get('status')=='observed' and result.get('plan_id')==self.demo_context.get('observation_plan_id'):
                from .core import atomic_json
                atomic_json(self.run/'demo_observe_result.json',result)
                self.demo_phase='planning';self.send('demo_place');return
            if phase=='place' and result.get('status')=='placed':
                self.demo_active=False;self.demo_phase=None
                self.finish_placement(result);return
            self.demo_active=False;self.demo_phase=None
            self.status.setText('演示已停止：'+result.get('reason','前一阶段未成功，未继续后续动作。'));self.controls();return
        self.status.setText('已停止：'+result['reason'] if result.get('reason') else '复位完成。' if result.get('status')=='prepared' else '抓取流程完成。' if result.get('status')=='completed' else '运动进程意外结束，请检查日志。');self.controls()
        if self.preparing and (self.run/'prepare_result.json').exists():
            receipt=json.loads((self.run/'prepare_result.json').read_text('utf-8'))
            if receipt.get('status')=='prepared':self.send('prepared')

    def finish_placement(self,result):
        self.live()
        if self.workflow.active:
            self.workflow.advance();return
        if self.cancel_pending or (self.run/'stop').exists():
            self.status.setText('放置完成，用户已停止，未自动复位。');self.controls();return
        if result.get('auto_prepare_allowed') is False:
            self.status.setText('放置完成，准备确认未通过：'+result.get('preparation_note','请检查保持状态'));self.controls();return
        # Reuse a physically confirmed prepared pose instead of repeating the
        # same arm movement. Otherwise use the existing checked reset workflow.
        self.send('prepared' if result.get('prepared_after_place') else 'prepare')
        self.status.setText('放置完成：已准备好下一轮，正在刷新识别。' if result.get('prepared_after_place')
                            else '放置完成：自动复位并准备下一轮…')

    def stop_motion(self,checked=False,*,reason=None,source='operator'):
        reason=reason or '用户请求停止'
        if hasattr(self,'automatic'):self.automatic.abort(reason)
        from .core import atomic_json
        receipt=dict(time=time.time(),source=source,reason=reason,
            phase=getattr(getattr(self,'workflow',None),'phase',''))
        if hasattr(self,'workflow'):self.workflow.base.stop()
        if hasattr(self,'workflow') and self.workflow.active:
            self.workflow.stopped(reason);self.flow_label.setText('任务流程已停止 · 可重试当前步骤')
        if hasattr(self,'studio'):self.studio.stop()
        if hasattr(self,'navigation'):self.navigation.command('stop')
        self.demo_active=False
        self.placement_active=False
        if self.motion is None:self.demo_phase=None
        if self.chassis_dialog:self.chassis_dialog.stop()
        self.invalidate_plan()
        if self.busy and self.request>0:self.cancel_pending=True
        (self.run/'stop').touch();self.execute.setEnabled(False)
        try:atomic_json(self.run/'stop_reason.json',receipt)
        except OSError:pass  # Diagnostics must never prevent an actual stop.
        self.status.setText(('已停止：'+reason+'。' if source!='operator' else '已请求停止；')+'危险时请按实体急停。')
        self.controls()

    def tick(self):
        (self.run/'heartbeat').touch()
        packet=self.run/'preview.npz'
        if packet.exists():
            try:
                stamp=packet.stat().st_mtime_ns
                if time.time()-packet.stat().st_mtime>1.:
                    self.video.pix=G.QPixmap();self.video.boxes=[];self.video.path=[];self.video.update();return
                if stamp!=self.preview_stamp:
                    jpeg,meta=decode_packet(packet.read_bytes())
                    pix=G.QPixmap();pix.loadFromData(jpeg,'JPG')
                    if pix.isNull():return
                    self.video.pix=pix;self.video.shape=meta['shape'];self.preview_stamp=stamp
                    matched=self.overlay_revision is not None and meta.get('revision')==self.overlay_revision
                    self.video.boxes=meta.get('items',[]) if matched else []
                    path_matches=self.path_revision is not None and meta.get('path_revision')==self.path_revision
                    self.video.path=meta.get('path',[]) if path_matches else []
                    self.video.notice=('部分标注跟踪丢失，请重新识别' if matched and meta.get('lost') else
                        '轨迹投影等待新鲜姿态' if path_matches and meta.get('path_status')=='unavailable' else '')
                    self.video.update()
            except (OSError,ValueError,KeyError):pass
            return
        path=self.run/'preview.jpg'
        if path.exists():
            stamp=path.stat().st_mtime
            if time.time()-stamp>1.:
                self.video.pix=G.QPixmap();self.video.update();return
            if stamp!=self.preview_stamp:
                self.video.pix=G.QPixmap(str(path));self.preview_stamp=stamp;self.video.update()

    def finished(self,*_):
        if self.demo_active or self.placement_active or self.workflow.active or self.automatic_active():
            self.stop_motion(reason='识别后台已退出',source='worker_exit')
        self.ready=False;self.busy=False;self.controls()

    def closeEvent(self,event):
        if self.demo_active or self.placement_active or self.workflow.active or self.automatic_active():
            self.stop_motion(reason='关闭抓取窗口',source='window_close')
        if self.workflow.base.moving():
            self.workflow.base.shutdown()
            if self.workflow.base.moving():self.status.setText('等待底盘停止后再关闭');event.ignore();return
        if self.studio.moving():
            self.studio.stop();self.status.setText('已请求停止动作，请确认停稳后再关闭');event.ignore();return
        if self.chassis_dialog:self.chassis_dialog.shutdown()
        if self.motion:
            self.stop_motion();self.status.setText('停止运动后再关闭界面');event.ignore();return
        if not self.navigation.shutdown():
            self.status.setText('等待本程序的导航取消确认，请停稳后再次关闭');event.ignore();return
        if self.process and self.process.state()!=C.QProcess.NotRunning:
            self.process.write(b'{"action":"quit"}\n');self.process.closeWriteChannel()
            if not self.process.waitForFinished(2000):self.process.terminate();self.process.waitForFinished(1000)
            if self.process.state()!=C.QProcess.NotRunning:self.process.kill();self.process.waitForFinished(1000)
        self.automatic.shutdown();self.studio.shutdown();event.accept()


def main():
    os.environ.setdefault('ROS_DOMAIN_ID','0')
    os.environ.setdefault('RMW_IMPLEMENTATION','rmw_cyclonedds_cpp')
    os.environ.setdefault('CYCLONEDDS_URI','<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>')
    root=Path(os.environ.get('SHELF_GRASP_ROOT','/home/ubuntu/ros2_ws/src/autolife_shelf_grasp_306'))
    app=Q.QApplication(sys.argv[:1]);app.setStyleSheet(STYLE)
    root.joinpath('data').mkdir(parents=True,exist_ok=True)
    # Single-instance guard only protects this app; no competing process killed.
    lock=C.QLockFile(str(root/'data/gui.lock'));lock.setStaleLockTime(0)
    if not lock.tryLock(100):Q.QMessageBox.information(None,'饮料抓取','饮料抓取窗口已经打开。');return
    run=Path(tempfile.mkdtemp(prefix='shelf306_'))
    window=Window(root,run);window.show();code=app.exec_();lock.unlock();sys.exit(code)


if __name__=='__main__':main()
