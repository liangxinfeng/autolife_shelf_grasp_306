"""Native Qt operator UI. CPU/GPU work never runs in the GUI event loop."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from PyQt5 import QtCore as C, QtGui as G, QtWidgets as Q
from .demo_navigation import DemoNavigation

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
        self.setMinimumSize(540,360);self.rect_image=C.QRectF()

    def paintEvent(self,event):
        painter=G.QPainter(self);painter.fillRect(self.rect(),G.QColor('#090e14'))
        if self.pix.isNull():
            painter.setPen(G.QColor('#8ea3b2'));painter.drawText(self.rect(),C.Qt.AlignCenter,'等待头部 RGB 图像');return
        fitted=self.pix.size().scaled(self.size(),C.Qt.KeepAspectRatio)
        r=C.QRectF((self.width()-fitted.width())/2,(self.height()-fitted.height())/2,fitted.width(),fitted.height());self.rect_image=r
        painter.drawPixmap(r,self.pix,C.QRectF(self.pix.rect()))
        sx,sy=r.width()/self.shape[1],r.height()/self.shape[0]
        def point(x,y):return C.QPointF(r.x()+x*sx,r.y()+y*sy)
        for box in self.boxes:
            x0,y0,x1,y1=box['box'];painter.setPen(G.QPen(G.QColor('#ffc86c' if box['id']==self.selected else '#45d9bd'),3))
            painter.drawRect(C.QRectF(point(x0,y0),point(x1,y1)));painter.drawText(point(x0,y0-8),f'#{box["id"]} '+box['name'])
        if self.path:
            painter.setPen(G.QPen(G.QColor('#f7c475'),3));points=[point(*p) for p in self.path]
            for a,b in zip(points,points[1:]):painter.drawLine(a,b)
            for i,p in enumerate(points):
                painter.setBrush(G.QColor('#f7c475'));painter.drawEllipse(p,5,5);painter.drawText(p+C.QPointF(9,-9),str(i+1))

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
        self.pending_plan=None;self.expected_target=None;self.picking_rim=False
        self.catalog_dialog=None;self.chassis_dialog=None
        self.demo_active=False;self.demo_phase=None;self.demo_context=None
        self.setWindowTitle('云蝶 · 物品抓取 · 放置观察版');self.resize(1260,800)
        outer=Q.QVBoxLayout(self);outer.setContentsMargins(26,22,26,20);outer.setSpacing(18)
        header=Q.QHBoxLayout();title=Q.QLabel('云蝶  /  物品抓取');title.setObjectName('title');header.addWidget(title);header.addStretch()
        self.mode=Q.QComboBox();self.mode.addItem('仅饮料','beverages');self.mode.addItem('通用物品','all');self.mode.setCurrentIndex(1)
        self.mode.currentIndexChanged.connect(self.change_mode);header.addWidget(self.mode)
        self.demo=Q.QCheckBox('演示模式');header.addWidget(self.demo)
        self.demo.toggled.connect(lambda _:self.controls())
        self.demo_settings=Q.QPushButton('演示参数');header.addWidget(self.demo_settings)
        self.demo_settings.clicked.connect(self.edit_demo_settings)
        self.badge=Q.QLabel('300 · 左臂 · 桌面实验');self.badge.setObjectName('muted');header.addWidget(self.badge);outer.addLayout(header)
        content=Q.QHBoxLayout();content.setSpacing(18);outer.addLayout(content,1)
        left=Q.QVBoxLayout();content.addLayout(left,3)
        self.video=ImageView();self.video.clicked.connect(self.image_clicked);left.addWidget(self.video,1)
        line=Q.QHBoxLayout();self.view_text=Q.QLabel('实时 RGB');self.view_text.setObjectName('muted');line.addWidget(self.view_text);line.addStretch()
        live=Q.QPushButton('返回实时画面');self.live_button=live;live.clicked.connect(self.live);line.addWidget(live);left.addLayout(line)
        self.chassis_button=Q.QPushButton('底盘调整');self.chassis_button.clicked.connect(self.adjust_chassis);line.addWidget(self.chassis_button)
        panel=Q.QFrame();panel.setObjectName('panel');panel.setMaximumWidth(350);panel.setMinimumWidth(290);right=Q.QVBoxLayout(panel);right.setContentsMargins(20,20,20,20);content.addWidget(panel,1)
        label=Q.QLabel('01   选择物品');label.setStyleSheet('font-size:19px;font-weight:600');right.addWidget(label)
        self.list=Q.QListWidget();self.list.currentItemChanged.connect(self.select);right.addWidget(self.list,1)
        self.prepare=Q.QPushButton('复位并准备');self.prepare.clicked.connect(self.prepare_robot);right.addWidget(self.prepare)
        self.prepare.setToolTip('点击即执行：双臂和头部复位、双夹爪全开。请先取出物品并清空运动空间。腰腿不动。')
        self.scan=Q.QPushButton('识别物品');self.scan.setObjectName('primary');self.scan.clicked.connect(lambda:self.send('scan'));right.addWidget(self.scan)
        tools=Q.QHBoxLayout();right.addLayout(tools)
        self.enroll=Q.QPushButton('录入 / 追加');self.enroll.clicked.connect(lambda:self.enrollment());tools.addWidget(self.enroll)
        self.manage=Q.QPushButton('商品库');self.manage.clicked.connect(lambda:self.send('catalog'));tools.addWidget(self.manage)
        self.rim=Q.QPushButton('选择袋口夹取点');self.rim.clicked.connect(self.choose_rim);right.addWidget(self.rim)
        right.addSpacing(10);self.flow=Q.QLabel('02   预览路径\n\n预抓取 → 抓取 → 抬起 → 收回');self.flow.setWordWrap(True);right.addWidget(self.flow)
        self.plan=Q.QPushButton('规划预览');self.plan.clicked.connect(lambda:self.send('plan',id=self.selected_id()));right.addWidget(self.plan)
        self.execute=Q.QPushButton('确认抓取');self.execute.setEnabled(False);self.execute.clicked.connect(self.start_execution);right.addWidget(self.execute)
        self.stop=Q.QPushButton('停止运动');self.stop.setObjectName('stop');self.stop.clicked.connect(self.stop_motion);right.addWidget(self.stop)
        self.status=Q.QLabel('加载模型中…');self.status.setWordWrap(True);self.status.setStyleSheet('font-size:13px');right.addWidget(self.status)
        self.footer=Q.QLabel(self);self.footer.hide()
        self.timer=C.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(80)
        self.preview_stamp=0;self.motion=None
        from .navigation_ui import NavigationPanel
        (self.run/'heartbeat').touch()
        self.navigation=NavigationPanel(self,start=start_worker);self.navigation.setMaximumWidth(270)
        left.removeWidget(self.video);view_row=Q.QHBoxLayout();view_row.addWidget(self.video,1);view_row.addWidget(self.navigation)
        left.insertLayout(0,view_row,1)
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

    def selected_id(self):
        item=self.list.currentItem();return item.data(C.Qt.UserRole) if item else None

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
        idle=self.ready and not self.busy and self.motion is None and not self.demo_active
        if hasattr(self,'navigation'):idle=idle and not self.navigation.inflight
        self.demo.setEnabled(idle);self.demo_settings.setEnabled(idle)
        self.live_button.setEnabled(idle)
        if hasattr(self,'navigation'):self.navigation.controls(idle)
        self.chassis_button.setEnabled(idle and self.chassis_dialog is None)
        self.scan.setEnabled(idle);self.list.setEnabled(idle)
        self.mode.setEnabled(idle);self.manage.setEnabled(idle)
        if self.catalog_dialog:
            for b in self.catalog_dialog.buttons:b.setEnabled(idle and bool(self.catalog_dialog.current()))
        self.prepare.setEnabled(idle)
        chosen=self.selected_id() is not None
        self.enroll.setEnabled(idle and chosen);self.plan.setEnabled(idle and chosen)
        item=next((v for v in self.items if v['id']==self.selected_id()),{})
        self.rim.setVisible(item.get('geometry')=='bag_edge')
        self.rim.setEnabled(idle and self.frozen and item.get('geometry')=='bag_edge')
        if self.demo.isChecked():
            self.execute.setText('确认演示抓取');self.execute.setEnabled(idle)

    def send(self,action,**values):
        if not self.process:return
        self.invalidate_plan()
        if action=='plan':
            chosen=next((v for v in self.items if v['id']==values.get('id')),None)
            self.expected_target=({'instance_id':chosen['id'],'product_id':chosen.get('product_id')}
                                  if chosen else None)
        self.request+=1;self.busy=True;self.cancel_pending=False;self.execute.setEnabled(False);self.video.path=[]
        self.process.write((json.dumps({'action':action,'request_id':self.request,**values},ensure_ascii=False)+'\n').encode())
        self.status.setText({'scan':'正在识别饮料…','plan':'正在定位和规划…','enroll':'保存商品样本…','prepare':'检查原VR复位路径…','remove_product':'正在删除商品…'}.get(action,'处理中…'))
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
        if kind=='status':self.status.setText(data['text']);return
        self.busy=False
        if 'mode' in data:
            self.mode.blockSignals(True);self.mode.setCurrentIndex(max(0,self.mode.findData(data['mode'])));self.mode.blockSignals(False)
        if kind=='ready':
            self.ready=True;self.blockers=data['blockers'];self.status.setText('模型已就绪。物品分开放好，点击“识别物品”。')
            self.execute.setToolTip('；'.join(self.blockers))
        elif kind=='reset_planned':
            self.live();self.items=[];self.list.clear()
            self.status.setText('保持式使能，复位到原VR位姿并张开双爪…')
            if data['blockers']:
                self.status.setText('；'.join(data['blockers']))
                if self.demo_active:self.demo_failure('；'.join(data['blockers']))
            else:self.launch_motion('last_reset_plan.json',preparing=True)
        elif kind=='observe_planned':
            if not self.demo_active or self.demo_phase!='planning':return
            self.demo_context['observation_plan_id']=data['plan_id']
            self.live();self.status.setText('保持夹紧，轻抬手臂后重新观察桌面…')
            self.demo_phase='observe';self.launch_motion(data['filename'],placement=data)
        elif kind=='place_planned':
            if not self.demo_active or self.demo_phase!='planning':return
            self.video.pix=G.QPixmap(str(self.run/'scene.jpg'));self.video.shape=data['shape']
            self.video.path=data['pixels'];self.video.boxes=[];self.frozen=True
            self.view_text.setText('自动放置点 · 桌面与轨迹检查已通过')
            self.demo_phase='place';self.launch_motion(data['filename'],placement=data)
        elif kind=='prepared':
            self.live();self.send('scan')
        elif kind in ('scene','enrolled','catalog'):
            previous_id=self.selected_id();self.items=data['items'];self.list.clear()
            for v in self.items:
                dims=v.get('dimensions_mm')
                text=f'#{v["id"]}  {v["name"]}\n'+(f'宽 {dims[0]:g} · 高 {dims[1]:g} mm' if dims else '选中后录入商品')
                item=Q.QListWidgetItem(text);item.setData(C.Qt.UserRole,v['id']);self.list.addItem(item)
            if kind=='scene':
                self.video.shape=data['shape'];self.video.pix=G.QPixmap(str(self.run/'scene.jpg'))
                self.frozen=True;self.video.boxes=self.items;self.view_text.setText('识别快照 · 规划前会重新确认目标')
                self.status.setText(f'找到 {len(self.items)} 个候选 · 识别 {data["elapsed_ms"]:.0f} ms')
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
                self.controls();return
            self.pending_plan=data
            self.frozen=True;self.video.pix=G.QPixmap(str(self.run/'scene.jpg'));self.video.shape=data['shape'];self.video.boxes=[];self.video.path=data['pixels']
            mode='柔性夹取' if data.get('orientation_mode')=='flexible' else '水平夹取'
            retreat=data.get('withdraw_yaw_deg',0.)
            retreat_text=f' · 收回转向 {retreat:+.0f}°' if retreat else ''
            if data.get('withdraw_orientation_mode')=='adaptive':retreat_text=' · 自然收回'
            if data.get('withdraw_orientation_mode')=='fixed_carry':retreat_text=' · 固定持物姿态'
            self.status.setText(f'{target["name"]} · {mode} · 接近 {data.get("approach_yaw_deg",0):.0f}°{retreat_text} · {data["seconds"]:.1f}s')
            self.view_text.setText('路径预览 · 1预抓取  2抓取  3抬起  4退离'+('  5固定持物' if data.get('carry_protocol_version') else ''))
            self.blockers=data['blockers'];self.execute.setToolTip('；'.join(self.blockers));self.execute.setEnabled(not self.blockers)
            self.execute.setText('确认抓取 · '+target['name'])
            if self.blockers:self.status.setText(target['name']+'路径已预览；'+'；'.join(self.blockers))
            self.footer.setText('仅预览 · '+('；'.join(self.blockers[1:3]) if self.blockers else '请确认操作空间清空'))
            if self.demo_active and self.demo_phase=='grasp_plan':self.demo_plan_ready(data)
        elif kind in ('error','fatal'):
            self.demo_active=False;self.demo_phase=None
            self.invalidate_plan();self.status.setText(data['text'])
            if self.catalog_dialog and self.catalog_dialog.isVisible():self.catalog_dialog.info.setText(data['text'])
            if kind=='fatal':self.ready=False
        self.video.update();self.controls()

    def invalidate_plan(self):
        self.pending_plan=None;self.expected_target=None;self.execute.setEnabled(False);self.execute.setText('确认抓取')

    def select(self,*_):
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
        self.frozen=False;self.video.boxes=[];self.video.path=[];self.execute.setEnabled(False);self.view_text.setText('实时 RGB');self.preview_stamp=0

    def image_clicked(self,x,y):
        if self.busy or self.motion:return
        if self.picking_rim:
            self.picking_rim=False;self.send('grasp_point',id=self.selected_id(),x=x,y=y);return
        for i,v in enumerate(self.items):
            a,b,c,d=v['box'];px=x*self.video.shape[1];py=y*self.video.shape[0]
            if self.frozen and a<=px<=c and b<=py<=d:self.list.setCurrentRow(i);break

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

    def start_execution(self):
        if self.motion:return
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
        if Q.QMessageBox.question(self,'确认抓取：'+data['target']['name'],'使用当前手眼标定及15mm夹取中心补偿；实物精度尚未验收。\n抓取段不检查桌面、货架及邻物，请确认预览路径空旷。\n闭爪稳定后自动抬起并收回；尚无夹稳检测。\n人员离开运动范围并准备实体急停，开始？',defaultButton=Q.QMessageBox.No)!=Q.QMessageBox.Yes:return
        self.launch_motion(data['filename'])

    def launch_motion(self,filename,preparing=False,placement=None):
        if self.cancel_pending:return
        extra=[]
        if placement is not None:
            if not self.demo_active or self.demo_phase not in ('place','observe'):return
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
            self.demo_active=False;self.demo_phase=None
            self.motion=None;self.status.setText('运动进程启动失败，未执行运动。');self.controls()

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
                self.demo_active=False;self.demo_phase=None;self.live()
                self.status.setText('演示完成：已执行放置并空手收回。');self.controls();return
            self.demo_active=False;self.demo_phase=None
            self.status.setText('演示已停止：'+result.get('reason','前一阶段未成功，未继续后续动作。'));self.controls();return
        self.status.setText('已停止：'+result['reason'] if result.get('reason') else '复位完成。' if result.get('status')=='prepared' else '抓取流程完成。' if result.get('status')=='completed' else '运动进程意外结束，请检查日志。');self.controls()
        if self.preparing and (self.run/'prepare_result.json').exists():
            receipt=json.loads((self.run/'prepare_result.json').read_text('utf-8'))
            if receipt.get('status')=='prepared':self.send('prepared')

    def stop_motion(self):
        if hasattr(self,'navigation'):self.navigation.command('stop')
        self.demo_active=False
        if self.motion is None:self.demo_phase=None
        if self.chassis_dialog:self.chassis_dialog.stop()
        self.invalidate_plan()
        if self.busy and self.request>0:self.cancel_pending=True
        (self.run/'stop').touch();self.execute.setEnabled(False)
        self.status.setText('已请求停止；危险时请按实体急停。')

    def tick(self):
        (self.run/'heartbeat').touch()
        path=self.run/'preview.jpg'
        if self.frozen:return
        if path.exists():
            stamp=path.stat().st_mtime
            if time.time()-stamp>1.:
                self.video.pix=G.QPixmap();self.video.update();return
            if stamp!=self.preview_stamp:
                self.video.pix=G.QPixmap(str(path));self.preview_stamp=stamp;self.video.update()

    def finished(self,*_):
        if self.demo_active:self.stop_motion()
        self.ready=False;self.busy=False;self.controls()

    def closeEvent(self,event):
        if self.demo_active:self.stop_motion()
        if self.chassis_dialog:self.chassis_dialog.shutdown()
        if self.motion:
            self.stop_motion();self.status.setText('停止运动后再关闭界面');event.ignore();return
        if not self.navigation.shutdown():
            self.status.setText('等待本程序的导航取消确认，请停稳后再次关闭');event.ignore();return
        if self.process and self.process.state()!=C.QProcess.NotRunning:
            self.process.write(b'{"action":"quit"}\n');self.process.closeWriteChannel()
            if not self.process.waitForFinished(2000):self.process.terminate();self.process.waitForFinished(1000)
            if self.process.state()!=C.QProcess.NotRunning:self.process.kill();self.process.waitForFinished(1000)
        event.accept()


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
