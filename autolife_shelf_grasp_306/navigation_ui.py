"""Compact live map and shared waypoint selectors; no navigation reimplementation."""
import json,math,time
from PyQt5 import QtCore as C,QtGui as G,QtWidgets as Q
from .navigation_map import world_to_pixel,manual_pose
from .navigation_logic import navigation_summary


class MapView(Q.QWidget):
    expanded=C.pyqtSignal()
    pose_selected=C.pyqtSignal(float,float,float)
    hint=C.pyqtSignal(str)
    def __init__(self,run,large=False):
        super().__init__();self.run=run;self.data={};self.large=large;self.manual=False
        self.zoom=1.;self.pan=C.QPointF();self.drag=None;self.end=None;self.pan_start=None
        self.setMinimumHeight(400 if large else 140)
        if not large:self.setMaximumHeight(190)
        self.setMouseTracking(True)
    def map_rect(self):
        m=self.data.get('meta')
        if not m:return C.QRectF()
        scale=min(self.width()/m['width'],self.height()/m['height'])*self.zoom
        w,h=m['width']*scale,m['height']*scale
        return C.QRectF((self.width()-w)/2+self.pan.x(),(self.height()-h)/2+self.pan.y(),w,h)
    def screen_point(self,x,y):
        m=self.data['meta'];px,py=world_to_pixel(m,x,y);r=self.map_rect()
        return C.QPointF(r.x()+px/m['width']*r.width(),r.y()+py/m['height']*r.height())
    def image_point(self,p):
        r=self.map_rect();m=self.data['meta']
        return ((p.x()-r.x())/r.width()*m['width'],(p.y()-r.y())/r.height()*m['height'])
    def fit(self):self.zoom=1.;self.pan=C.QPointF();self.update()
    def center_robot(self):
        pose=self.data.get('pose') or self.data.get('pending_pose')
        if not pose or not self.data.get('meta'):return
        self.pan+=C.QPointF(self.width()/2,self.height()/2)-self.screen_point(*pose[:2]);self.update()
    def set_manual(self,enabled):
        self.manual=enabled;self.drag=None;self.end=None
        self.setCursor(C.Qt.CrossCursor if enabled else C.Qt.ArrowCursor);self.update()
    def mouseDoubleClickEvent(self,event):
        if not self.large:self.expanded.emit()
    def wheelEvent(self,event):
        if not self.large:return
        old=self.zoom;self.zoom=min(8.,max(1.,old*(1.2 if event.angleDelta().y()>0 else 1/1.2)))
        center=C.QPointF(self.width()/2,self.height()/2);at=event.posF()-center
        self.pan=at-(at-self.pan)*(self.zoom/old);self.update()
    def mousePressEvent(self,event):
        if not self.large or not self.data.get('meta'):return
        if event.button()==C.Qt.LeftButton and self.manual and self.map_rect().contains(event.localPos()):
            self.drag=self.image_point(event.localPos());self.end=self.drag;self.update()
        elif event.button() in (C.Qt.LeftButton,C.Qt.MiddleButton):self.pan_start=(event.localPos(),C.QPointF(self.pan))
    def mouseMoveEvent(self,event):
        if self.drag:self.end=self.image_point(event.localPos());self.update()
        elif self.pan_start:self.pan=self.pan_start[1]+event.localPos()-self.pan_start[0];self.update()
    def mouseReleaseEvent(self,event):
        self.pan_start=None
        if self.drag:
            try:
                pose=manual_pose(self.data['meta'],self.drag,self.image_point(event.localPos()))
                self.pose_selected.emit(*pose)
            except ValueError as exc:self.hint.emit(str(exc))
            self.drag=None;self.end=None;self.update()
    def paintEvent(self,event):
        p=G.QPainter(self);p.fillRect(self.rect(),G.QColor('#090e14'))
        pix=G.QPixmap(str(self.run/'navigation_map.png'));meta=self.data.get('meta')
        if pix.isNull() or not meta:
            p.setPen(G.QColor('#91a4b5'));p.drawText(self.rect(),C.Qt.AlignCenter,'点击“启动导航”加载地图');return
        rect=self.map_rect()
        p.drawPixmap(rect,pix,C.QRectF(pix.rect()))
        def point(x,y):
            return self.screen_point(x,y)
        p.setClipRect(rect)
        obstacle=G.QPixmap(str(self.run/'navigation_obstacles.png'));om=self.data.get('obstacle_meta')
        if not obstacle.isNull() and om and abs(om.get('yaw',0)-meta['yaw'])<1e-6:
            # Costmap grid is axis-aligned in map; both corners go through the
            # same world transform as the robot, including non-zero map yaw.
            a=point(om['x'],om['y']+om['height']*om['resolution']);b=point(om['x']+om['width']*om['resolution'],om['y'])
            p.drawPixmap(C.QRectF(a,b).normalized(),obstacle,C.QRectF(obstacle.rect()))
        for name,pose in self.data.get('waypoints',{}).items():
            pt=point(*pose[:2]);p.setPen(G.QColor('#19bacc'));p.setBrush(G.QColor('#19bacc'));p.drawEllipse(pt,3,3)
            p.drawText(pt+C.QPointF(5,-3),name)
        pose=self.data.get('pose') or self.data.get('pending_pose')
        if pose:
            pending=not self.data.get('pose')
            pt=point(*pose[:2]);a=pose[2]-meta['yaw']
            forward=C.QPointF(math.cos(a),-math.sin(a));side=C.QPointF(math.sin(a),math.cos(a))
            p.setRenderHint(G.QPainter.Antialiasing)
            p.setPen(G.QPen(G.QColor('#073d28'),2));p.setBrush(G.QColor('#ffb45c' if pending else '#45f59a'))
            p.drawPolygon(G.QPolygonF([pt+forward*16,pt-forward*8+side*8,pt-forward*8-side*8]))
            if self.large:
                label=C.QRectF(pt+C.QPointF(18,-26),C.QSizeF(155,24))
                p.fillRect(label,G.QColor('#13251e'));p.setPen(G.QColor('#82ffc1'))
                p.drawText(label,C.Qt.AlignCenter,'指定位置 · 等待确认' if pending else '实时机器人')
        if self.drag and self.end:
            def sp(v):return C.QPointF(rect.x()+v[0]/meta['width']*rect.width(),rect.y()+v[1]/meta['height']*rect.height())
            p.setPen(G.QPen(G.QColor('#88ff88'),3));p.drawLine(sp(self.drag),sp(self.end));p.drawEllipse(sp(self.drag),5,5)


class NavigationDialog(Q.QDialog):
    def __init__(self,panel,parent=None):
        super().__init__(parent or panel);self.panel=panel;self.setWindowTitle('导航地图 · 定位');self.resize(1100,800)
        layout=Q.QVBoxLayout(self);row=Q.QHBoxLayout();layout.addLayout(row)
        self.start=Q.QPushButton('启动导航');self.global_button=Q.QPushButton('全局重定位')
        self.manual=Q.QPushButton('手动重定位');self.manual.setCheckable(True)
        fit=Q.QPushButton('适应窗口');self.center=Q.QPushButton('定位机器人');stop=Q.QPushButton('停止当前任务')
        for b in (self.start,self.global_button,self.manual,fit,self.center,stop):row.addWidget(b)
        self.start.clicked.connect(panel.start_navigation);self.global_button.clicked.connect(panel.relocalize)
        stop.clicked.connect(panel.owner.stop_motion)
        self.map=MapView(panel.owner.run,large=True);layout.addWidget(self.map,1)
        self.manual.toggled.connect(self.map.set_manual);fit.clicked.connect(self.map.fit)
        self.center.clicked.connect(self.map.center_robot)
        self.map.pose_selected.connect(self.selected);self.map.hint.connect(lambda s:self.instructions.setText(s))
        self.instructions=Q.QLabel('滚轮缩放，拖动平移。手动重定位：点机器人实际位置，拖出车头朝向。红色为实时障碍。')
        self.instructions.setWordWrap(True);layout.addWidget(self.instructions)
        self.status=Q.QLabel();self.status.setWordWrap(True);layout.addWidget(self.status)
        self.coordinates=Q.QLabel();self.coordinates.setStyleSheet('color:#45c98b;font-size:15px');layout.addWidget(self.coordinates)
        self.refresh()
    def selected(self,x,y,yaw):
        if not self.manual.isEnabled():return
        if Q.QMessageBox.question(self,'确认手动定位',f'机器人位置：({x:.2f}, {y:.2f})，朝向：{math.degrees(yaw):.1f}°。\n请确认与现场一致；这只更新定位，不移动机器人。',defaultButton=Q.QMessageBox.No)==Q.QMessageBox.Yes:
            self.panel.command('initial_pose',x=x,y=y,yaw=yaw,map=self.panel.data.get('active_map'))
            self.manual.setChecked(False)
    def refresh(self):
        data=self.panel.data
        if self.map.data.get('active_map')!=data.get('active_map'):
            self.manual.setChecked(False);self.map.fit()
        self.map.data=data;self.map.update();self.status.setText(self.panel.info.text())
        pose=data.get('pose');pending=data.get('pending_pose');self.center.setEnabled(bool(pose or pending))
        self.coordinates.setText(f'机器人坐标：X {pose[0]:.3f} m   Y {pose[1]:.3f} m   朝向 {math.degrees(pose[2]):.1f}°' if pose else '机器人坐标：等待有效定位确认')
        if not pose and pending:
            self.coordinates.setText(f'指定坐标：X {pending[0]:.3f} m   Y {pending[1]:.3f} m   朝向 {math.degrees(pending[2]):.1f}° · 待AMCL确认')
        allowed=self.panel.edit_allowed
        self.start.setEnabled(allowed and not data.get('ready') and not data.get('starting'))
        self.global_button.setEnabled(allowed and data.get('localization_ready',False))
        self.manual.setEnabled(allowed and data.get('manual_localization_ready',data.get('localization_ready',False)))
        if not self.manual.isEnabled():self.manual.setChecked(False)


class NavigationPanel(Q.QWidget):
    def __init__(self,owner,start=True):
        super().__init__();self.owner=owner;self.data={};self.buffer=b'';self.proc=None;self.closing=False;self.inflight=False
        self.dialog=None;self.edit_allowed=False;self.notice='';self.notice_until=0.
        layout=Q.QVBoxLayout(self);layout.setContentsMargins(0,0,0,0);layout.setSpacing(5)
        self.map=MapView(owner.run);layout.addWidget(self.map)
        row=Q.QHBoxLayout();layout.addLayout(row)
        self.start_button=Q.QPushButton('启动导航');row.addWidget(self.start_button);self.start_button.clicked.connect(self.start_navigation)
        self.expand=Q.QPushButton('放大地图 / 定位');row.addWidget(self.expand);self.expand.clicked.connect(self.show_map);self.map.expanded.connect(self.show_map)
        row=Q.QHBoxLayout();layout.addLayout(row);self.maps=Q.QComboBox();row.addWidget(self.maps,1)
        self.load=Q.QPushButton('换图');self.load.clicked.connect(self.change_map);row.addWidget(self.load)
        form=Q.QFormLayout();layout.addLayout(form)
        self.front=Q.QComboBox();self.shelf=Q.QComboBox();self.product=Q.QComboBox()
        form.addRow('前台航点',self.front);form.addRow('货架航点',self.shelf);form.addRow('演示饮料',self.product)
        row=Q.QHBoxLayout();layout.addLayout(row)
        self.list_button=Q.QPushButton('航点列表');row.addWidget(self.list_button);self.list_button.clicked.connect(self.show_waypoints)
        self.locate=Q.QPushButton('全局重定位');row.addWidget(self.locate);self.locate.clicked.connect(self.relocalize)
        self.info=Q.QLabel('连接共享导航后台…');self.info.setWordWrap(True);self.info.setStyleSheet('font-size:12px;color:#91a4b5');layout.addWidget(self.info)
        self.saved={}
        try:self.saved=json.loads((owner.root/'data/demo_route.json').read_text())
        except (ValueError,OSError):pass
        self.refresh_products()
        self.product_timer=C.QTimer(self);self.product_timer.timeout.connect(self.refresh_products);self.product_timer.start(2000)
        if start:
            self.proc=C.QProcess(self);self.proc.readyReadStandardOutput.connect(self.read)
            self.proc.readyReadStandardError.connect(self.log)
            self.proc.finished.connect(self.finished)
            self.proc.start('/usr/bin/python3',['-m','autolife_shelf_grasp_306.navigation_bridge','--run',str(owner.run)])
    def refill(self,box,entries,default=None):
        selected=box.currentData() or default
        if [(box.itemText(i),box.itemData(i)) for i in range(box.count())]==entries:return
        box.blockSignals(True);box.clear()
        for label,value in entries:box.addItem(label,value)
        box.setCurrentIndex(max(0,box.findData(selected)));box.blockSignals(False)
    def refresh_products(self):
        if self.owner.demo_active:return
        try:
            products=json.loads((self.owner.root/'data/products.json').read_text())['products']
            self.refill(self.product,[(p['name'],p['id']) for p in products if p.get('geometry','cylinder')=='cylinder' and .08<=p['height_m']<=.4],self.saved.get('product_id'))
        except (ValueError,OSError,KeyError):pass
    def controls(self,idle):
        allowed=idle and not self.data.get('busy',False) and not self.data.get('starting',False)
        self.edit_allowed=allowed
        for widget in (self.maps,self.load,self.front,self.shelf,self.product):widget.setEnabled(allowed)
        self.load.setEnabled(allowed and self.data.get('ready',False))
        self.locate.setEnabled(allowed and self.data.get('localization_ready',False))
        self.start_button.setEnabled(allowed and not self.data.get('ready',False))
        self.start_button.setText('导航已连接' if self.data.get('ready') else '启动中…' if self.data.get('starting') else '启动导航')
        if self.dialog:self.dialog.refresh()
    def show_map(self):
        if self.dialog:self.dialog.raise_();self.dialog.activateWindow();return
        self.dialog=NavigationDialog(self);self.dialog.setAttribute(C.Qt.WA_DeleteOnClose)
        self.dialog.finished.connect(lambda _:setattr(self,'dialog',None));self.dialog.show()
    def start_navigation(self):
        if Q.QMessageBox.question(self,'启动共享导航','将启动与控制中心相同的导航后台；如原厂导航正在运行，会停止原厂导航以避免冲突。\n不会发起移动目标，不重启机械臂或底盘服务。继续？',defaultButton=Q.QMessageBox.No)==Q.QMessageBox.Yes:
            self.command('start_navigation');self.show_notice('正在启动导航…')
    def relocalize(self):
        if Q.QMessageBox.question(self,'全局重定位','请确保机器人停稳。重新搜索全地图定位？此操作不会移动机器人。',defaultButton=Q.QMessageBox.No)==Q.QMessageBox.Yes:
            self.command('relocalize');self.show_notice('正在请求全局重定位…')
    def show_notice(self,text):
        self.notice=text;self.notice_until=time.monotonic()+8;self.info.setText(text)
        if self.dialog:self.dialog.refresh()
    def selection(self):
        if time.time()-self.data.get('snapshot_time',0)>2.:raise ValueError('抓取窗口的导航状态更新中断，请重开抓取窗口')
        if self.data.get('map_fault'):raise ValueError('地图切换尚未完整确认，请重新加载地图')
        if not self.data.get('ready'):raise ValueError('请点击“启动导航”，然后在放大地图中完成定位')
        if not self.data.get('pose'):raise ValueError(self.data.get('pose_reason') or '尚未收到有效定位坐标')
        front=self.front.currentData();shelf=self.shelf.currentData();pid=self.product.currentData()
        if not front or not shelf or front==shelf or not pid:raise ValueError('请选择不同的前台、货架航点和饮料')
        record=dict(map=self.data['active_map'],front=front,shelf=shelf,product_id=pid,product_name=self.product.currentText())
        from .core import atomic_json
        atomic_json(self.owner.root/'data/demo_route.json',record)
        return record
    def command(self,action,**data):
        if self.proc and self.proc.state()==C.QProcess.Running:
            if action=='navigate':self.inflight=True
            self.proc.write((json.dumps(dict(action=action,**data),ensure_ascii=False)+'\n').encode())
        else:self.info.setText('导航连接未启动，请重新打开抓取程序')
    def change_map(self):
        if Q.QMessageBox.question(self,'更换共享地图','将同步更换控制中心的地图与航点库，并重新定位。继续？',defaultButton=Q.QMessageBox.No)==Q.QMessageBox.Yes:
            self.command('map',path=self.maps.currentData());self.info.setText('正在更换地图…')
    def show_waypoints(self):
        text='\n'.join(self.data.get('names',[]) or self.data.get('library',{})) or '当前地图没有航点，请在控制中心保存。'
        Q.QMessageBox.information(self,'当前地图航点',text)
    def read(self):
        self.buffer+=bytes(self.proc.readAllStandardOutput())
        while b'\n' in self.buffer:
            line,self.buffer=self.buffer.split(b'\n',1)
            try:data=json.loads(line)
            except ValueError:continue
            event=data.get('event')
            if event=='nav_snapshot':
                previous=self.data.get('active_map');self.data=data;self.map.data=data;self.map.update()
                self.refill(self.maps,[(('当前 · ' if m['path']==data['active_map'] else '')+m['name'],m['path']) for m in data['maps']],data['active_map'])
                if previous and previous!=data['active_map']:
                    self.front.clear();self.shelf.clear();self.saved={}
                entries=[(n,n) for n in (data['names'] or data.get('library',{}))]
                self.refill(self.front,entries,self.saved.get('front'));self.refill(self.shelf,entries,self.saved.get('shelf'))
                text=navigation_summary(data,self.inflight)
                self.info.setText(self.notice if time.monotonic()<self.notice_until else text)
                self.owner.controls()
            elif event=='nav_started':
                workflow=getattr(self.owner,'workflow',None)
                if workflow:workflow.navigation_started(data)
            elif event=='nav_done':
                self.inflight=False;self.owner.navigation_done(data)
            elif event=='nav_error':
                self.show_notice(data['text'])
                workflow=getattr(self.owner,'workflow',None)
                if workflow and workflow.active and data.get('run_id')==workflow.cursor.run_id:
                    self.inflight=False;workflow.fail(data['text'])
                if self.owner.demo_context and data.get('run_id')==self.owner.demo_context.get('run_id'):
                    self.inflight=False
                    self.owner.demo_failure(data['text'])
            elif event=='nav_message':self.show_notice(data['text'])
    def log(self):
        with (self.owner.run/'navigation.log').open('ab') as stream:stream.write(bytes(self.proc.readAllStandardError()))
    def finished(self):
        self.inflight=False
        self.info.setText('导航连接已关闭')
        if self.owner.demo_active:self.owner.demo_failure('导航连接中断，未继续抓放；请在控制中心确认底盘已停止')
        workflow=getattr(self.owner,'workflow',None)
        if workflow and workflow.active:workflow.fail('导航连接中断，请确认底盘已停止')
    def shutdown(self):
        self.product_timer.stop()
        if self.proc and self.proc.state()!=C.QProcess.NotRunning:
            self.command('quit')
            return self.proc.waitForFinished(1000)
        return True
