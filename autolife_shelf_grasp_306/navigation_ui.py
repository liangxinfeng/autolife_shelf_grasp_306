"""Compact live map and shared waypoint selectors; no navigation reimplementation."""
import json,math,time
from PyQt5 import QtCore as C,QtGui as G,QtWidgets as Q


class MapView(Q.QWidget):
    def __init__(self,run):
        super().__init__();self.run=run;self.data={};self.setMinimumHeight(140);self.setMaximumHeight(190)
    def paintEvent(self,event):
        p=G.QPainter(self);p.fillRect(self.rect(),G.QColor('#090e14'))
        pix=G.QPixmap(str(self.run/'navigation_map.png'));meta=self.data.get('meta')
        if pix.isNull() or not meta:
            p.setPen(G.QColor('#91a4b5'));p.drawText(self.rect(),C.Qt.AlignCenter,'等待控制中心地图');return
        size=pix.size().scaled(self.size(),C.Qt.KeepAspectRatio)
        rect=C.QRectF((self.width()-size.width())/2,(self.height()-size.height())/2,size.width(),size.height())
        p.drawPixmap(rect,pix,C.QRectF(pix.rect()))
        def point(x,y):
            dx=x-meta['x'];dy=y-meta['y'];a=meta['yaw'];res=meta['resolution']
            return C.QPointF(rect.x()+(math.cos(a)*dx+math.sin(a)*dy)/res/meta['width']*rect.width(),
                             rect.bottom()-(-math.sin(a)*dx+math.cos(a)*dy)/res/meta['height']*rect.height())
        p.setClipRect(rect)
        for name,pose in self.data.get('waypoints',{}).items():
            pt=point(*pose[:2]);p.setPen(G.QColor('#19bacc'));p.setBrush(G.QColor('#19bacc'));p.drawEllipse(pt,3,3)
            p.drawText(pt+C.QPointF(5,-3),name)
        pose=self.data.get('pose')
        if pose:
            pt=point(*pose[:2]);a=pose[2]-meta['yaw'];p.setPen(G.QPen(G.QColor('#ffac45'),2));p.setBrush(G.QColor('#ffac45'))
            p.drawEllipse(pt,4,4);p.drawLine(pt,pt+C.QPointF(12*math.cos(a),-12*math.sin(a)))


class NavigationPanel(Q.QWidget):
    def __init__(self,owner,start=True):
        super().__init__();self.owner=owner;self.data={};self.buffer=b'';self.proc=None;self.closing=False;self.inflight=False
        layout=Q.QVBoxLayout(self);layout.setContentsMargins(0,0,0,0);layout.setSpacing(5)
        self.map=MapView(owner.run);layout.addWidget(self.map)
        row=Q.QHBoxLayout();layout.addLayout(row);self.maps=Q.QComboBox();row.addWidget(self.maps,1)
        self.load=Q.QPushButton('换图');self.load.clicked.connect(self.change_map);row.addWidget(self.load)
        form=Q.QFormLayout();layout.addLayout(form)
        self.front=Q.QComboBox();self.shelf=Q.QComboBox();self.product=Q.QComboBox()
        form.addRow('前台航点',self.front);form.addRow('货架航点',self.shelf);form.addRow('演示饮料',self.product)
        row=Q.QHBoxLayout();layout.addLayout(row)
        self.list_button=Q.QPushButton('航点列表');row.addWidget(self.list_button);self.list_button.clicked.connect(self.show_waypoints)
        self.locate=Q.QPushButton('重定位');row.addWidget(self.locate);self.locate.clicked.connect(lambda:self.command('relocalize'))
        self.center=Q.QPushButton('打开控制中心');layout.addWidget(self.center)
        self.center.clicked.connect(lambda:C.QProcess.startDetached('gio',['launch','/home/ubuntu/Desktop/机器人综合控制中心-306.desktop']))
        self.info=Q.QLabel('连接控制中心导航…');self.info.setWordWrap(True);self.info.setStyleSheet('font-size:12px;color:#91a4b5');layout.addWidget(self.info)
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
        allowed=idle and not self.data.get('busy',False)
        for widget in (self.maps,self.load,self.front,self.shelf,self.product,self.locate):widget.setEnabled(allowed)
        self.center.setVisible(not self.data.get('ready',False))
    def selection(self):
        if time.time()-self.data.get('snapshot_time',0)>2.:raise ValueError('抓取窗口的导航状态更新中断，请重开抓取窗口')
        if self.data.get('map_fault'):raise ValueError('地图切换尚未完整确认，请重新加载地图')
        if not self.data.get('ready'):raise ValueError('尚未连接控制中心航点服务，请确认导航模块已启动')
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
                self.info.setText('导航已连接 · 定位正常' if data['ready'] and data['pose'] else data.get('pose_reason','等待定位') if data['ready'] else '地图切换未完成' if data.get('map_fault') else '请在控制中心启动导航模块')
                self.owner.controls()
            elif event=='nav_done':
                self.inflight=False;self.owner.navigation_done(data)
            elif event=='nav_error':
                self.info.setText(data['text'])
                if self.owner.demo_context and data.get('run_id')==self.owner.demo_context.get('run_id'):
                    self.inflight=False
                    self.owner.demo_failure(data['text'])
            elif event=='nav_message':self.info.setText(data['text'])
    def log(self):
        with (self.owner.run/'navigation.log').open('ab') as stream:stream.write(bytes(self.proc.readAllStandardError()))
    def finished(self):
        self.inflight=False
        self.info.setText('导航连接已关闭')
        if self.owner.demo_active:self.owner.demo_failure('导航连接中断，未继续抓放；请在控制中心确认底盘已停止')
    def shutdown(self):
        self.product_timer.stop()
        if self.proc and self.proc.state()!=C.QProcess.NotRunning:
            self.command('quit')
            return self.proc.waitForFinished(1000)
        return True
