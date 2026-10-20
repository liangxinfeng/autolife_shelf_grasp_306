"""Portrait-first robot display; rotation is local to this window only."""
from PyQt5 import QtCore as C,QtGui as G,QtWidgets as Q


class Logo(Q.QWidget):
    def __init__(self,pix,parent=None):
        super().__init__(parent);self.pix=pix;self.setMinimumSize(0,0)
        self.setSizePolicy(Q.QSizePolicy.Ignored,Q.QSizePolicy.Ignored)
    def paintEvent(self,event):
        p=G.QPainter(self);p.setRenderHint(G.QPainter.SmoothPixmapTransform)
        # Treat the image and English name as one centered, scalable brand block.
        available_w=max(1,int(self.width()*.78))
        available_h=max(1,int(self.height()*.78))
        font=G.QFont('Noto Sans CJK SC')
        font.setPixelSize(max(16,min(68,int(available_h*.18),int(available_w*.14))))
        font.setBold(True)
        font.setLetterSpacing(G.QFont.AbsoluteSpacing,3)
        p.setFont(font)
        text_h=G.QFontMetrics(font).height()
        gap=max(4,int(text_h*.22))
        target=self.pix.size() if not self.pix.isNull() else C.QSize(300,150)
        target.scale(C.QSize(available_w,max(1,available_h-text_h-gap)),C.Qt.KeepAspectRatio)
        top=max(0,(self.height()-target.height()-gap-text_h)//2-int(self.height()*.045))
        logo_rect=C.QRect((self.width()-target.width())//2,top,target.width(),target.height())
        p.setPen(G.QColor('#edf7ff'))
        if self.pix.isNull():
            p.drawText(logo_rect,C.Qt.AlignCenter,'云蝶科技')
        else:
            p.drawPixmap(logo_rect,self.pix)
        p.drawText(C.QRect(0,top+target.height()+gap,self.width(),text_h),
                   C.Qt.AlignCenter,'UNDY')


class TaskDisplay(Q.QGraphicsView):
    def __init__(self,page):
        super().__init__(page);self.setWindowFlags(C.Qt.Window);self.page=page
        self.setWindowTitle('云蝶 · 自动任务');self.resize(1100,750)
        self.setHorizontalScrollBarPolicy(C.Qt.ScrollBarAlwaysOff);self.setVerticalScrollBarPolicy(C.Qt.ScrollBarAlwaysOff)
        self.setFrameShape(Q.QFrame.NoFrame);self.setBackgroundBrush(G.QColor('#071c30'))
        self.scene_=Q.QGraphicsScene(self);self.setScene(self.scene_)
        self.content=Q.QWidget();self.content.setStyleSheet(
            'QWidget{background:#071c30;color:#edf7ff;} QLabel{background:transparent;}'
            'QPushButton{background:#154669;color:white;border:1px solid #41809b;border-radius:10px;padding:14px;font-size:22px;}'
            'QPushButton:disabled{color:#70818c;background:#162734;} QPushButton#stop{background:#b62838;border:1px solid #fa7484;}')
        layout=Q.QVBoxLayout(self.content);layout.setContentsMargins(24,18,24,22)
        top=Q.QHBoxLayout();top.addStretch();self.fullscreen=Q.QPushButton('缩小窗口')
        self.fullscreen.clicked.connect(self.toggle_fullscreen);top.addWidget(self.fullscreen);layout.addLayout(top)
        logo_path=page.owner.root.parent/'autolife_robot_control_center/logo/yundie.webp'
        self.pix=G.QPixmap()
        try:
            from PIL import Image
            with Image.open(logo_path) as source:
                im=source.convert('RGBA');bbox=im.getchannel('A').getbbox()
                if bbox:im=im.crop(bbox)
                raw=im.tobytes();self.pix=G.QPixmap.fromImage(G.QImage(raw,im.width,im.height,im.width*4,G.QImage.Format_RGBA8888).copy())
        except (ImportError,OSError,ValueError):self.pix.load(str(logo_path))
        self.logo=Logo(self.pix);layout.addWidget(self.logo,7)
        self.title=Q.QLabel();self.title.setWordWrap(True);self.title.setAlignment(C.Qt.AlignCenter)
        self.title.setStyleSheet('font-size:36px;font-weight:700;');layout.addWidget(self.title)
        self.state=Q.QLabel();self.state.setWordWrap(True);self.state.setAlignment(C.Qt.AlignCenter)
        self.state.setStyleSheet('font-size:26px;color:#89e0e4;');layout.addWidget(self.state,2)
        self.again=Q.QPushButton('再次运行单次任务');self.again.clicked.connect(page.repeat_single);layout.addWidget(self.again)
        self.pause=Q.QPushButton('暂停任务');self.pause.clicked.connect(page.toggle_pause);layout.addWidget(self.pause)
        stop=Q.QPushButton('紧急停止任务');stop.setObjectName('stop');stop.clicked.connect(page.stop);layout.addWidget(stop)
        note=Q.QLabel('新任务从空载开始；危险时按机器人实体急停。');note.setWordWrap(True);note.setAlignment(C.Qt.AlignCenter);layout.addWidget(note)
        self.proxy=self.scene_.addWidget(self.content);self.relayout()

    def relayout(self):
        if not hasattr(self,'proxy'):return
        w=max(1,self.viewport().width());h=max(1,self.viewport().height())
        mode=getattr(self.page,'display_direction',None)
        value=mode.currentData() if mode else 'robot'
        angle=(-90 if value=='robot' else 90) if self.isFullScreen() and w>h and value!='desktop' else 0
        self.resetTransform()
        self.proxy.setRotation(0);self.proxy.setPos(0,0)
        self.content.resize(h if angle else w,w if angle else h)
        self.proxy.setRotation(angle)
        bounds=self.proxy.sceneBoundingRect();self.proxy.setPos(-bounds.left(),-bounds.top())
        # Qt minimum layout sizes can exceed the viewport. Fit the entire
        # interactive scene rather than silently cropping its bottom/edges.
        bounds=self.proxy.sceneBoundingRect()
        self.scene_.setSceneRect(bounds)
        self.fitInView(bounds.adjusted(-12,-12,12,12),C.Qt.KeepAspectRatio)

    def resizeEvent(self,event):super().resizeEvent(event);self.relayout()
    def showEvent(self,event):super().showEvent(event);C.QTimer.singleShot(0,self.relayout)
    def toggle_fullscreen(self):
        if self.isFullScreen():self.showNormal();self.fullscreen.setText('全屏显示')
        else:self.showFullScreen();self.fullscreen.setText('缩小窗口')
        self.relayout()
    def keyPressEvent(self,event):
        if event.key()==C.Qt.Key_Escape:self.showNormal();self.fullscreen.setText('全屏显示');self.relayout();event.accept()
        elif event.key()==C.Qt.Key_Space:self.page.stop();event.accept()
        else:super().keyPressEvent(event)
    def closeEvent(self,event):
        if self.page.active:self.page.stop()
        event.accept()

