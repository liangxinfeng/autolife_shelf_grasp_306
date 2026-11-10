"""物体录入专用拖框视图；不改变抓取页原有单击选择逻辑。"""
from PyQt5 import QtCore as C,QtGui as G
from .app import ImageView


class BoxImageView(ImageView):
    boxSelected=C.pyqtSignal(list)
    def __init__(self):
        super().__init__();self.drag=None;self.selection=None;self.selection_enabled=False
    def normalized(self,pos):
        r=self.rect_image
        return C.QPointF(max(0.,min(1.,(pos.x()-r.x())/r.width())),max(0.,min(1.,(pos.y()-r.y())/r.height())))
    def mousePressEvent(self,event):
        if self.selection_enabled and event.button()==C.Qt.LeftButton and self.rect_image.contains(event.localPos()):
            self.drag=self.normalized(event.localPos());self.selection=[self.drag.x(),self.drag.y(),self.drag.x(),self.drag.y()];self.update()
    def mouseMoveEvent(self,event):
        if self.drag is not None:
            end=self.normalized(event.localPos());self.selection=[self.drag.x(),self.drag.y(),end.x(),end.y()];self.update()
    def mouseReleaseEvent(self,event):
        if self.drag is None or event.button()!=C.Qt.LeftButton:return
        end=self.normalized(event.localPos());start=self.drag;self.drag=None
        self.selection=[min(start.x(),end.x()),min(start.y(),end.y()),max(start.x(),end.x()),max(start.y(),end.y())]
        self.update();self.boxSelected.emit(list(self.selection))
    def paintEvent(self,event):
        super().paintEvent(event)
        if self.selection:
            r=self.rect_image;x0,y0,x1,y1=self.selection;p=G.QPainter(self);p.setPen(G.QPen(G.QColor('#ffc86c'),2,C.Qt.DashLine));p.setBrush(C.Qt.NoBrush)
            p.drawRect(C.QRectF(r.x()+x0*r.width(),r.y()+y0*r.height(),(x1-x0)*r.width(),(y1-y0)*r.height()))
