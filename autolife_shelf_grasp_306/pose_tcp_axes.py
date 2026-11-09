"""IK末端坐标轴显示层。只做FK/投影/绘图，不连接或发送运动控制。"""
import math
import numpy as np
from PyQt5 import QtCore as C,QtGui as G,QtWidgets as Q

COLORS=('#ff6b6b','#5be69d','#70a7ff')

def axis_points(matrix,tool=False,length=.12):
    t=np.asarray(matrix,float)
    if t.shape!=(4,4) or not np.isfinite(t).all():raise ValueError('末端显示矩阵无效')
    origin=t[:3,3];directions=t[:3,:3] if tool else np.eye(3)
    return origin,np.array([origin+length*d for d in directions.T])

def project(view,point):
    """与本机RobotOpenGLWidget的45°视场和gluLookAt完全一致；返回逻辑像素。"""
    center=np.asarray(view.center,float)
    eye=center+view.distance*np.array([math.cos(view.pitch)*math.cos(view.yaw),math.cos(view.pitch)*math.sin(view.yaw),math.sin(view.pitch)])
    forward=center-eye;forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right)
    up=np.cross(right,forward);delta=np.asarray(point)-eye;depth=float(delta@forward)
    if not .03<depth<30.:return None
    scale=view.height()/(2*math.tan(math.radians(45)/2)*depth)
    return C.QPointF(view.width()/2+float(delta@right)*scale,view.height()/2-float(delta@up)*scale)


class TcpAxesOverlay(Q.QWidget):
    def __init__(self,view):
        super().__init__(view);self.view=view;self.matrix=None;self.tool=False;self.label='';self.label_rects=[]
        self.setAttribute(C.Qt.WA_TransparentForMouseEvents,True)
        self.setAttribute(C.Qt.WA_NoSystemBackground,True)
        self.setAutoFillBackground(False);self.setGeometry(view.rect());view.installEventFilter(self);self.hide()

    def eventFilter(self,obj,event):
        if event.type()==C.QEvent.Resize:self.setGeometry(self.view.rect())
        if event.type() in (C.QEvent.Resize,C.QEvent.MouseMove,C.QEvent.Wheel,C.QEvent.MouseButtonRelease):self.update()
        return False

    def set_frame(self,matrix,tool=False,label=''):
        self.matrix=None if matrix is None else np.asarray(matrix,float).copy();self.tool=bool(tool);self.label=label
        self.setVisible(matrix is not None)
        if matrix is not None:self.raise_();self.update()

    def paintEvent(self,event):
        if self.matrix is None:return
        p=G.QPainter(self);p.setRenderHint(G.QPainter.Antialiasing);font=p.font();font.setPixelSize(17);font.setBold(True);p.setFont(font)
        origin,tips=axis_points(self.matrix,self.tool);start=project(self.view,origin);self.label_rects=[]
        p.setPen(G.QColor('#e8eff6'));p.drawText(C.QPointF(12,24),self.label+' · '+('夹爪自身方向' if self.tool else '机器人方向'))
        font.setPixelSize(14);p.setFont(font)
        p.drawText(C.QPointF(12,46),'红 X　绿 Y　蓝 Z　｜　箭头指向正方向')
        font.setPixelSize(19);p.setFont(font)
        if start is None or not self.rect().contains(start.toPoint()):p.end();return
        p.setPen(G.QPen(G.QColor('#ffffff'),2));p.setBrush(G.QColor('#ffd166'));p.drawEllipse(start,4,4)
        for letter,color,tip in zip('XYZ',COLORS,tips):
            end=project(self.view,tip)
            if end is None:continue
            p.setPen(G.QPen(G.QColor(color),3));p.drawLine(start,end)
            delta=np.array([end.x()-start.x(),end.y()-start.y()]);norm=np.linalg.norm(delta)
            direction=delta/norm if norm>.2 else np.array([0.,-1.]);normal=np.array([-direction[1],direction[0]])
            if norm>5:
                a=np.array([end.x(),end.y()])-direction*9
                p.setBrush(G.QColor(color));p.drawPolygon(G.QPolygonF([end,C.QPointF(*(a+normal*4)),C.QPointF(*(a-normal*4))]))
            anchor=np.array([end.x(),end.y()])+direction*17
            for dy in (0,-26,26,-52,52):
                rect=C.QRectF(float(np.clip(anchor[0]-12,0,max(0,self.width()-25))),float(np.clip(anchor[1]-12+dy,50,max(50,self.height()-26))),25,26)
                if not any(rect.intersects(other) for other in self.label_rects):break
            self.label_rects.append(rect);p.setPen(G.QPen(G.QColor(color),1));p.setBrush(G.QColor(13,24,32,230));p.drawRoundedRect(rect,4,4)
            p.setPen(G.QColor(color));p.drawText(rect,C.Qt.AlignCenter,letter)
        p.end()
