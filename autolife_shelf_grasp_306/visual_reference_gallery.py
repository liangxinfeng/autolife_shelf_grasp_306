"""只读图片浏览与补录视角管理；不创建视觉/ROS/运动进程。"""
from datetime import datetime
import numpy as np
from PyQt5 import QtCore as C,QtGui as G,QtWidgets as Q


def picture(arrays,overlay=True):
    import cv2
    image=np.asarray(arrays['image'],np.uint8).copy()
    mask=np.asarray(arrays.get('mask',np.zeros(image.shape[:2])),bool)
    if overlay and mask.shape==image.shape[:2] and mask.any():
        image[~mask]=(image[~mask].astype(float)*.45).astype(np.uint8)
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image,contours,-1,(190,230,40),2)
    rgb=np.ascontiguousarray(image[:,:,::-1]);h,w=rgb.shape[:2]
    return G.QPixmap.fromImage(G.QImage(rgb.data,w,h,rgb.strides[0],G.QImage.Format_RGB888).copy())


def stamp(meta):
    try:return datetime.fromtimestamp(float(meta['created'])).strftime('%m-%d %H:%M:%S')
    except (ValueError,KeyError,TypeError,OverflowError):return '早期记录'


class CardDelegate(Q.QStyledItemDelegate):
    def paint(self,painter,option,index):
        opt=Q.QStyleOptionViewItem(option);self.initStyleOption(opt,index)
        # 图片/文字由卡片控件绘制；这里仅绘制选择背景，避免底层文字叠印。
        opt.text='';opt.icon=G.QIcon()
        style=opt.widget.style() if opt.widget else Q.QApplication.style()
        style.drawControl(Q.QStyle.CE_ItemViewItem,opt,painter,opt.widget)


class ReferenceGallery(Q.QDialog):
    def __init__(self,panel,ident):
        super().__init__(panel);self.panel=panel;self.store=panel.store;self.ident=ident;self.changed=False
        self.setWindowTitle('物体图片管理');self.resize(1000,660)
        outer=Q.QVBoxLayout(self)
        meta,_=self.store.load(ident)
        self.title=Q.QLabel(meta['name']+' · 编号 '+ident[:6]);self.title.setStyleSheet('font-size:20px;font-weight:bold');outer.addWidget(self.title)
        hint=Q.QLabel('★ 初始帧提供位姿基准，不能单独删除；补录帧丰富外观。选图查看大图，双击可切换原图/轮廓。删除补录帧不会更改初始位姿。');hint.setWordWrap(True);outer.addWidget(hint)
        split=Q.QSplitter();outer.addWidget(split,1)
        self.list=Q.QListWidget();self.list.setItemDelegate(CardDelegate(self.list));self.list.setSpacing(6);self.list.setMinimumWidth(275);self.list.setStyleSheet('QListWidget::item {padding:2px;}');split.addWidget(self.list)
        right=Q.QWidget();layout=Q.QVBoxLayout(right);split.addWidget(right);split.setSizes([290,690])
        self.image=Q.QLabel('请选择图片');self.image.setAlignment(C.Qt.AlignCenter);self.image.setMinimumSize(360,280);self.image.setSizePolicy(Q.QSizePolicy.Ignored,Q.QSizePolicy.Expanding);layout.addWidget(self.image,1)
        self.info=Q.QLabel();self.info.setWordWrap(True);layout.addWidget(self.info)
        row=Q.QHBoxLayout();layout.addLayout(row)
        self.overlay=Q.QCheckBox('显示分割轮廓');self.overlay.setChecked(True);self.overlay.toggled.connect(self.show_selected);row.addWidget(self.overlay)
        self.remove=Q.QPushButton('删除选中补录视角');self.remove.clicked.connect(self.delete_selected);row.addWidget(self.remove)
        self.note=Q.QLabel();self.note.setWordWrap(True);outer.addWidget(self.note)
        row=Q.QHBoxLayout();outer.addLayout(row)
        refresh=Q.QPushButton('刷新');refresh.clicked.connect(self.reload);row.addWidget(refresh)
        close=Q.QPushButton('完成');close.clicked.connect(self.accept);row.addWidget(close)
        self.list.currentItemChanged.connect(self.show_selected)
        self.list.itemDoubleClicked.connect(lambda *_:self.overlay.setChecked(not self.overlay.isChecked()))
        self.pix=G.QPixmap();self.reload()
    def reload(self,*_):
        selected=self.list.currentItem().data(C.Qt.UserRole) if self.list.currentItem() else self.ident
        self.list.blockSignals(True);self.list.clear()
        try:
            for index,child in enumerate(self.store.view_ids(self.ident)):
                initial=child==self.ident
                try:
                    meta,arrays=self.store.load(child);thumb=picture(arrays)
                    state='位姿基准' if initial else '已关联初始几何' if meta.get('metric_registered',True) else '外观已存档'
                    label=('★ 初始帧' if initial else f'补录视角 {index}')+'\n'+state+' · '+stamp(meta)
                    item=Q.QListWidgetItem(G.QIcon(thumb),label)
                except (ValueError,OSError,KeyError,TypeError) as exc:
                    item=Q.QListWidgetItem(('初始帧' if initial else f'补录视角 {index}')+' · 图片读取失败');item.setToolTip(str(exc))
                item.setData(C.Qt.UserRole,child);item.setSizeHint(C.QSize(255,185))
                if initial:item.setForeground(G.QColor('#ffd37a'))
                self.list.addItem(item)
                card=Q.QWidget();card.setStyleSheet('background:transparent');v=Q.QVBoxLayout(card);v.setContentsMargins(4,4,4,4);v.setSpacing(2)
                photo=Q.QLabel();photo.setAlignment(C.Qt.AlignCenter)
                if not item.icon().isNull():photo.setPixmap(item.icon().pixmap(190,100))
                v.addWidget(photo);label=Q.QLabel(item.text());label.setAlignment(C.Qt.AlignCenter);label.setWordWrap(True);label.setStyleSheet('background:transparent;font-size:13px;'+('color:#ffd37a;' if initial else ''));v.addWidget(label)
                card.setAttribute(C.Qt.WA_TransparentForMouseEvents);self.list.setItemWidget(item,card)
            found=next((i for i in range(self.list.count()) if self.list.item(i).data(C.Qt.UserRole)==selected),0)
            if self.list.count():self.list.setCurrentRow(found)
        except Exception as exc:self.note.setText(str(exc))
        finally:self.list.blockSignals(False)
        self.show_selected()
    def show_selected(self,*_):
        item=self.list.currentItem();self.remove.setEnabled(False)
        if not item:return
        child=item.data(C.Qt.UserRole);initial=child==self.ident;self.remove.setEnabled(not initial)
        self.remove.setToolTip('初始帧不能单独删除；删除整个物体请返回主页面操作' if initial else '删除后该视角不再参与识别与纠偏，历史快照仍保留')
        try:
            meta,arrays=self.store.load(child);self.pix=picture(arrays,self.overlay.isChecked())
            pos=np.asarray(self.store.load(self.ident)[0]['object_pose'])[:3,3]*1000
            state='初始帧 · 所有补录共用的位姿基准' if initial else ('补录视角 · 已保存与初始帧的几何关联' if meta.get('metric_registered',True) else '补录视角 · 已用于识别；位置纠偏可结合初始完整轮廓与深度，不要求过渡视角')
            self.info.setText(state+'\n拍摄时间：'+stamp(meta)+'\n初始参照 XYZ：'+', '.join(f'{v:.1f}' for v in pos)+' mm'+ '\n图片编号：'+child[:12])
            self.paint_image()
        except Exception as exc:self.pix=G.QPixmap();self.image.setText('图片无法读取');self.info.setText(str(exc))
    def paint_image(self):
        if not self.pix.isNull():self.image.setPixmap(self.pix.scaled(self.image.size(),C.Qt.KeepAspectRatio,C.Qt.SmoothTransformation))
    def resizeEvent(self,event):
        super().resizeEvent(event)
        if hasattr(self,'pix'):self.paint_image()
    def delete_selected(self):
        item=self.list.currentItem()
        if not item:return
        try:
            self.panel.workspace.available()
            if self.panel.workspace.studio.visual.busy:raise ValueError('请等待当前视觉操作完成后删除视角')
            child=item.data(C.Qt.UserRole)
            if child==self.ident:raise ValueError('初始帧不能单独删除')
            if Q.QMessageBox.question(self,'删除补录视角','删除这张补录图片？删除后不再参与识别；初始帧、其他视角和已保存动作保留。',Q.QMessageBox.Yes|Q.QMessageBox.No,Q.QMessageBox.No)!=Q.QMessageBox.Yes:return
            self.store.remove_view(self.ident,child);self.changed=True;self.panel.token=None;self.panel.adding_view=None
            if hasattr(self.panel,'preview_token'):self.panel.preview_token=None;self.panel.save_ready(False)
            self.reload();self.note.setText('已删除选中视角，不再用于后续识别。初始位姿和已有动作保持不变。')
        except Exception as exc:self.note.setText(str(exc))
