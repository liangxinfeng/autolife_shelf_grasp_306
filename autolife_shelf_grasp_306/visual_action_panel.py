"""姿态页的可折叠参照面板；沿用主程序配色与点击图像交互。"""
from PyQt5 import QtCore as C,QtGui as G,QtWidgets as Q
from .visual_reference import ReferenceStore


class VisualPanel(Q.QWidget):
    def __init__(self,workspace):
        super().__init__();self.workspace=workspace;self.owner=workspace.owner;self.token=None;self.adding_view=None;self.preview_token=None
        self.store=ReferenceStore(workspace.studio.library.directory)
        outer=Q.QVBoxLayout(self);outer.setContentsMargins(0,0,0,0)
        toggle=Q.QToolButton();toggle.setText('物体参照 · 可选：让动作适应位置变化');toggle.setCheckable(True);toggle.setToolButtonStyle(C.Qt.ToolButtonTextBesideIcon);outer.addWidget(toggle)
        body=Q.QWidget();self.body=body;self.toggle=toggle;outer.addWidget(body);body.hide();toggle.toggled.connect(body.setVisible);p=Q.QVBoxLayout(body);p.setContentsMargins(0,4,0,4)
        self.name=Q.QLineEdit();self.name.setPlaceholderText('参照名称，如：洗衣机把手区域（不需要分类）')
        p.addWidget(self.name)
        from .visual_box_view import BoxImageView
        self.image=BoxImageView();self.image.setMinimumSize(300,180);self.image.setMaximumHeight(230);self.image.boxSelected.connect(self.preview_box);p.addWidget(self.image)
        row=Q.QHBoxLayout();p.addLayout(row)
        freeze=Q.QPushButton('取景 → 框选新物体');freeze.clicked.connect(lambda:self.guard(self.freeze));row.addWidget(freeze)
        refresh=Q.QPushButton('刷新列表');refresh.clicked.connect(lambda:self.guard(self.refresh));row.addWidget(refresh)
        self.save_view_button=Q.QPushButton('确认轮廓并保存');self.save_view_button.setEnabled(False);self.save_view_button.clicked.connect(lambda:self.guard(self.commit_view));p.addWidget(self.save_view_button)
        self.reference=Q.QComboBox();p.addWidget(self.reference)
        self.gallery_button=Q.QPushButton('图片管理 · 选择物体查看');self.gallery_button.clicked.connect(lambda:self.guard(self.open_gallery));p.addWidget(self.gallery_button)
        row=Q.QHBoxLayout();p.addLayout(row)
        more=Q.QPushButton('给所选编号补录');more.clicked.connect(lambda:self.guard(lambda:self.freeze(add_view=True)));row.addWidget(more)
        remove=Q.QPushButton('删除所选物体');remove.clicked.connect(lambda:self.guard(self.remove));row.addWidget(remove)
        row=Q.QHBoxLayout();p.addLayout(row);self.side=Q.QComboBox();self.side.addItem('关联右手','right');self.side.addItem('关联左手','left');row.addWidget(self.side)
        self.mode=Q.QComboBox();self.mode.addItem('只纠正位置','position');self.mode.addItem('位置＋朝向','rigid');row.addWidget(self.mode)
        self.note=Q.QLabel('保存姿态仅关联所选物体，以初始帧位姿为参照，不需要物体出现在画面中。预览或执行时再识别纠偏；补录只丰富外观，不改变初始参照。');self.note.setWordWrap(True);p.addWidget(self.note)
        self.reference.currentIndexChanged.connect(self.caption);self.side.currentIndexChanged.connect(self.caption);self.mode.currentIndexChanged.connect(self.caption)
        self.refresh()
    def caption(self,*_):
        name=self.reference.currentText() if self.reference.currentData() else '不关联物体'
        self.toggle.setText('物体参照 · '+(name[:18]+'…' if len(name)>18 else name))
        self.toggle.setToolTip(name+'\n展开拍照、点选和配置关联手臂；收起后仍保留当前选择')
        ident=self.reference.currentData();self.gallery_button.setEnabled(bool(ident))
        try:count=len(self.store.view_ids(ident)) if ident else 0
        except (OSError,ValueError,KeyError,TypeError):count=0
        self.gallery_button.setText(f'图片管理 · {count} 帧（含初始帧）' if ident else '图片管理 · 请先选择物体')
    def guard(self,fn):
        try:fn()
        except Exception as exc:self.note.setText(str(exc))
    def refresh(self):
        selected=self.reference.currentData();self.reference.clear();self.reference.addItem('不关联物体（普通姿态）',None)
        for item in self.store.list():self.reference.addItem(item['name']+' · '+item['id'][:6],item['id'])
        self.reference.setCurrentIndex(max(0,self.reference.findData(selected)))
        self.caption()
        if self.store.errors:self.note.setText(f'有{len(self.store.errors)}个参照文件损坏，未加入列表；普通姿态和抓取不受影响。')
    def open_gallery(self):
        ident=self.reference.currentData()
        if not ident:raise ValueError('请先选择要查看的物体')
        if self.workspace.studio.visual.busy:raise ValueError('请等待当前视觉操作完成后打开图片管理')
        # 管理窗口显示的是已存帧，不能误当作刚拍摄的可点击画面。
        self.token=None;self.adding_view=None;self.preview_token=None;self.save_ready(False);self.image.selection_enabled=False
        from .visual_reference_gallery import ReferenceGallery
        dialog=ReferenceGallery(self,ident);dialog.exec_()
        self.refresh();self.caption()
        if dialog.changed:self.note.setText('视角列表已更新，删除的补录帧不再参与识别。第一帧位姿及已有动作未改动。')
        dialog.deleteLater()

    def remove(self):
        self.workspace.available();ident=self.reference.currentData()
        if not ident:raise ValueError('请先选择要删除的物体')
        if Q.QMessageBox.question(self,'删除物体记忆','从物体选择列表删除“'+self.reference.currentText()+'”？已保存姿态引用的快照会保留，避免破坏旧动作；未保存的当前选择将清空。')!=Q.QMessageBox.Yes:return
        self.store.remove(ident);self.token=None;self.adding_view=None;self.preview_token=None;self.save_ready(False);self.image.selection_enabled=False;self.refresh();self.reference.setCurrentIndex(0)
        self.note.setText('已从物体列表删除。已有动作引用保留；新姿态可重新记录或选择其他物体。')
    def record_note(self,text):
        self.note.setText(text)
        if getattr(self,'selector_note',None):self.selector_note.setText(text)
    def save_ready(self,ready):
        self.save_view_button.setEnabled(ready)
        if getattr(self,'selector_save',None):self.selector_save.setEnabled(ready)
    def freeze(self,add_view=False):
        self.workspace.available()
        if self.workspace.studio.visual.busy:raise ValueError('请等待当前视觉操作完成')
        ident=self.reference.currentData() if add_view else None
        if add_view and not ident:raise ValueError('先选择要补录的物体')
        if not add_view and not self.name.text().strip():raise ValueError('先填写参照名称')
        self.token=None;self.preview_token=None;self.adding_view=ident;self.save_ready(False)
        self.image.selection_enabled=False;self.image.selection=None
        def ready(data):
            if data.get('error'):self.record_note(data['error']);return
            r=data['result'];self.token=r['token'];self.image.pix=G.QPixmap(r['image']);self.image.shape=r['shape'];self.image.boxes=[];self.image.selection_enabled=True;self.image.update()
            self.record_note('拖动矩形框包住整个物体，松开后查看分割轮廓；不满意可重新拖框，确认后才保存。'+('补录仍归编号 '+ident[:6]+'，初始位姿不变。' if ident else '这次保存后将成为初始帧。'))
            self.open_selector()
        self.workspace.studio.visual.request('visual_freeze',ready)
    def open_selector(self):
        from .visual_box_view import BoxImageView
        dialog=Q.QDialog(self);self.selector=dialog;dialog.setWindowTitle('框选完整物体 · '+('补录 '+self.adding_view[:6] if self.adding_view else '新建初始帧'));dialog.resize(980,740)
        layout=Q.QVBoxLayout(dialog);self.selector_view=BoxImageView();view=self.selector_view
        view.pix=self.image.pix;view.shape=self.image.shape;view.selection_enabled=True;view.boxSelected.connect(self.preview_box);layout.addWidget(view,1)
        self.selector_note=Q.QLabel(self.note.text());self.selector_note.setWordWrap(True);layout.addWidget(self.selector_note)
        row=Q.QHBoxLayout();layout.addLayout(row);self.selector_save=Q.QPushButton('确认轮廓并保存');self.selector_save.setEnabled(False);self.selector_save.clicked.connect(lambda:self.guard(self.commit_view));row.addWidget(self.selector_save)
        close=Q.QPushButton('关闭 / 暂不保存');close.clicked.connect(dialog.reject);row.addWidget(close)
        dialog.exec_()
        self.selector=None;self.selector_view=None;self.selector_note=None;self.selector_save=None;dialog.deleteLater()
    def preview_box(self,box):
        def act():
            self.workspace.available()
            if self.workspace.studio.visual.busy:raise ValueError('正在分割，请稍候再框选')
            if not self.token:raise ValueError('请先取景')
            self.preview_token=None;self.save_ready(False);self.record_note('正在分割框内物体，尚未保存…')
            def ready(data):
                if data.get('error'):self.record_note(data['error']+'；可在本帧重新拖框。');return
                r=data['result'];self.preview_token=r['preview_token'];self.image.boxes=[r['item']];self.image.selected=0;self.image.update()
                if getattr(self,'selector_view',None):
                    self.selector_view.boxes=[r['item']];self.selector_view.selected=0;self.selector_view.update()
                self.save_ready(True);self.record_note('彩色轮廓是将保存的物体区域。请检查是否覆盖整个物体；不满意可重新拖框，正确后点击“确认轮廓并保存”。')
            self.workspace.studio.visual.request('visual_add_view' if self.adding_view else 'visual_remember',ready,token=self.token,box=box,preview_only=True,name=self.name.text().strip(),reference=self.adding_view)
        try:act()
        except Exception as exc:self.record_note(str(exc))
    def commit_view(self):
        self.workspace.available()
        if not self.preview_token:raise ValueError('请先框选并生成分割预览')
        token=self.preview_token;self.save_ready(False)
        def ready(data):
            if data.get('error'):self.record_note(data['error']);self.save_ready(True);return
            r=data['result'];self.token=None;self.preview_token=None;self.image.selection_enabled=False
            self.refresh();self.reference.setCurrentIndex(self.reference.findData(r['reference']));self.caption()
            text=('已补录至原编号 '+r['reference'][:6]+'，初始位姿不变。' if r.get('supplement') else '已保存初始帧，编号 '+r['reference'][:6]+'。')
            if not r.get('metric_registered',True):text+='补录已参与识别；“只纠正位置”可用整体轮廓和深度定位，不必补录过渡面。'
            self.record_note(text+'可在“图片管理”中查看或删除补录帧。')
            if getattr(self,'selector',None):self.selector.accept()
        self.workspace.studio.visual.request('visual_commit_view',ready,preview_token=token)
    def bind(self,q,callback):
        """保存只关联已有初始参照；无取景、识别、视觉进程或运动请求。"""
        self.workspace.available()
        ident=self.reference.currentData()
        if not ident:callback(None);return
        from .pose_geometry import ModelGeometry
        from .visual_reference import make_binding
        geometry=ModelGeometry(self.workspace.studio.cfg['urdf_path'])
        bound=make_binding(self.store,ident,self.side.currentData(),self.mode.currentData(),q,geometry)
        callback(bound)
    def select_binding(self,bound):
        self.refresh();self.reference.setCurrentIndex(max(0,self.reference.findData(bound['reference'] if bound else None)))
        if bound:
            self.side.setCurrentIndex(self.side.findData(bound['side']));self.mode.setCurrentIndex(self.mode.findData(bound['mode']))
    def show_correction(self,result):
        self.token=None;self.preview_token=None;self.save_ready(False);self.image.selection_enabled=False;self.image.selection=None
        self.image.pix=G.QPixmap(result['image']);self.image.shape=result['shape'];boxes=[]
        for i,item in enumerate(result['diagnostics']):
            points=item['pixels'];xs=[p[0] for p in points];ys=[p[1] for p in points]
            boxes.append(dict(id=i,name=item['name']+' · 已识别',box=item.get('object_box',[min(xs),min(ys),max(xs),max(ys)]),mask_contours=item.get('mask_contours',[])))
        self.image.boxes=boxes;self.image.update()
        modes=[d.get('method_label','纹理深度定位') for d in result['diagnostics']]
        text='；'.join(dict.fromkeys(modes))
        if any(d.get('recognition_method')=='dinov2_object' for d in result['diagnostics']):
            text='整物体外观记忆识别 → '+text
        if any(d.get('method')=='shape_position' for d in result['diagnostics']):text+='。只调整位置，夹爪朝向保持示教值；请核对模型预览。'
        self.note.setText(text+' XYZ纠偏量见页面顶部；每次真机运行重新定位，原始动作保持不变。')
    def show_image(self):
        if self.image.pix.isNull():raise ValueError('请先记录物体或点击“定位参照并预览纠偏”')
        from .app import ImageView
        dialog=Q.QDialog(self);dialog.setWindowTitle('物体参照 · 本次画面');dialog.resize(840,660)
        layout=Q.QVBoxLayout(dialog);view=ImageView();view.pix=self.image.pix;view.shape=self.image.shape;view.boxes=self.image.boxes;layout.addWidget(view,1)
        label=Q.QLabel(self.workspace.notice.text());label.setWordWrap(True);layout.addWidget(label)
        close=Q.QPushButton('关闭');close.clicked.connect(dialog.accept);layout.addWidget(close);dialog.exec_()
