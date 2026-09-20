"""Compact parameter dialog; editing never starts robot motion."""
from PyQt5 import QtWidgets as Q
from .demo_config import load,save,LIMITS


def edit_settings(owner):
    try:values=load(owner.root)
    except Exception as exc:Q.QMessageBox.warning(owner,'演示参数',str(exc));return
    dialog=Q.QDialog(owner);dialog.setWindowTitle('演示模式参数');form=Q.QFormLayout(dialog);fields={}
    labels=[('table_height_m','桌面参考高度（cm）',100),
            ('height_search_m','桌高识别范围（±cm）',100),('edge_margin_m','首选桌边余量（cm）',100),
            ('minimum_edge_margin_m','最小桌边余量（cm，不含瓶底半径）',100),
            ('approach_height_m','桌面上方接近余量（cm）',100),('place_speed_m_s','放下饮料速度（cm/s）',100)]
    for key,label,scale in labels:
        box=Q.QDoubleSpinBox();lo,hi=LIMITS[key];box.setRange(lo*scale,hi*scale)
        box.setDecimals(2 if scale==1 and key!='turn_deg' else 1);box.setValue(values[key]*scale)
        form.addRow(label,box);fields[key]=(box,scale)
    hint=Q.QLabel('停稳后自动识别桌面并选点；只支持饮料瓶/罐。\n参考高度不是盲放高度。未可靠识别或遇障时停止，保持饮料。')
    hint.setWordWrap(True);form.addRow(hint)
    buttons=Q.QDialogButtonBox(Q.QDialogButtonBox.Save|Q.QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject);form.addRow(buttons)
    if dialog.exec_()==Q.QDialog.Accepted:
        values.update({k:box.value()/scale for k,(box,scale) in fields.items()});save(owner.root,values)
