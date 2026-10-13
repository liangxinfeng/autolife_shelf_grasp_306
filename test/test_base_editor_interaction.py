import os
os.environ['QT_QPA_PLATFORM']='offscreen'
from PyQt5 import QtCore as C, QtWidgets as Q
from PyQt5.QtTest import QTest
from autolife_shelf_grasp_306.base_sequence_ui import BaseEditor


def click_button(dialog, text):
    button=next(b for b in dialog.findChildren(Q.QPushButton) if b.text()==text)
    QTest.mouseClick(button, C.Qt.LeftButton)
    Q.QApplication.processEvents()


def enter(spin, text):
    assert spin.isEnabled()
    field=spin.lineEdit()
    field.setFocus(); field.selectAll()
    QTest.keyClicks(field, text)
    QTest.keyClick(field, C.Qt.Key_Tab)
    Q.QApplication.processEvents()


def test_unchecked_sequence_editable_by_keyboard_and_mouse():
    app=Q.QApplication.instance() or Q.QApplication([])
    d=BaseEditor(None, {})
    d.show(); app.processEvents()
    try:
        assert not d.enabled.isChecked()
        click_button(d, '添加')
        combo=d.table.cellWidget(0,0)
        assert combo.isEnabled()
        combo.setFocus(); QTest.keyClick(combo,C.Qt.Key_End)
        assert combo.currentData()=='turn_right'
        enter(d.table.cellWidget(0,1),'90')
        enter(d.table.cellWidget(0,2),'18')
        expected=dict(kind='turn_right',amount=90.,speed=18.)
        assert d.record()==dict(enabled=False,moves=[expected])
        for checked in (True,False):
            QTest.mouseClick(d.enabled,C.Qt.LeftButton,pos=C.QPoint(8,d.enabled.height()//2))
            assert d.enabled.isChecked()==checked
            assert combo.isEnabled() and d.table.cellWidget(0,1).isEnabled()
            assert d.record()['moves']==[expected]
        click_button(d,'添加')
        click_button(d,'上移')
        assert d.record()['moves'][1]==expected
        click_button(d,'删除')
        assert d.record()['moves']==[expected]
        buttons=d.findChild(Q.QDialogButtonBox)
        QTest.mouseClick(buttons.button(Q.QDialogButtonBox.Ok),C.Qt.LeftButton)
        assert d.result()==Q.QDialog.Accepted
        saved=d.record()
        reopened=BaseEditor(None,saved)
        try:
            assert reopened.record()==saved
            assert reopened.table.cellWidget(0,0).isEnabled()
            assert not reopened.enabled.isChecked()
        finally: reopened.close()
    finally: d.close(); app.processEvents()
