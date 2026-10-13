"""Parameter fields require an intentional edit, never a scroll gesture."""
from PyQt5 import QtWidgets as Q


class NoWheelDoubleSpinBox(Q.QDoubleSpinBox):
    def wheelEvent(self,event):
        # Ignore even while focused; let the containing page handle scrolling.
        event.ignore()


class NoWheelComboBox(Q.QComboBox):
    def wheelEvent(self,event):
        event.ignore()
