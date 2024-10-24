from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys


class RoundShadow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.border_width = 8
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)

    def paintEvent(self, event):
        # 阴影
        # path = QPainterPath()
        # path.setFillRule(Qt.WindingFill)

        # pat = QPainter(self)
        # pat.setRenderHint(pat.Antialiasing)
        # pat.fillPath(path, QBrush(Qt.white))

        # color = QColor(192, 192, 192, 50)

        # for i in range(10):
        #     i_path = QPainterPath()
        #     i_path.setFillRule(Qt.WindingFill)
        #     ref = QRectF(10-i, 10-i, self.width()-(10-i)*2, self.height()-(10-i)*2)
        #     # i_path.addRect(ref)
        #     i_path.addRoundedRect(ref, self.border_width, self.border_width)
        #     color.setAlpha(10)
        #     pat.setPen(color)
        #     pat.drawPath(i_path)

        # 圆角
        pat2 = QPainter(self)
        pat2.setRenderHint(pat2.Antialiasing)
        pat2.setBrush(Qt.white)
        pat2.setPen(Qt.transparent)

        rect = self.rect()
        rect.setLeft(9)
        rect.setTop(9)
        rect.setWidth(rect.width() - 9)
        rect.setHeight(rect.height() - 9)
        pat2.drawRoundedRect(rect, 4, 4)


class TestWindow(RoundShadow, QWidget):
    """测试窗口"""

    def __init__(self, parent=None):
        super(TestWindow, self).__init__(parent)

        self.resize(300, 300)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    t = TestWindow()
    # t = RoundImage('./Asset/new_icons/close.png')
    t.show()
    app.exec_()
