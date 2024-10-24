# coding:utf-8
from PyQt5.QtCore import Qt, pyqtSignal, QUrl, QEvent
from PyQt5.QtGui import QDesktopServices, QPainter, QPen, QColor
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame

from qfluentwidgets import (ScrollArea, PushButton, ToolButton, FluentIcon,
                            isDarkTheme, IconWidget, Theme, ToolTipFilter, TitleLabel, CaptionLabel,
                            StrongBodyLabel, BodyLabel, toggleTheme)



class SeparatorWidget(QWidget):
    """ Seperator widget """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setFixedSize(6, 16)

    def paintEvent(self, e):
        painter = QPainter(self)
        pen = QPen(1)
        pen.setCosmetic(True)
        c = QColor(255, 255, 255, 21) if isDarkTheme() else QColor(0, 0, 0, 15)
        pen.setColor(c)
        painter.setPen(pen)

        x = self.width() // 2
        painter.drawLine(x, 0, x, self.height())
        
class ExampleCard(QWidget):
    """ Example card """

    def __init__(self, title, widget: QWidget, stretch=0, parent=None):
        super().__init__(parent=parent)
        self.widget = widget
        self.stretch = stretch

        self.titleLabel = BodyLabel(title, self)
        self.card = QFrame(self)


        self.vBoxLayout = QVBoxLayout(self)
        self.cardLayout = QVBoxLayout(self.card)
        self.topLayout = QHBoxLayout()


        self.__initWidget()

    def __initWidget(self):

        self.__initLayout()
        self.card.setObjectName('card')


    def __initLayout(self):
        self.vBoxLayout.setSizeConstraint(QVBoxLayout.SetMinimumSize)
        self.cardLayout.setSizeConstraint(QVBoxLayout.SetMinimumSize)
        self.topLayout.setSizeConstraint(QHBoxLayout.SetMinimumSize)

        self.vBoxLayout.setSpacing(6)
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.topLayout.setContentsMargins(0, 0, 0, 0)
        self.cardLayout.setContentsMargins(0, 0, 0, 0)

        self.vBoxLayout.addWidget(self.titleLabel, 0, Qt.AlignTop)
        self.vBoxLayout.addWidget(self.card, 0, Qt.AlignTop)
        self.vBoxLayout.setAlignment(Qt.AlignTop)

        self.cardLayout.setSpacing(0)
        self.cardLayout.setAlignment(Qt.AlignTop)
        self.cardLayout.addLayout(self.topLayout, 0)


        self.widget.setParent(self.card)
        self.topLayout.addWidget(self.widget)
        if self.stretch == 0:
            self.topLayout.addStretch(1)

        self.widget.show()





class ToolBar(QWidget):
    """ Tool bar """

    def __init__(self, title, subtitle, parent=None):
        super().__init__(parent=parent)
        self.titleLabel = TitleLabel(title, self)
        self.subtitleLabel = CaptionLabel(subtitle, self)

        self.documentButton = PushButton(
            self.tr('Documentation'), self, FluentIcon.DOCUMENT)
        self.sourceButton = PushButton(self.tr('Source'), self, FluentIcon.GITHUB)
        self.separator = SeparatorWidget(self)
        self.supportButton = ToolButton(FluentIcon.HEART, self)
        self.feedbackButton = ToolButton(FluentIcon.FEEDBACK, self)

        self.vBoxLayout = QVBoxLayout(self)
        self.buttonLayout = QHBoxLayout()

        self.__initWidget()

    def __initWidget(self):
        self.setFixedHeight(138)
        self.vBoxLayout.setSpacing(0)
        self.vBoxLayout.setContentsMargins(36, 22, 36, 12)
        self.vBoxLayout.addWidget(self.titleLabel)
        self.vBoxLayout.addSpacing(4)
        self.vBoxLayout.addWidget(self.subtitleLabel)
        self.vBoxLayout.addSpacing(4)
        self.vBoxLayout.addLayout(self.buttonLayout, 1)
        self.vBoxLayout.setAlignment(Qt.AlignTop)

        self.buttonLayout.setSpacing(4)
        self.buttonLayout.setContentsMargins(0, 0, 0, 0)
        self.buttonLayout.addWidget(self.documentButton, 0, Qt.AlignLeft)
        self.buttonLayout.addWidget(self.sourceButton, 0, Qt.AlignLeft)
        self.buttonLayout.addStretch(1)
        self.buttonLayout.addWidget(self.separator, 0, Qt.AlignRight)
        self.buttonLayout.addWidget(self.supportButton, 0, Qt.AlignRight)
        self.buttonLayout.addWidget(self.feedbackButton, 0, Qt.AlignRight)
        self.buttonLayout.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.subtitleLabel.setTextColor(QColor(96, 96, 96), QColor(216, 216, 216))


