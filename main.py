# coding: utf-8
from PyQt5.QtCore import Qt, QUrl, QSize, QTimer
from PyQt5.QtGui import QIcon, QDesktopServices
from PyQt5.QtWidgets import QApplication

from qfluentwidgets import (
    NavigationAvatarWidget,
    NavigationItemPosition,
    MessageBox,
    FluentWindow,
    SplashScreen,
)
from qfluentwidgets import FluentIcon as FIF
import interface
from interface import icon as Icon
import interface.s2nninterface as s2nn
import interface.camerainterface as camerainterface
import interface.eventinterface as eventinterface
import interface.imageInterface as imageInterface
from interface import RoundShadow
from PyQt5.QtGui import QPainter, QColor, QBrush
from PyQt5.QtCore import Qt, QRectF


class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()
        self.initWindow()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.finishSplashScreen)
        self.homeInterface = s2nn.s2nninterface(self)
        self.cameraInterface = camerainterface.CameraWidget(self)
        self.eventInterface = eventinterface.Eventinterface(self)
        self.imageInterface = imageInterface.ImageProcessorInterface(self)

        # enable acrylic effect
        self.navigationInterface.setAcrylicEnabled(True)

        # add items to navigation interface
        self.initNavigation()
        self.timer.start(1000)

    def initNavigation(self):
        t = interface.Translator()
        self.addSubInterface(self.homeInterface, FIF.HOME, self.tr("Home"))
        self.addSubInterface(self.cameraInterface, FIF.CAMERA, self.tr("Camera"))
        self.addSubInterface(self.eventInterface, FIF.STOP_WATCH, self.tr("Event"))
        # self.addSubInterface(self.iconInterface, Icon.EMOJI_TAB_SYMBOLS, t.icons)
        self.navigationInterface.addSeparator()
        pos = NavigationItemPosition.SCROLL
        self.addSubInterface(
            self.imageInterface, FIF.ALBUM, self.tr("Image"), position=pos
        )

    def initWindow(self):
        self.resize(1200, 780)
        self.setMinimumWidth(1100)
        self.setWindowTitle("Control GUI")
        self.setWindowIcon(QIcon("resource/place_holder.jpeg"))
        self.setMicaEffectEnabled(True)
        # create splash screen

        self.splashScreen = SplashScreen(self.windowIcon(), self)
        self.splashScreen.setIconSize(QSize(106, 106))
        self.splashScreen.raise_()

        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)
        self.show()
        QApplication.processEvents()

    def finishSplashScreen(self):
        self.splashScreen.finish()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, "splashScreen"):
            self.splashScreen.resize(self.size())


if __name__ == "__main__":
    import sys

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)

    window = MainWindow()
    sys.exit(app.exec_())
