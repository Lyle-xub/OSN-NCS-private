from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout

from qfluentwidgets import InfoBarIcon, InfoBar, PushButton, setTheme, Theme, FluentIcon, InfoBarPosition, InfoBarManager


def createSuccessInfoBar(title, content, parent=None):
    # convenient class mothod
    InfoBar.success(
        title=title,
        content=content,
        orient=Qt.Horizontal,
        isClosable=True,
        position=InfoBarPosition.TOP,
        # position='Custom',   # NOTE: use custom info bar manager
        duration=2000,
        parent=parent
    )
    
def createWarningInfoBar(title, content, parent=None):
    InfoBar.warning(
        title=title,
        content=content,
        orient=Qt.Horizontal,
        isClosable=False,   # disable close button
        position=InfoBarPosition.TOP_LEFT,
        duration=2000,
        parent=parent
    )

def createErrorInfoBar(title, content, parent=None):
    InfoBar.error(
        title=title,
        content=content,
        orient=Qt.Horizontal,
        isClosable=True,
        position=InfoBarPosition.BOTTOM_RIGHT,
        duration=-1,    # won't disappear automatically
        parent=parent
    )