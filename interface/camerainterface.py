import sys
from PyQt5.QtCore import Qt, QSize, QThread
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, pyqtSignal
import cv2
from DCAM.dcamapi4 import *
from DCAM.dcam import Dcamapi, Dcam
from PyQt5.QtGui import QColor
from qfluentwidgets import TransparentTogglePushButton, TogglePushButton
from qfluentwidgets import ProgressBar, TitleLabel
from qfluentwidgets.components.widgets.acrylic_label import AcrylicLabel

import numpy as np
from function.cmos_func import *
from DCAM import *
import interface


class CameraInitThread(QThread):
    frame_data_signal = pyqtSignal(np.ndarray)
    errorOccurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        try:
            dcam = init_camera()
            setup_camera_for_external_trigger(dcam)
            capture_image_on_trigger(dcam, self.frame_data_signal.emit, self)
            dcam.cap_stop()
            dcam.buf_release()
            dcam.dev_close()
            Dcamapi.uninit()
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            self.running = False

    def stop(self):
        self.running = False


class CameraLiveThread(QThread):
    frame_data_signal = pyqtSignal(np.ndarray)
    errorOccurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = False

    def run(self):
        self.running = True
        try:
            dcam_live_capturing(self, callback=self.frame_data_signal.emit)
        except Exception as e:
            print(f"Error occurred: {e}")
            self.errorOccurred.emit(str(e))
        finally:
            self.running = False

    def stop(self):
        self.running = False


class CameraWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CameraWidget")
        self.dcam = None
        self.timer = QTimer(self)
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(36, 22, 36, 12)

        self.titleLabel = TitleLabel("CMOS", self)
        self.imagelayout = QVBoxLayout()
        self.buttonlayout = QHBoxLayout()

        self.init_button = TogglePushButton("Trigger Camera")
        self.init_button.setFixedSize(QSize(200, 30))
        # keep the button on the top right corner
        self.live_button = TogglePushButton("Live Camera")
        self.live_button.setFixedSize(QSize(200, 30))

        self.buttonlayout.addWidget(self.init_button, 0, Qt.AlignLeft)
        self.buttonlayout.addWidget(self.live_button, 0, Qt.AlignLeft)
        self.buttonlayout.addStretch()

        self.image_label = QLabel("waiting for camera initialization...")
        Photo = QPixmap("resource/place_holder.jpeg")
        self.image_label.setPixmap(Photo)
        self.image_label.setFixedSize(700, 700)

        self.init_button.clicked.connect(self.initCameraInThread)
        self.live_button.clicked.connect(self.initCameraLiveThread)
        self.buttonlayout.addSpacing(10)
        self.layout.addWidget(self.titleLabel)
        self.imagelayout.addLayout(self.buttonlayout)
        self.imagelayout.addWidget(self.image_label)
        self.layout.addLayout(self.imagelayout)

        self.layout.addStretch(1)

    def initCameraInThread(self):
        if self.init_button.isChecked():
            self.camera_thread = CameraInitThread()
            self.camera_thread.frame_data_signal.connect(self.updateImage)
            self.camera_thread.errorOccurred.connect(self.showError)
            self.camera_thread.start()
        else:
            self.timer.stop()
            self.image_label.setPixmap(QPixmap("resource/place_holder.jpeg"))
            self.camera_thread.stop()
            self.camera_thread.wait()

    def initCameraLiveThread(self):
        if self.live_button.isChecked():
            self.camera_thread = CameraLiveThread()
            self.camera_thread.frame_data_signal.connect(self.updateImage)
            self.camera_thread.errorOccurred.connect(self.showError)
            self.camera_thread.start()
        else:
            self.timer.stop()
            self.image_label.setPixmap(QPixmap("resource/place_holder.jpeg"))
            self.camera_thread.stop()
            self.camera_thread.wait()

    def updateImage(self, frame_data):
        height, width = frame_data.shape
        frame_scaled = cv2.convertScaleAbs(frame_data, alpha=(255.0 / 65535.0))
        bytes_per_line = width

        q_image = QImage(
            frame_scaled.data, width, height, bytes_per_line, QImage.Format_Grayscale8
        )
        pixmap = QPixmap.fromImage(q_image)

        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)

    def showError(self, error):
        interface.createErrorInfoBar("Error", error, self)

    def closeEvent(self, event):
        self.timer.stop()
        if self.dcam:
            self.dcam.cap_stop()
            self.dcam.buf_release()
            self.dcam.dev_close()
            Dcamapi.uninit()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    camera_widget = CameraWidget()
    camera_widget.show()
    sys.exit(app.exec_())
