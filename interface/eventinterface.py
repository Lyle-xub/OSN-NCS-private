import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
import argparse
import cv2
import os
import time
import numpy as np
from skvideo.io import FFmpegWriter

from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator
from metavision_sdk_core import (
    PeriodicFrameGenerationAlgorithm,
    ColorPalette,
    OnDemandFrameGenerationAlgorithm,
    BaseFrameGenerationAlgorithm,
)
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
from function.utils import get_biases_from_file
from function.utils import get_roi_from_file
from function.event_func import *
import metavision_hal

from qfluentwidgets import TransparentTogglePushButton, TogglePushButton, TitleLabel
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, pyqtSignal, QThread, Qt, QSize

import interface


class eventThread(QThread):
    new_frame = pyqtSignal(np.ndarray)
    errorOccurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def run(self):
        self.running = True
        try:
            device = initialize_camera()
            play_and_record(device, "checkpoint/event/out.raw", self.on_new_frame, self)
        except Exception as e:
            print(f"Error occurred: {e}")
            self.errorOccurred.emit(str(e))
        finally:
            self.running = False

    def stop(self):
        self.running = False

    def on_new_frame(self, ts, frame):
        self.new_frame.emit(frame)


class Eventinterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("Eventinterface")
        self.initUI()

    def initUI(self):
        # Button to initialize the camera\
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(36, 22, 36, 12)
        self.titleLabel = TitleLabel("Event Camere", self)
        self.init_btn = TogglePushButton("Initialize Camera", self)
        self.init_btn.setFixedSize(QSize(200, 30))
        self.init_btn.clicked.connect(self.initEventThread)

        # Image display
        self.image_label = QLabel(self)
        self.image_label.setPixmap(QPixmap("resource/place_holder.jpeg"))
        self.image_label.resize(640, 480)  # Adjust size as needed
        self.layout.addWidget(self.titleLabel)
        self.layout.addWidget(self.init_btn)
        self.layout.addWidget(self.image_label)
        self.layout.addStretch()

    def initEventThread(self):
        if self.init_btn.isChecked():
            self.eventThread = eventThread()
            self.eventThread.new_frame.connect(self.display_image)  # 连接信号
            self.eventThread.errorOccurred.connect(self.showError)
            self.eventThread.start()
        else:
            self.eventThread.quit()
            self.eventThread.wait()

    def display_image(self, image_np):
        # print(image_np.shape)
        height, width, channel = image_np.shape
        bytesPerLine = 3 * width
        q_img = QImage(image_np.data, width, height, bytesPerLine, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)

    def showError(self, error):
        interface.createErrorInfoBar("EVENT", error, self)
