import sys
import os
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QDesktopServices, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QSize, QThread
from PyQt5.QtCore import pyqtSignal

# from PyQt5.Qtcore import QThread
from PyQt5.QtWidgets import (
    QApplication,
    QSplitter,
    QWidget,
    QVBoxLayout,
    QAction,
    QGridLayout,
)
from qfluentwidgets import TransparentTogglePushButton, TogglePushButton
from qfluentwidgets import ProgressBar
from qfluentwidgets import FluentIcon as FIF
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize
from omegaconf import OmegaConf
from qfluentwidgets import ProgressBar, FluentIcon
from qfluentwidgets import (
    Action,
    FluentIcon,
    TitleLabel,
    CaptionLabel,
    toggleTheme,
    RoundMenu,
    SplitPushButton,
    SpinBox,
)


import function
from function import (
    reconstruction_pysical_trainthread_with_cmos,
    # reconstruction_pysical_trainthread_with_event,
)

from function import cfg
from ctypes import *

import torch, torchvision
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
import lightridge.data as dataset
from interface import ToolBar, ImageDisplayWidget, SeparatorWidget, ExampleCard
import interface

from qfluentwidgets import TextEdit


class s2nninterface(QWidget):
    """
    The s2nninterface class represents the user interface for controlling the s2nn system.

    Attributes:
        camera (None): Placeholder for the camera object.
        value (int): Placeholder for a value.
        DMD_flag (None): Placeholder for the DMD flag.
        SlmFlag (None): Placeholder for the SLM flag.
        train (bool): Flag indicating whether training is enabled.

    Methods:
        __init__(self, parent=None): Initializes the s2nninterface object.
        _create_menu(self): Creates the menus for DMD, SLM, and camera options.
        _create_components(self): Creates the UI components for the s2nninterface.
        _setup_layout(self): Sets up the layout for the s2nninterface.
        _setup_connections(self): Sets up the signal-slot connections for the UI components.
    """

    camera = None
    value = 1
    DMD_flag = None
    SlmFlag = None
    train = True
    insitu_train = False
    error_train = False
    calibration = False
    model = "lightridge"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("s2nninterface")
        self._create_menu()
        self._create_components()
        self._setup_layout()
        self._init_training_thread()
        self._setup_connections()

    def _create_menu(self):
        self.DMD_menu = RoundMenu(parent=self)
        for option in ["DMD Off", "Loop Mode", "Trigger Mode"]:
            DMD_action = Action(option, self)
            DMD_action.triggered.connect(lambda checked, text=option: self.DMD_on(text))
            self.DMD_menu.addAction(DMD_action)

        self.SLM_menu = RoundMenu(parent=self)
        for option in ["SLM Off", "Loop Mode", "Trigger Mode"]:
            slm_action = Action(option, self)
            slm_action.triggered.connect(lambda checked, text=option: self.SLM_on(text))
            self.SLM_menu.addAction(slm_action)

        self.menu = RoundMenu(parent=self)
        for option in ["NONE", "CMOS", "EVENT"]:
            action = Action(option, self)
            action.triggered.connect(
                lambda checked, text=option: self.on_menu_item_selected(text)
            )
            self.menu.addAction(action)

        self.model_menu = RoundMenu(parent=self)
        for option in ["lightridge", "reconstruction"]:
            action = Action(option, self)
            action.triggered.connect(
                lambda checked, text=option: self.model_select(text)
            )
            self.model_menu.addAction(action)

    def _create_components(self):
        self.titleLabel = TitleLabel("S2NN", self)
        self.subtitleLabel = CaptionLabel("real-time train", self)

        self.DMD_button = SplitPushButton(f"DMD", self)
        self.DMD_button.setFlyout(self.DMD_menu)

        self.SLM_button = SplitPushButton(f"SLM", self)
        self.SLM_button.setFlyout(self.SLM_menu)

        self.EVENT_button = SplitPushButton("Select Camera", self)
        self.EVENT_button.setFlyout(self.menu)

        self.train_button = TogglePushButton(FluentIcon.PAUSE, f"TRAIN", self)
        self.train_insitu_button = TogglePushButton(FluentIcon.PAUSE, f"INSITU", self)
        self.train_button.setFixedSize(QSize(100, 30))
        self.train_insitu_button.setFixedSize(QSize(100, 30))

        self.spinBox = SpinBox(self)
        self.spinBox.setAccelerated(True)
        self.spinBox.setFixedSize(QSize(120, 30))
        self.spinBox.setRange(1, cfg.depth)

        self.progressBar = ProgressBar(self)
        self.trainingProgressBar = ProgressBar(self)
        self.progresscard = ExampleCard("Total Process", self.progressBar)
        self.trainprogresscard = ExampleCard("Training", self.trainingProgressBar)
        self.trainaccuracyLabel = CaptionLabel("Train Accuracy: 0%")
        self.valaccuracyLabel = CaptionLabel("Val Accuracy: 0%")
        self.phase_image = ImageDisplayWidget(images_per_row=cfg.depth)
        self.output_image = ImageDisplayWidget(images_per_row=cfg.depth)
        self.textEdit = TextEdit(self)
        self.eval_button = TogglePushButton("Evaluate Insitu", self)
        self.calibration_button = TogglePushButton("Calibration", self)
        self.error_button = TogglePushButton("Error train", self)
        self.model_button = SplitPushButton("Select Model", self)
        self.model_button.setFlyout(self.model_menu)

    def _setup_layout(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(36, 22, 36, 12)

        vBoxLayout = QVBoxLayout()

        vBoxLayout.setSpacing(0)
        # vBoxLayout.setContentsMargins(36, 22, 36, 12)
        vBoxLayout.addWidget(self.titleLabel)
        vBoxLayout.addSpacing(4)
        vBoxLayout.addWidget(self.subtitleLabel)
        vBoxLayout.addSpacing(4)
        vBoxLayout.setAlignment(Qt.AlignTop)

        buttonLayout = QHBoxLayout()

        buttonLayout.setSpacing(4)
        buttonLayout.setContentsMargins(0, 0, 0, 0)

        # 添加按钮
        buttonLayout.addWidget(self.SLM_button, 0, Qt.AlignLeft)
        buttonLayout.addWidget(self.DMD_button, 0, Qt.AlignLeft)

        buttonLayout.addStretch()
        buttonLayout.addWidget(self.EVENT_button, 0, Qt.AlignLeft)
        buttonLayout.addWidget(self.progresscard, 0, Qt.AlignLeft)
        buttonLayout.addWidget(self.trainprogresscard, 0, Qt.AlignLeft)

        buttonLayout.addWidget(self.trainaccuracyLabel, 0, Qt.AlignRight)
        buttonLayout.addWidget(self.valaccuracyLabel, 0, Qt.AlignRight)

        # 添加训练按钮，保持在布局的右侧
        buttonLayout.addStretch(1)
        buttonLayout.addWidget(self.train_button, 0, Qt.AlignRight)
        buttonLayout.addWidget(self.train_insitu_button, 0, Qt.AlignRight)
        buttonLayout.addWidget(self.spinBox, 0, Qt.AlignRight | Qt.AlignVCenter)

        buttonLayout.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.subtitleLabel.setTextColor(QColor(96, 96, 96), QColor(216, 216, 216))

        self.textEdit.setPlainText(
            OmegaConf.to_yaml(OmegaConf.load("config/reconstruction.yaml"))
        )
        self.textEdit.setFixedHeight(150)

        # Adding layouts
        self.layout.addLayout(vBoxLayout)
        self.layout.addLayout(buttonLayout)
        # self.layout.addLayout(buttons_layout)
        # layout.addLayout(bar_layout)
        self.layout.addWidget(self.phase_image)
        self.layout.addWidget(self.output_image)
        self.layout.addWidget(self.textEdit)
        self.layout.addWidget(self.eval_button, 0, Qt.AlignRight)
        self.layout.addWidget(self.calibration_button, 0, Qt.AlignRight)
        self.layout.addWidget(self.error_button, 0, Qt.AlignRight)
        self.layout.addWidget(self.model_button, 0, Qt.AlignRight)
        self.layout.addStretch()

    def _setup_connections(self):
        self.train_button.clicked.connect(self.training)
        self.train_insitu_button.clicked.connect(self.training_insitu)
        self.eval_button.clicked.connect(self.eval_insitu)
        self.calibration_button.clicked.connect(self.calibration_on)
        self.error_button.clicked.connect(self.error_on)
        self.spinBox.valueChanged.connect(self.TrainedLayerChaneged)
        self.textEdit.textChanged.connect(self.update_config)
        self.SLM_Trigger_Thread.errorOccurred.connect(self.showError)
        self.SLM_thread.errorOccurred.connect(self.showError)
        self.DMD_Trigger_Thread.errorOccurred.connect(self.showError)
        self.DMD_thread.errorOccurred.connect(self.showError)

    def _init_training_thread(self):
        # self.training_thread = TrainingThread(
        #     self.value, callback=self.update_training_progress
        # )

        self.DMD_thread = function.DMD_thread()
        self.DMD_Trigger_Thread = function.DMD_Trigger_Thread()
        self.SLM_thread = function.SLM_thread()
        self.SLM_Trigger_Thread = function.SLM_trigger_thread()

        if self.camera == "CMOS":
            self.training_thread_insitu = reconstruction_pysical_trainthread_with_cmos(
                self.value,
                train=self.train,
                insitu_train=self.insitu_train,
                calibration=self.calibration,
                error=self.error_train,
                callback=self.update_training_progress,
            )
        elif self.camera == "EVENT":
            # self.training_thread_insitu = reconstruction_pysical_trainthread_with_event(
            #     self.value,
            #     train=self.train,
            #     insitu_train=self.insitu_train,
            #     calibration=self.calibration,
            #     error=self.error_train,
            #     callback=self.update_training_progress,
            # )
            raise NotImplementedError
            pass
        else:
            self.training_thread_insitu = reconstruction_pysical_trainthread_with_cmos(
                self.value,
                train=self.train,
                calibration=self.calibration,
                error=self.error_train,
                callback=self.update_training_progress,
            )

        # self.training_thread.update_progress.connect(self.on_update_progress)
        # self.training_thread.update_images.connect(self.update_image_display)
        # self.training_thread.update_val_accuracy.connect(self.update_val_accuracy)
        # self.training_thread.errorOccurred.connect(self.showError)

        self.training_thread_insitu.update_progress.connect(self.on_update_progress)
        self.training_thread_insitu.update_images.connect(self.update_image_display)
        self.training_thread_insitu.update_val_accuracy.connect(
            self.update_val_accuracy
        )
        self.training_thread_insitu.errorOccurred.connect(self.showError)

    def training(self):
        if self.train_button.isChecked():
            self.train = True
            self.insitu_train = False
            self._init_training_thread()
            self.training_thread_insitu.start()
            self.progressBar.pause()
            self.train_button.setIcon(FluentIcon.PLAY)
        else:
            self.train = False
            self.training_thread.stop()
            self.progressBar.resume()
            self.train_button.setIcon(FluentIcon.PAUSE)

    def training_insitu(self):
        if self.train_insitu_button.isChecked():
            self.insitu_train = True
            self.train = False
            self._init_training_thread()
            if self.camera == "CMOS" or self.camera == "EVENT":
                self.training_thread_insitu.start()
                self.progressBar.pause()
                self.train_insitu_button.setIcon(FluentIcon.PLAY)
            else:
                interface.createErrorInfoBar("ERROR", "please choose a camera!", self)

        else:
            self.training_thread_insitu.stop()
            self.insitu_train = False
            self.progressBar.resume()
            self.train_insitu_button.setIcon(FluentIcon.PAUSE)

    def eval_insitu(self):
        if self.eval_button.isChecked():
            self.train = False
            self._init_training_thread()
            self.training_thread_insitu.start()
        else:
            self.training_thread_insitu.stop()
            self.progressBar.resume()

    def calibration_on(self):
        if self.calibration_button.isChecked():
            self.calibration = True
            self.train = False
            self._init_training_thread()
            self.training_thread_insitu.start()
        else:
            self.training_thread_insitu.stop()
            self.progressBar.resume()

    def error_on(self):
        if self.error_button.isChecked():
            self.train = False
            self.error_train = True
            self._init_training_thread()
            self.training_thread_insitu.start()
        else:
            self.error_train = False
            self.training_thread_insitu.stop()
            self.progressBar.resume()

    def DMD_on(self, text):
        if text == "Trigger Mode":
            self.DMD_flag = 0
            self.DMD_button.setText("Trigger Mode")
            self.DMD_Trigger_Thread.start()
            interface.createSuccessInfoBar("DMD", "Trigger Mode On", self)
        elif text == "Loop Mode":
            self.DMD_flag = 1
            self.DMD_button.setText("Loop Mode")
            self.DMD_thread.start()
            interface.createSuccessInfoBar("DMD", "Loop Mode On", self)
        else:
            if self.DMD_flag == 0:
                self.DMD_Trigger_Thread.stop()
            elif self.DMD_flag == 1:
                self.DMD_thread.stop()
            else:
                pass
            self.DMD_button.setText("DMD OFF")
            interface.createSuccessInfoBar("DMD", "Success Off", self)

    def SLM_on(self, text):
        if text == "Trigger Mode":
            self.SlmFlag = 0
            self.SLM_button.setText("Trigger Mode")
            self.SLM_Trigger_Thread.start()
            interface.createSuccessInfoBar("SLM", "Trigger Mode On", self)
        elif text == "Loop Mode":
            self.SlmFlag = 1
            self.SLM_button.setText("Loop Mode")
            self.SLM_thread.start()
            interface.createSuccessInfoBar("SLM", "Loop Mode On", self)
        else:
            if self.SlmFlag == 0:
                self.SLM_Trigger_Thread.stop()
            elif self.SlmFlag == 1:
                self.SLM_thread.stop()
            else:
                pass
            self.SLM_button.setText("SLM OFF")
            interface.createSuccessInfoBar("SLM", "Success Off", self)

    def model_select(self, text):
        self.model = text
        if text == "lightridge":
            self.model_button.setText("lightridge")
        else:
            self.model_button.setText("reconstruction")

        self._init_training_thread()

    def on_update_progress(self, value):
        self.progressBar.setValue(value)

    def TrainedLayerChaneged(self):
        self.value = self.spinBox.value()
        self._init_training_thread()

    def on_menu_item_selected(self, text):
        self.EVENT_button.setText(text)
        self.camera = text
        if text == "CMOS":
            # self.training_thread_insitu = pysical_trainthread_with_cmos(self.value)
            interface.createSuccessInfoBar("CMOS", "train with cmos on", self)
        elif text == "EVENT":
            # self.training_thread_insitu = pysical_trainthread_with_event(self.value)
            interface.createSuccessInfoBar("Event", "train with event on", self)
        else:
            interface.createSuccessInfoBar("None", "train in simulation", self)

        self._init_training_thread()

    def update_val_accuracy(self, accuracy):
        self.valaccuracyLabel.setText(f"Val Accuracy: {accuracy:.2f}%")

    def update_training_progress(
        self, current_batch, total_batches, train_loss, train_accuracy
    ):
        progress = int((current_batch / total_batches) * 100)
        self.trainingProgressBar.setValue(progress)
        self.trainaccuracyLabel.setText(f"Train Accuracy: {train_accuracy * 100:.2f}%")

    def update_image_display(self, image_paths):
        self.phase_image.update_images(image_paths[: cfg.depth])
        self.output_image.update_images(image_paths[cfg.depth :])

    def update_config(self):
        text = self.textEdit.toPlainText()

        try:
            config_original = OmegaConf.load("config/reconstruction.yaml")
            config_update = OmegaConf.create(text)
            for key in config_original:
                if key in config_original:
                    config_original[key] = config_update[key]
        except Exception as e:
            print(f"错误: {e}")

        cfg.merge_with(config_original)
        # 将更新后的配置写回文件
        with open("config/reconstruction.yaml", "w") as f:
            OmegaConf.save(config_original, f)

        try:
            config_original = OmegaConf.load("config/filepath.yaml")
            print(config_original.output)

            config_original["phase"] = [
                f"img/phase_{index}.png" for index in range(cfg.depth)
            ]
            config_original["output"] = [
                f"img/mnist_{index+1}.png" for index in range(cfg.depth)
            ]

        except Exception as e:
            print(f"错误: {e}")

        with open("config/filepath.yaml", "w") as f:
            OmegaConf.save(config_original, f)

    def showError(self, error):
        interface.createErrorInfoBar("Error", error, self)
        self.eval_button.setChecked(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = s2nninterface()
    ex.show()
    sys.exit(app.exec_())
