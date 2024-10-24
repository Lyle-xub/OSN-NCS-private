import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QLabel,
    QFileDialog,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
)
from PyQt5.QtGui import QPixmap, QImage, QColor
from PIL import Image
import numpy as np
import cv2
from qfluentwidgets import TransparentTogglePushButton, TogglePushButton
from qfluentwidgets import ProgressBar, TitleLabel, CaptionLabel


def crop_center(img, size=400):
    width, height = img.size
    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2
    return img.crop((left, top, right, bottom))


def superimpose_images(image1, image2):
    img1 = np.array(image1)
    img2 = np.array(image2)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    superimpose = np.mod(img1 + img2, 255).astype(np.uint8)
    return superimpose


def pad_image(img, new_size=(1272, 1024)):
    padded_image = Image.new("L", (new_size[0], new_size[1]), color="black")
    offset = ((new_size[0] - img.width) // 2, (new_size[1] - img.height) // 2)
    padded_image.paste(img, offset)
    return padded_image


# Convert PIL Image to QImage
def pil2pixmap(im):
    if im.mode == "RGB":
        pass
    elif im.mode == "L":
        im = im.convert("RGBA")
    data = im.tobytes("raw", im.mode)
    qim = QImage(data, im.size[0], im.size[1], QImage.Format_ARGB32)
    return QPixmap.fromImage(qim)


class ImageProcessorInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("ImageProcessorInterface")
        self.image1 = None
        self.image2 = None
        self.initUI()

    def initUI(self):
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(36, 22, 36, 12)
        self.titleLabel = TitleLabel("SuperImpose Image", self)
        self.subtitleLabel = CaptionLabel("superimpose with blazed", self)
        self.subtitleLabel.setTextColor(QColor(96, 96, 96), QColor(216, 216, 216))

        # Buttons
        self.buttonlayout = QHBoxLayout()
        self.load_image1_button = TogglePushButton("Load Image 1")
        self.load_image1_button.clicked.connect(self.loadImage1)

        self.load_image2_button = TogglePushButton("Load Image 2")
        self.load_image2_button.clicked.connect(self.loadImage2)

        self.process_button = TogglePushButton("Process Images")
        self.process_button.clicked.connect(self.processImages)

        self.clear_button = TogglePushButton("Clear")
        self.clear_button.clicked.connect(self.clearImages)

        # Image display
        self.image_label = QLabel(self)
        self.image_label.setPixmap(QPixmap("resource/place_holder.jpeg"))
        self.buttonlayout.addWidget(self.load_image1_button)
        self.buttonlayout.addWidget(self.load_image2_button)
        self.buttonlayout.addWidget(self.process_button)
        self.buttonlayout.addWidget(self.clear_button)

        self.layout.addSpacing(4)
        self.layout.addWidget(self.titleLabel)
        self.layout.addSpacing(4)
        self.layout.addWidget(self.subtitleLabel)
        self.layout.addSpacing(4)
        self.layout.addLayout(self.buttonlayout)

        self.layout.addWidget(self.image_label)
        self.layout.addStretch()

    def loadImage1(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open file", "/home", "Image files (*.jpg *.gif *.png *.bmp)"
        )
        if fname:
            self.image1 = Image.open(fname)

    def loadImage2(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open file", "/home", "Image files (*.jpg *.gif *.png *.bmp)"
        )
        if fname:
            self.image2 = Image.open(fname)

    def processImages(self):
        if self.image1 and self.image2:
            cropped_image1 = crop_center(self.image1)
            cropped_image2 = crop_center(self.image2)
            superimposed_image = superimpose_images(cropped_image1, cropped_image2)
            superimposed_image_pil = Image.fromarray(superimposed_image)
            padded_image = pad_image(superimposed_image_pil)
            pixmap = pil2pixmap(padded_image)
            self.image_label.setPixmap(pixmap)
            # save image
            padded_image.save("img/superimposed.bmp")
        else:
            print("Please load both images first.")

    def clearImages(self):
        if self.clear_button.isChecked():
            self.image1 = None
            self.image2 = None
            self.image_label.setPixmap(QPixmap("resource/place_holder.jpeg"))
            self.load_image1_button.setChecked(False)
            self.load_image2_button.setChecked(False)
            self.process_button.setChecked(False)
