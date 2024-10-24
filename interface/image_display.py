import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QSize


class ImageDisplayWidget(QWidget):
    def __init__(self, images_per_row, max_image_size=QSize(200, 200)):
        super().__init__()
        self.images_per_row = images_per_row
        self.max_image_size = max_image_size
        self.main_layout = QVBoxLayout(self)
        self.setLayout(self.main_layout)
        self.add_blank_images()

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def add_images(self, image_paths):
        self.clear_layout(self.main_layout)
        row = None
        for i, path in enumerate(image_paths):
            if i % self.images_per_row == 0:
                row = QHBoxLayout()
                self.main_layout.addLayout(row)

            label = QLabel(self)
            label.setAlignment(Qt.AlignCenter)
            pixmap = QPixmap(path)
            scaled_pixmap = pixmap.scaled(
                self.max_image_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
            row.addWidget(label)

    def add_blank_images(self):
        blank_image_paths = ["resource\\blank.png"] * self.images_per_row
        self.add_images(blank_image_paths)

    def update_images(self, new_image_paths):
        self.add_images(new_image_paths)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = ImageDisplayWidget(3)  # 例如，每行3张图片
    widget.show()
    sys.exit(app.exec_())
