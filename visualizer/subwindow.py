

from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout


class Display(QWidget):
    def __init__(self, size):
        super().__init__()
        self.setWindowTitle("Display")
        self.label = QLabel()

        self.size = size
        self.label.setFixedSize(*self.size)

        self.layout_country = QVBoxLayout()
        self.layout_country.addWidget(self.label)

        self.setLayout(self.layout_country)

        self.pixmap = None
        self.show()

    def set_image(self, pixmap, size=None):
        self.pixmap = pixmap
        self.show_image()
        if size is not None:
            self.size = size
            self.label.setFixedSize(*self.size)
        self.show()

    def show_image(self):
        if self.pixmap is not None:
            self.label.setPixmap(self.pixmap.scaled(*self.size))