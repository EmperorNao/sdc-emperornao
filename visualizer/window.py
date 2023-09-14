import os
import sys
from os.path import join

from vispy.io import read_png

from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QCheckBox,
    QLineEdit,
    QSlider,
    QMdiArea,
    QFileDialog
)
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator, QPixmap

from utils.filesystem import mkdir

from visualizer.canvas2d import CanvasImageDrawer
from visualizer.canvas3d import CanvasPointCloudDrawer
from visualizer.subwindow import Display
from datasets.cadc.cadc import CADCProxySequence
from gui_lib.image import create_bev_image


class CarVisualizer(QtWidgets.QMainWindow):

    def __init__(self, args):
        super().__init__()

        self.setWindowTitle("Autonomous driving")

        self.general_layout = QVBoxLayout()
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.centralWidget.setLayout(self.general_layout)

        self.options_layout = QHBoxLayout()
        self.options_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.general_layout.addLayout(self.options_layout)

        self.common_layout = QHBoxLayout()
        self.general_layout.addLayout(self.common_layout)

        self.canvas_drawer = None
        self._create_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        self.dataset_path = args.dataset_path if args else "/home/emperornao/personal/sdc2/data/cadc/2018_03_06/0001"
        self.calib_path = args.calib_path if args else "/home/emperornao/personal/sdc2/data/cadc/2018_03_06/calib"

        self.create_subwindows({})

        self.sequence = None
        self.cache_dir = join(sys.path[0], ".cache")
        mkdir(self.cache_dir, True)

        self.showMaximized()

    def _create_ui(self):

        # show mode
        self.mode = QComboBox()

        self.mode.setMinimumWidth(200)
        self.mode.setMaximumWidth(200)
        self.mode.setMaximumHeight(70)

        self.mode.addItem("cameras + point cloud")
        self.mode.addItem("cameras + bev")
        self.mode.currentTextChanged.connect(self.create_canvas)

        self.options_layout.addWidget(self.mode)
        # self.mode.currentTextChanged.connect(self.restart)

        # open dir with dataset
        self.get_dir_with_date = QPushButton("open dataset")
        self.get_dir_with_date.setMinimumWidth(100)
        self.get_dir_with_date.setMaximumWidth(200)
        self.get_dir_with_date.setMaximumHeight(70)
        self.get_dir_with_date.clicked.connect(self.open_directory)
        self.options_layout.addWidget(self.get_dir_with_date)

        # open dir with calib
        self.get_dir_with_calib = QPushButton("open calib")
        self.get_dir_with_calib.setMinimumWidth(100)
        self.get_dir_with_calib.setMaximumWidth(200)
        self.get_dir_with_calib.setMaximumHeight(70)
        self.get_dir_with_calib.clicked.connect(self.open_directory_calib)
        self.options_layout.addWidget(self.get_dir_with_calib)

        # close all windows
        self.close_sub_windows_btn = QPushButton("close windows")
        self.close_sub_windows_btn.setMinimumWidth(100)
        self.close_sub_windows_btn.setMaximumWidth(200)
        self.close_sub_windows_btn.setMaximumHeight(70)
        self.close_sub_windows_btn.clicked.connect(self.close_subwindows)
        self.options_layout.addWidget(self.close_sub_windows_btn)

        # cache for bev
        self.use_cache = QCheckBox()
        self.options_layout.addWidget(self.use_cache)

        # concentration
        self.concentration_input = QLineEdit(self)
        self.concentration_input.setMinimumWidth(100)
        self.concentration_input.setMaximumWidth(200)
        self.concentration_input.setMaximumHeight(70)

        self.concentration_input.setValidator(QIntValidator(16, 512, self))
        self.concentration_input.setText("50")
        self.concentration_input.textChanged.connect(self.change_concentration)
        self.options_layout.addWidget(self.concentration_input)

        # run
        self.run = QPushButton("run")
        self.run.setMinimumWidth(100)
        self.run.setMaximumWidth(200)
        self.run.setMaximumHeight(70)
        self.run.setCheckable(True)
        self.run.setChecked(False)
        self.run.clicked.connect(self.run_or_stop)
        self.options_layout.addWidget(self.run)

        # slider for frames
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.run_frame)
        self.slider.setMaximum(0)
        self.options_layout.addWidget(self.slider)

        # mdi for displaying images
        self.mdi = QMdiArea()
        self.mdi.setMaximumHeight(1500)
        self.mdi.setMaximumWidth(1500)
        self.common_layout.addWidget(self.mdi)
        self.subs = []

        self.create_canvas("cameras + point cloud")

    def create_canvas(self, text):
        if self.canvas_drawer:
            self.common_layout.removeWidget(self.canvas_drawer.canvas.native)
            self.canvas_drawer.canvas.native.deleteLater()
            self.canvas_drawer = None

        if text == "cameras + bev":
            self.canvas_drawer = CanvasImageDrawer()
        elif text == "cameras + point cloud":
            self.canvas_drawer = CanvasPointCloudDrawer()

        self.common_layout.addWidget(self.canvas_drawer.canvas.native)

    def change_concentration(self, text):
        if self.mode.currentText() == "cameras + point cloud":
            self.run_frame(self.slider.value())

    def run_frame(self, value):

        if value >= len(self.sequence):
            self.run.click()
            return

        current_dir = join(self.cache_dir, "_".join(self.dataset_path.split("/")))
        if not os.path.exists(current_dir):
            os.mkdir(current_dir)

        l = {
            0: 3,
            1: 4,
            2: 5,
            8: 7,
            3: 2,
            5: 6,
            6: 1,
            7: 0
        }
        for i in range(9):
            if i in l:
                self.subs[i].set_image(QPixmap(f"{self.dataset_path}/labeled/image_{str(l[i]).rjust(2, '0')}/data/0000000{str(value).rjust(3, '0')}.png"))
            if i == 9:
                pass

        state = self.sequence[value]
        points = state['points']
        targets = state['boxes']

        if self.mode.currentText() == "cameras + bev":
            current_file = join(current_dir, str(value) + "_bev.png")
            if not self.use_cache.isChecked() or not os.path.exists(current_file):
                create_bev_image((50, 50), points, targets["boxes"], current_file)
            image = read_png(current_file)
            if self.self.canvas_drawer:
                self.canvas_drawer.set_image(image)

        elif self.mode.currentText() == "cameras + point cloud":
            if self.canvas_drawer:
                self.canvas_drawer.set_point_cloud(points[:, :3],
                                               concentration=float(self.concentration_input.text()),
                                               labels=targets)

    def next_frame(self):
        if not self.sequence:
            self.sequence = CADCProxySequence(
                data_dir=f'{self.dataset_path}',
                calib_dir=f'{self.calib_path}'
            )
            self.slider.setMaximum(len(self.sequence) - 1)

        if self.run.isChecked():
            self.slider.setValue(self.slider.value() + 1)
            self.timer.start(50)
        else:
            self.timer.stop()

    def open_directory(self):
        self.dataset_path = QFileDialog.getExistingDirectory(self, 'Select folder with sequence')
        self.sequence = None

    def open_directory_calib(self):
        self.calib_path = QFileDialog.getExistingDirectory(self, 'Select folder with cailbrations')
        self.sequence = None

    def close_subwindows(self):
        # self.mdi.closeAllSubWindows()
        for sub in self.subs:
            sub.close()

    def run_or_stop(self):
        if not self.run.isChecked():
            self.run.setText("run")
            self.run.setDown(False)
            self.timer.stop()
        else:
            self.run.setText("stop")
            self.run.setDown(True)
            self.slider.setValue(self.slider.value())
            self.timer.start(50)

    def restart(self, new):

        self.close_subwindows()
        options_to_windows = {

        }
        self.create_subwindows(options_to_windows)

    def create_subwindows(self, options):

        for i in range(9):
            sub = Display((400, 400))
            sub.setWindowTitle("window " + str(i))
            self.subs.append(sub)
            self.mdi.addSubWindow(sub)
        self.mdi.tileSubWindows()


# This import must be last, because it fixes import error with QT
from utils import qt_cv2_fix  # noqa
