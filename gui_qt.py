import os
import sys

import cv2
# OpenCV2+PyQt5 issue workaround for Linux
# https://forum.qt.io/topic/119109/using-pyqt5-with-opencv-python-cv2-causes-error-could-not-load-qt-platform-plugin-xcb-even-though-it-was-found/21
from cv2.version import ci_build, headless

ci_and_not_headless = ci_build and not headless
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
    os.environ.pop("QT_QPA_FONTDIR")

import numpy as np
from PyQt5.QtCore import QPoint, QRect, Qt
from PyQt5.QtGui import QPixmap, QPainter, QColor, QBrush, QImage, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget, QInputDialog, \
    QErrorMessage, QScrollArea, QHBoxLayout, QRadioButton, QGroupBox, QListWidget, QAbstractItemView, QListWidgetItem, \
    QSlider

app = QApplication(sys.argv)


class RoiWidget(QLabel):
    def __init__(self):
        super().__init__()
        self._begin = QPoint()
        self._end = QPoint()
        self._roi_dict = dict()
        self._cur_key = None

        # Using ColorMap from matplotlib
        import matplotlib.pyplot as plt
        self._cm = plt.cm.get_cmap("tab10")
        self._cm_count = 0

    def add(self, name) -> bool:
        # Name existed, return false as failed
        if name in self._roi_dict:
            return False

        # Assign new color
        self._cm_count += 1
        r, g, b, _ = self._cm((self._cm_count % 10) / 10)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)

        # Create new entry
        self._roi_dict[name] = dict(rgb=(r, g, b), roi=QRect())
        self._cur_key = name
        print(f"[ROI] Add", name)
        return True

    def switch_to(self, name) -> bool:
        # Name not found, return false as failed
        if name not in self._roi_dict:
            return False
        print(f"[ROI] Switch to", name)
        self._cur_key = name
        return True

    def remove(self, name):
        # Name not found, return false as failed
        if name not in self._roi_dict.keys():
            return False

        # Remove entry from dict
        del self._roi_dict[name]
        print(f"[ROI] Remove", name)
        self._cur_key = None

        # # Not need, already handled by PyQt
        # if self._roi_dict:
        #     # Set current key as the key of the last entry
        #     self._cur_key = list(self._roi_dict.keys())[-1]
        # else:
        #     # No entry left, so set the current key to None
        #     self._cur_key = None

        # Redraw
        self.update()
        return True

    def get_roi(self, name):
        if name in self._roi_dict:
            return self._roi_dict[name]["roi"]
        return None

    def get_rgb(self, name):
        if name in self._roi_dict:
            return self._roi_dict[name]["rgb"]
        return None

    def get_result(self):
        """
        :return: Format: (x,y,w,h)
        """
        return self._roi_dict

    def update_roi(self):
        # Update current ROI
        cur_item = self._roi_dict[self._cur_key]
        # Normalized (Ensure that no negative width and height)
        rect_roi = QRect(self._begin, self._end).normalized()
        # Ensure in-bound, get intersected with the image's rect
        rect_roi = rect_roi.intersected(self.rect())
        cur_item["roi"] = rect_roi
        # Debug
        # print("[ROI]", self._roi_dict[self._cur_key])

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._roi_dict:
            # Draw all ROIs
            painter = QPainter(self)
            for name, item in self._roi_dict.items():
                rgb, roi = item["rgb"], item["roi"]
                brush = QBrush(QColor(*rgb, 100))
                painter.setBrush(brush)
                painter.drawRect(roi)
                painter.drawText(roi.topLeft(), name)

    def mousePressEvent(self, event):
        if self._roi_dict:
            self._begin = event.pos()
            self._end = event.pos()
            self.update_roi()
            self.update()

    def mouseMoveEvent(self, event):
        if self._roi_dict:
            self._end = event.pos()
            self.update_roi()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._roi_dict:
            self._end = event.pos()
            self.update_roi()
            self.update()


def cv2_to_qpixmap(img: np.ndarray):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channel = img.shape
    bytes_per_line = 3 * width
    qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    qpixmap = QPixmap.fromImage(qimage)
    return qpixmap


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GUI")
        self.layout = QVBoxLayout()

        # self.pixmap_image = QPixmap("test.jpg")
        self.qpixmap = cv2_to_qpixmap(cv2.cvtColor(cv2.imread("test2k.png"), cv2.COLOR_BGR2GRAY))

        hbox_top = QHBoxLayout()
        # Source image
        vbox_src = QVBoxLayout()
        vbox_src.addWidget(QLabel("Source image"))
        self.roi_widget = RoiWidget()
        self.roi_widget.setPixmap(self.qpixmap)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.roi_widget)
        vbox_src.addWidget(scroll_area)
        hbox_top.addLayout(vbox_src, stretch=2)

        hbox_bin_cell = QHBoxLayout()
        # Binarization
        vbox_bin = QVBoxLayout()
        vbox_bin.addWidget(QLabel("Binarization"))
        self.bin_display = QLabel()
        self.bin_display.setPixmap(self.qpixmap)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.bin_display)
        vbox_bin.addWidget(scroll_area)
        hbox_bin_cell.addLayout(vbox_bin)

        # Cell Area
        vbox_cell = QVBoxLayout()
        vbox_cell.addWidget(QLabel("Cell Area"))
        self.cell_display = QLabel()
        self.cell_display.setPixmap(self.qpixmap)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.cell_display)
        vbox_cell.addWidget(scroll_area)
        hbox_bin_cell.addLayout(vbox_cell)

        vbox_bin_cell_ctrl = QVBoxLayout()
        vbox_bin_cell_ctrl.addLayout(hbox_bin_cell)
        # Controls
        group_box = QGroupBox("Threshold method")
        self.r0 = QRadioButton("Manual threshold")
        self.r0.toggled.connect(self.on_radio_button_toggled_r0)
        self.r1 = QRadioButton("Auto threshold (Otsu's method)")
        self.r1.toggled.connect(self.on_radio_button_toggled_r1)
        self.r2 = QRadioButton("Adaptive threshold")
        self.r2.toggled.connect(self.on_radio_button_toggled_r2)
        # Params
        self.param_group_box = QGroupBox("Parameters")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.block_size_slider = QSlider(Qt.Horizontal)
        self.c_slider = QSlider(Qt.Horizontal)
        vbox_sliders = QVBoxLayout()
        vbox_sliders.addWidget(self.threshold_slider)
        vbox_sliders.addWidget(self.block_size_slider)
        vbox_sliders.addWidget(self.c_slider)
        self.param_group_box.setLayout(vbox_sliders)
        vbox_ctrl = QVBoxLayout()
        vbox_ctrl.addWidget(self.r0)
        vbox_ctrl.addWidget(self.r1)
        vbox_ctrl.addWidget(self.r2)
        vbox_ctrl.addWidget(self.param_group_box)
        group_box.setLayout(vbox_ctrl)
        vbox_bin_cell_ctrl.addWidget(group_box)
        hbox_top.addLayout(vbox_bin_cell_ctrl, stretch=1)
        self.layout.addLayout(hbox_top)
        # Default check
        self.r2.setChecked(True)

        # ROI Selector
        group_box = QGroupBox("ROI Selection")
        self.roi_selector = QListWidget()
        self.roi_selector.setSelectionMode(QAbstractItemView.SingleSelection)
        self.roi_selector.itemSelectionChanged.connect(self.on_item_selection_changed)
        button_hbox = QHBoxLayout()
        self.button_add = QPushButton("New ROI")
        self.button_add.clicked.connect(self.on_click_add)
        button_hbox.addWidget(self.button_add)
        self.button_remove = QPushButton("Remove ROI")
        self.button_remove.clicked.connect(self.on_click_remove)
        button_hbox.addWidget(self.button_remove)
        vbox = QVBoxLayout()
        vbox.addWidget(self.roi_selector)
        vbox.addLayout(button_hbox)
        group_box.setLayout(vbox)
        self.layout.addWidget(group_box)

        self.button_ok = QPushButton("End selection")
        self.button_ok.clicked.connect(self.on_click_ok)
        self.layout.addWidget(self.button_ok)

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)
        self.showMaximized()

    def on_radio_button_toggled_r0(self, selected):
        # Manual threshold
        if selected:
            self.param_group_box.show()
            self.threshold_slider.show()
            self.block_size_slider.hide()
            self.c_slider.hide()
            print("r0")

    def on_radio_button_toggled_r1(self, selected):
        # Auto threshold (Otsu's method)
        if selected:
            self.param_group_box.hide()
            print("r1")

    def on_radio_button_toggled_r2(self, selected):
        # Adaptive threshold
        if selected:
            self.param_group_box.show()
            self.threshold_slider.hide()
            self.block_size_slider.show()
            self.c_slider.show()
            print("r2")

    def on_click_ok(self):
        import pprint
        pprint.pprint(self.roi_widget.get_result())

    def on_click_add(self):
        name, ret = QInputDialog.getText(self, "New ROI", "Enter a name")
        if ret:
            if name:
                success = self.roi_widget.add(name)
                if success:
                    pixmap = QPixmap(16, 16)
                    pixmap.fill(QColor().fromRgb(*self.roi_widget.get_rgb(name)))
                    icon = QIcon(pixmap)
                    list_item = QListWidgetItem(icon, name)
                    # list_item.setIcon(icon)
                    self.roi_selector.addItem(list_item)
                    self.roi_selector.setCurrentItem(list_item)
                else:
                    error_dialog = QErrorMessage(self)
                    error_dialog.setWindowTitle("Error")
                    error_dialog.showMessage("Name already exists")
            else:
                error_dialog = QErrorMessage(self)
                error_dialog.setWindowTitle("Error")
                error_dialog.showMessage("Invalid name")

    def on_click_remove(self):
        cur_item = self.roi_selector.currentItem()
        if cur_item:
            name = cur_item.text()
            row = self.roi_selector.row(cur_item)
            success = self.roi_widget.remove(name)
            if success:
                self.roi_selector.takeItem(row)

    def on_item_selection_changed(self):
        cur_item = self.roi_selector.currentItem()
        if cur_item:
            name = cur_item.text()
            success = self.roi_widget.switch_to(name)


window = MainWindow()
window.show()

app.exec()
