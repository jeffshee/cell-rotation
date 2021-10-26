import os
import sys
import threading
from multiprocessing import freeze_support
from pprint import pprint

# OpenCV2+PyQt5 issue workaround for Linux
# https://forum.qt.io/topic/119109/using-pyqt5-with-opencv-python-cv2-causes-error-could-not-load-qt-platform-plugin-xcb-even-though-it-was-found/21
from PyQt5 import QtGui
from cv2.version import ci_build, headless

from calc_v2 import main
from utils_v2 import *

ci_and_not_headless = ci_build and not headless
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
    os.environ.pop("QT_QPA_FONTDIR")

import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtCore import QPoint, QRect, Qt, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QPainter, QColor, QBrush, QImage, QIcon, QKeySequence
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QInputDialog, \
    QHBoxLayout, QRadioButton, QGroupBox, QListWidgetItem, \
    QSlider, QScrollArea, QListWidget, QAbstractItemView, QPushButton, QFileDialog, QMessageBox, QAction


# # Zoom in/out with scrollbar adjustment
# # Reference:
# # https://docs.huihoo.com/qt/4.5/widgets-imageviewer.html
# def adjust_scrollbar(scrollbar: QScrollBar, factor):
#     scrollbar.setValue(int(factor * scrollbar.value() + (factor - 1) * scrollbar.pageStep() / 2))


class RoiWidget(QLabel):
    RoiChanged = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._begin = QPoint()
        self._end = QPoint()
        self._roi_dict = dict()
        self._cur_key = None

        # Using ColorMap from matplotlib
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
        print(f"[ROI] add", name)
        return True

    def switch_to(self, name) -> bool:
        # Name not found, return false as failed
        if name not in self._roi_dict:
            return False
        print(f"[ROI] switch_to", name)
        self._cur_key = name
        self.RoiChanged.emit()
        return True

    def remove(self, name):
        # Name not found, return false as failed
        if name not in self._roi_dict.keys():
            return False

        # Remove entry from dict
        del self._roi_dict[name]
        print(f"[ROI] remove", name)
        self._cur_key = None
        self.RoiChanged.emit()

        # Redraw
        self.update()
        return True

    def get_current(self):
        return self._cur_key

    def get_roi(self, name):
        if name in self._roi_dict:
            return self._roi_dict[name]["roi"]
        return QRect()

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
        self.RoiChanged.emit()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._roi_dict:
            # Get current scaling
            scale = self.width() / self.pixmap().width()
            # Draw all ROIs
            painter = QPainter(self)
            for name, item in self._roi_dict.items():
                rgb, roi = item["rgb"], item["roi"]
                brush = QBrush(QColor(*rgb, 100))
                painter.setBrush(brush)
                roi_scaled = QRect(int(roi.x() * scale), int(roi.y() * scale),
                                   int(roi.width() * scale), int(roi.height() * scale))
                painter.drawRect(roi_scaled)
                painter.drawText(roi_scaled.topLeft(), name)

    def mousePressEvent(self, event):
        if self._roi_dict:
            # Get current scaling
            scale = self.width() / self.pixmap().width()
            self._begin = event.pos() / scale
            self._end = event.pos() / scale
            self.update_roi()
            self.update()

    def mouseMoveEvent(self, event):
        if self._roi_dict:
            # Get current scaling
            scale = self.width() / self.pixmap().width()
            self._end = event.pos() / scale
            self.update_roi()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._roi_dict:
            # Get current scaling
            scale = self.width() / self.pixmap().width()
            self._end = event.pos() / scale
            self.update_roi()
            self.update()


def cv2_to_qpixmap(img: np.ndarray):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channel = img.shape
    bytes_per_line = 3 * width
    qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    qpixmap = QPixmap.fromImage(qimage)
    return qpixmap


class ZoomScrollArea(QScrollArea):
    def __init__(self, scale_min=1.0, scale_max=4.0, step=0.1):
        super().__init__()
        self.scale_factor_min = scale_min
        self.scale_factor_max = scale_max
        self.scale_factor_step = step
        self._scale_factor = 1.0

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if event.modifiers() == Qt.ControlModifier:
            prev_scale_factor = self._scale_factor
            if event.angleDelta().y() > 0:
                self._scale_factor += self.scale_factor_step
            else:
                self._scale_factor -= self.scale_factor_step
            self._scale_factor = max(min(self._scale_factor, self.scale_factor_max), self.scale_factor_min)
            print("[Zoom] {:.1f}".format(self._scale_factor))
            cur_size = self.widget().size()
            original_size = cur_size / prev_scale_factor
            new_size = original_size * self._scale_factor
            """
            Reference: https://stackoverflow.com/questions/3725342/zooming-in-out-on-a-mouser-point/32269574
            QPointF ScrollbarPos = QPointF(horizontalScrollBar()->value(), verticalScrollBar()->value());
            QPointF DeltaToPos = e->posF() / OldScale - widget()->pos() / OldScale;
            QPointF Delta = DeltaToPos * NewScale - DeltaToPos * OldScale;
            """
            scrollbar_pos = QPoint(self.horizontalScrollBar().value(), self.verticalScrollBar().value())
            delta_to_pos = event.pos() / prev_scale_factor - self.widget().pos() / prev_scale_factor
            delta = delta_to_pos * self._scale_factor - delta_to_pos * prev_scale_factor
            self.widget().resize(new_size)
            # Adjust Scrollbar
            self.horizontalScrollBar().setValue(scrollbar_pos.x() + delta.x())
            self.verticalScrollBar().setValue(scrollbar_pos.y() + delta.y())
        else:
            super().wheelEvent(event)

    def get_scale_factor(self):
        return self._scale_factor


class ControlPanel(QVBoxLayout):
    def __init__(self):
        super().__init__()
        # Control Panel
        self.vbox_control_panel = QVBoxLayout()

        # Threshold method
        self.radio_manual = QRadioButton("Manual threshold")
        self.radio_auto = QRadioButton("Auto threshold (Otsu's method)")
        self.radio_adap_mean = QRadioButton("Adaptive threshold (MEAN_C)")
        self.radio_adap_gaussian = QRadioButton("Adaptive threshold (GAUSSIAN_C)")
        vbox = QVBoxLayout()
        vbox.addWidget(self.radio_manual)
        vbox.addWidget(self.radio_auto)
        vbox.addWidget(self.radio_adap_mean)
        vbox.addWidget(self.radio_adap_gaussian)
        self.group_box_thresh_method = QGroupBox("Threshold method")
        self.group_box_thresh_method.setLayout(vbox)
        self.addWidget(self.group_box_thresh_method)

        # Manual param
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.group_box_threshold = QGroupBox(f"Threshold {self.threshold_slider.value()}")
        hbox = QHBoxLayout()
        hbox.addWidget(self.threshold_slider)
        self.group_box_threshold.setLayout(hbox)
        self.addWidget(self.group_box_threshold)

        # Adaptive params
        self.block_size_slider = QSlider(Qt.Horizontal)
        self.group_box_block_size = QGroupBox(f"Block Size {self.block_size_slider.value()}")
        hbox = QHBoxLayout()
        hbox.addWidget(self.block_size_slider)
        self.group_box_block_size.setLayout(hbox)
        self.addWidget(self.group_box_block_size)

        self.c_slider = QSlider(Qt.Horizontal)
        self.group_box_c = QGroupBox(f"C {self.c_slider.value()}")
        hbox = QHBoxLayout()
        hbox.addWidget(self.c_slider)
        self.group_box_c.setLayout(hbox)
        self.addWidget(self.group_box_c)

        # Frame
        self.frame_slider = QSlider(Qt.Horizontal)
        self.group_box_frame = QGroupBox(f"Frame {self.frame_slider.value()}")
        hbox = QHBoxLayout()
        hbox.addWidget(self.frame_slider)
        self.group_box_frame.setLayout(hbox)
        self.addWidget(self.group_box_frame)

    def enable_controls(self, enabled):
        self.group_box_thresh_method.setEnabled(enabled)
        self.group_box_threshold.setEnabled(enabled)
        self.group_box_block_size.setEnabled(enabled)
        self.group_box_c.setEnabled(enabled)


class RoiPanel(QVBoxLayout):
    def __init__(self):
        super().__init__()

        # List
        self.roi_selector = QListWidget()
        self.roi_selector.setSelectionMode(QAbstractItemView.SingleSelection)

        # Buttons
        hbox_button = QHBoxLayout()
        self.button_add = QPushButton("Add ROI <Ctrl+A>")
        hbox_button.addWidget(self.button_add)
        self.button_remove = QPushButton("Remove ROI <Ctrl+R>")
        hbox_button.addWidget(self.button_remove)

        vbox = QVBoxLayout()
        vbox.addWidget(self.roi_selector)
        vbox.addLayout(hbox_button)
        group_box_roi_selection = QGroupBox("ROI Selection")
        group_box_roi_selection.setLayout(vbox)
        self.addWidget(group_box_roi_selection)


def block_size_to_int(block_size):
    return (block_size - 1) // 2


def int_to_block_size(i):
    return i * 2 + 1


class MainWindow(QMainWindow):
    def __init__(self, video_path: str):
        super().__init__()
        self.setWindowTitle(f"ROIs selector \"{os.path.realpath(video_path)}\"")
        self.video_path = video_path
        self._video_capture = cv2.VideoCapture(video_path)
        self._video_length = get_video_length(self._video_capture)
        self._frame = None
        self._frame_roi = None
        self._adaptive_thresh = DEFAULT_ADAPTIVE_THRESH
        self._auto_thresh = DEFAULT_AUTO_THRESH
        self._adaptive_method = DEFAULT_ADAPTIVE_METHOD

        self._threshold = DEFAULT_THRESHOLD  # cv2.threshold
        self._block_size = DEFAULT_BLOCK_SIZE  # cv2.adaptiveThreshold
        self._C = DEFAULT_C  # cv2.adaptiveThreshold

        self._result_dict = dict()
        self.src_scale_factor = 1.0

        self.layout = QVBoxLayout()

        hbox_display = QHBoxLayout()
        # Source image
        self.roi_widget = RoiWidget()
        self.roi_widget.setScaledContents(True)
        # self.src_scroll_area = QScrollArea()
        self.src_scroll_area = ZoomScrollArea()
        self.src_scroll_area.setWidget(self.roi_widget)
        group_box_src = QGroupBox("Source Image <Ctrl+Scroll to zoom-in/out>")
        hbox = QHBoxLayout()
        hbox.addWidget(self.src_scroll_area)
        group_box_src.setLayout(hbox)
        hbox_display.addWidget(group_box_src, stretch=5)

        vbox_bin_cell = QVBoxLayout()
        # Binarization
        self.bin_display = QLabel()
        self.bin_display.setScaledContents(True)
        # self.bin_display.setPixmap(self.qpixmap)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.bin_display)
        group_box_bin = QGroupBox("Binarization (Preview)")
        hbox = QHBoxLayout()
        hbox.addWidget(scroll_area)
        group_box_bin.setLayout(hbox)
        vbox_bin_cell.addWidget(group_box_bin)

        # Cell Area
        self.cell_display = QLabel()
        self.cell_display.setScaledContents(True)
        # self.cell_display.setPixmap(self.qpixmap)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.cell_display)
        group_box_cell = QGroupBox("Cell Area (Preview)")
        hbox = QHBoxLayout()
        hbox.addWidget(scroll_area)
        group_box_cell.setLayout(hbox)
        vbox_bin_cell.addWidget(group_box_cell)
        #
        hbox_display.addLayout(vbox_bin_cell, stretch=1)
        self.layout.addLayout(hbox_display, stretch=5)

        # Panels
        hbox_roi_ctrl = QHBoxLayout()
        self.roi_panel = RoiPanel()
        hbox_roi_ctrl.addLayout(self.roi_panel)
        self.control_panel = ControlPanel()
        hbox_roi_ctrl.addLayout(self.control_panel)
        self.layout.addLayout(hbox_roi_ctrl, stretch=1)

        # End selection button
        self.button_ok = QPushButton("End selection")
        self.layout.addWidget(self.button_ok)

        # Shortcuts
        action = QAction("Add ROI", self)
        action.triggered.connect(self.on_clicked_add)
        action.setShortcut(QKeySequence("Ctrl+A"))
        self.addAction(action)

        action = QAction("Remove ROI", self)
        action.triggered.connect(self.on_clicked_remove)
        action.setShortcut(QKeySequence("Ctrl+R"))
        self.addAction(action)

        # Callbacks
        self.roi_widget.RoiChanged.connect(self.on_roi_changed)
        self.roi_panel.button_add.clicked.connect(self.on_clicked_add)
        self.roi_panel.button_remove.clicked.connect(self.on_clicked_remove)
        self.roi_panel.roi_selector.itemSelectionChanged.connect(self.on_item_selection_changed)
        self.control_panel.radio_manual.toggled.connect(self.on_radio_button_toggled_manual)
        self.control_panel.radio_auto.toggled.connect(self.on_radio_button_toggled_auto)
        self.control_panel.radio_adap_mean.toggled.connect(self.on_radio_button_toggled_adap_mean)
        self.control_panel.radio_adap_gaussian.toggled.connect(self.on_radio_button_toggled_adap_gaussian)
        self.control_panel.frame_slider.valueChanged.connect(self.on_value_changed_frame)
        self.control_panel.threshold_slider.valueChanged.connect(self.on_value_changed_threshold)
        self.control_panel.block_size_slider.valueChanged.connect(self.on_value_changed_block_size)
        self.control_panel.c_slider.valueChanged.connect(self.on_value_changed_c)
        self.button_ok.clicked.connect(self.on_clicked_end)

        # Defaults
        # self.control_panel.radio_adap_mean.setChecked(True)
        self.control_panel.radio_auto.setChecked(True)
        self.control_panel.frame_slider.setMaximum(self._video_length)

        self.control_panel.threshold_slider.setMaximum(255)
        self.control_panel.threshold_slider.setValue(self._threshold)

        self.control_panel.block_size_slider.setMaximum(block_size_to_int(13))
        self.control_panel.block_size_slider.setValue(block_size_to_int(self._block_size))
        self.control_panel.block_size_slider.setMinimum(block_size_to_int(3))

        self.control_panel.c_slider.setMaximum(30)
        self.control_panel.c_slider.setValue(self._C)
        self.control_panel.enable_controls(False)

        self._load_source(0)

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)
        self.showMaximized()

    def _new_current(self):
        current_name = self.roi_widget.get_current()
        if current_name:
            current = dict(adaptive_thresh=DEFAULT_ADAPTIVE_THRESH,
                           auto_thresh=DEFAULT_AUTO_THRESH,
                           adaptive_method=DEFAULT_ADAPTIVE_METHOD,
                           threshold=DEFAULT_THRESHOLD,
                           block_size=DEFAULT_BLOCK_SIZE,
                           C=DEFAULT_C
                           )
            self._result_dict[current_name] = current
            print("[GUI] new_current")
            pprint(self._result_dict[current_name])

    def _save_current(self):
        current_name = self.roi_widget.get_current()
        if current_name:
            current = dict(adaptive_thresh=self._adaptive_thresh,
                           auto_thresh=self._auto_thresh,
                           adaptive_method=self._adaptive_method,
                           threshold=self._threshold,
                           block_size=self._block_size,
                           C=self._C
                           )
            self._result_dict[current_name] = current
            print("[GUI] save_current")
            pprint(self._result_dict[current_name])

    def _load_current(self):
        current_name = self.roi_widget.get_current()
        if current_name:
            saved = self._result_dict[current_name]
            self._adaptive_thresh = saved["adaptive_thresh"]
            self._auto_thresh = saved["auto_thresh"]
            self._adaptive_method = saved["adaptive_method"]
            self._threshold = saved["threshold"]
            self._block_size = saved["block_size"]
            self._C = saved["C"]
            # Refresh radio buttons and sliders.
            if self._adaptive_thresh:
                self.control_panel.radio_adap_mean.setChecked(True)
            elif self._auto_thresh:
                self.control_panel.radio_auto.setChecked(True)
            else:
                self.control_panel.radio_manual.setChecked(True)
            self.control_panel.threshold_slider.setValue(self._threshold)
            self.control_panel.block_size_slider.setValue(block_size_to_int(self._block_size))
            self.control_panel.c_slider.setValue(self._C)
            print("[GUI] load_current")
            pprint(saved)

    def _remove_current(self):
        current_name = self.roi_widget.get_current()
        if current_name:
            print("[GUI] remove_current")
            pprint(self._result_dict[current_name])
            del self._result_dict[current_name]

    def _load_source(self, val):
        set_frame_position(self._video_capture, val)
        ret, frame = self._video_capture.read()
        if ret:
            self._frame = frame
            h, w, c = frame.shape
            self.roi_widget.setPixmap(cv2_to_qpixmap(frame))
            new_size = QSize(w, h) * self.src_scroll_area.get_scale_factor()
            self.roi_widget.resize(new_size)
        else:
            self.roi_widget.clear()
        self._load_preview()

    def _load_preview(self):
        if not (self.bin_display and self.cell_display):
            return
        roi = self.roi_widget.get_roi(self.roi_widget.get_current())
        if roi.isEmpty():
            self.bin_display.clear()
            self.cell_display.clear()
        else:
            x, y, w, h = roi.x(), roi.y(), roi.width(), roi.height()
            self._frame_roi = self._frame[y:y + h, x:x + w]

            frame_roi = self._frame_roi.copy()
            h, w, c = frame_roi.shape
            frame_roi_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
            if APPLY_GAUSSIAN_BLUR:
                frame_blur = cv2.GaussianBlur(frame_roi_gray, (GAUSSIAN_BLUR_KSIZE, GAUSSIAN_BLUR_KSIZE),
                                              GAUSSIAN_BLUR_SIGMA)
            else:
                frame_blur = frame_roi_gray
            if self._adaptive_thresh:
                frame_bin = cv2.adaptiveThreshold(frame_blur, 255, self._adaptive_method, THRESH_TYPE,
                                                  self._block_size, self._C)
            elif self._auto_thresh:
                threshold, frame_bin = cv2.threshold(frame_blur, 0, 255, THRESH_TYPE | cv2.THRESH_OTSU)
            else:
                _, frame_bin = cv2.threshold(frame_blur, self._threshold, 255, THRESH_TYPE)

            self.bin_display.setPixmap(cv2_to_qpixmap(frame_bin))
            self.bin_display.resize(w * PREVIEW_SCALE, h * PREVIEW_SCALE)

            contours, hierarchy = cv2.findContours(frame_bin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
            # Filter out the contours that unlikely to be a circle
            contours = [contours_i for contours_i in contours if len(contours_i) > 4]
            # Show message if no contours found
            if len(contours) == 0:
                # print("No contour found! Please adjust parameters.")
                self.cell_display.setText("No contour found! \nPlease adjust parameters.")
                self.cell_display.resize(256, 48)
            else:
                # Get all points from contours
                ptr_list = np.concatenate(contours)
                # Find the convex hull object for all points in contours
                hull_list = [cv2.convexHull(ptr_list)]
                frame_contours = cv2.drawContours(frame_roi, contours=hull_list, contourIdx=-1, color=(0, 0, 255),
                                                  thickness=-1)
                self.cell_display.setPixmap(cv2_to_qpixmap(frame_contours))
                self.cell_display.resize(w * PREVIEW_SCALE, h * PREVIEW_SCALE)

    # def _scale_image(self, factor):
    #     if self._frame is not None:
    #         height, width, channel = self._frame.shape
    #         self.roi_widget.resize(int(factor * width), int(factor * height))
    #         adjust_scrollbar(self.src_scroll_area.horizontalScrollBar(), factor)
    #         adjust_scrollbar(self.src_scroll_area.verticalScrollBar(), factor)

    # def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
    #     if event.modifiers() == Qt.ControlModifier:
    #         if event.angleDelta().y() > 0:
    #             self.src_scale_factor += 0.1
    #         else:
    #             self.src_scale_factor -= 0.1
    #         self.src_scale_factor = max(min(self.src_scale_factor, 8.0), 1.0)
    #         print(self.src_scale_factor)
    #         self._scale_image(self.src_scale_factor)

    def on_roi_changed(self):
        self._load_preview()

    def on_value_changed_frame(self, val):
        self.control_panel.group_box_frame.setTitle(f"Frame {val}")
        self._load_source(val)

    def on_value_changed_threshold(self, val):
        self.control_panel.group_box_threshold.setTitle(f"Threshold {val}")
        self._threshold = val
        self._load_preview()
        self._save_current()

    def on_value_changed_block_size(self, val):
        val = int_to_block_size(val)
        self.control_panel.group_box_block_size.setTitle(f"Block Size {val}x{val}")
        self._block_size = val
        self._load_preview()
        self._save_current()

    def on_value_changed_c(self, val):
        self.control_panel.group_box_c.setTitle(f"C {val}")
        self._C = val
        self._load_preview()
        self._save_current()

    def on_radio_button_toggled_manual(self, selected):
        # Manual threshold
        if selected:
            self.control_panel.group_box_threshold.setVisible(True)
            self.control_panel.group_box_block_size.setVisible(False)
            self.control_panel.group_box_c.setVisible(False)
            self._adaptive_thresh = False
            self._auto_thresh = False
            self._load_preview()
            self._save_current()

    def on_radio_button_toggled_auto(self, selected):
        # Auto threshold (Otsu's method)
        if selected:
            self.control_panel.group_box_threshold.setVisible(False)
            self.control_panel.group_box_block_size.setVisible(False)
            self.control_panel.group_box_c.setVisible(False)
            self._adaptive_thresh = False
            self._auto_thresh = True
            self._load_preview()
            self._save_current()

    def on_radio_button_toggled_adap(self):
        # Adaptive threshold
        self.control_panel.group_box_threshold.setVisible(False)
        self.control_panel.group_box_block_size.setVisible(True)
        self.control_panel.group_box_c.setVisible(True)
        self._adaptive_thresh = True
        self._auto_thresh = False
        self._load_preview()
        self._save_current()

    def on_radio_button_toggled_adap_mean(self, selected):
        if selected:
            self._adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
            self.on_radio_button_toggled_adap()

    def on_radio_button_toggled_adap_gaussian(self, selected):
        if selected:
            self._adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            self.on_radio_button_toggled_adap()

    def on_clicked_end(self):
        gui_result = self.get_result()
        if gui_result:
            print("[GUI] end_selection")
            pprint(gui_result)
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            index = len(
                list(filter(lambda p: p.startswith("params") and p.endswith(".txt"), os.listdir(output_dir))))
            with open(os.path.join(output_dir, "params_{:02d}.txt".format(index + 1)), "w") as f:
                pprint(gui_result, f)
            thread = threading.Thread(target=main,
                                      kwargs=dict(video_path=self.video_path,
                                                  output_dir=output_dir,
                                                  gui_result=gui_result))
            thread.start()
            QMessageBox.information(self, "Processing", "Check console for output.")
        else:
            QMessageBox.critical(self, "Error", "Nothing to process!")

    def on_clicked_add(self):
        name, ret = QInputDialog.getText(self, "Add ROI", "Enter a name:")
        if ret:
            if name:
                success = self.roi_widget.add(name)
                if success:
                    self._new_current()
                    pixmap = QPixmap(16, 16)
                    pixmap.fill(QColor().fromRgb(*self.roi_widget.get_rgb(name)))
                    icon = QIcon(pixmap)
                    list_item = QListWidgetItem(icon, name)
                    self.roi_panel.roi_selector.addItem(list_item)
                    self.roi_panel.roi_selector.setCurrentItem(list_item)
                    self.control_panel.enable_controls(True)
                else:
                    QMessageBox.critical(self, "Error", "Name already exists!")
            else:
                QMessageBox.critical(self, "Error", "Invalid name!")

    def on_clicked_remove(self):
        self._remove_current()
        cur_item = self.roi_panel.roi_selector.currentItem()
        if cur_item:
            name = cur_item.text()
            row = self.roi_panel.roi_selector.row(cur_item)
            success = self.roi_widget.remove(name)
            if success:
                self.roi_panel.roi_selector.takeItem(row)
                if self.roi_panel.roi_selector.currentItem() is None:
                    # Last item
                    self.control_panel.enable_controls(False)

    def on_item_selection_changed(self):
        cur_item = self.roi_panel.roi_selector.currentItem()
        if cur_item:
            name = cur_item.text()
            success = self.roi_widget.switch_to(name)
            if success:
                self._load_current()

    def get_result(self):
        ret_roi = self.roi_widget.get_result()
        ret_param = self._result_dict
        assert ret_roi.keys() == ret_param.keys()
        combined = dict()
        for key in ret_roi.keys():
            combined[key] = {**ret_roi[key], **ret_param[key]}
            # Convert ROI to tuple
            roi = combined[key]["roi"]
            combined[key]["roi"] = roi.x(), roi.y(), roi.width(), roi.height()
            # Remove unnecessary entry
            del combined[key]["rgb"]
            if combined[key]["adaptive_thresh"]:
                del combined[key]["threshold"]
            elif combined[key]["auto_thresh"]:
                del combined[key]["adaptive_method"]
                del combined[key]["threshold"]
                del combined[key]["block_size"]
                del combined[key]["C"]
            else:
                del combined[key]["adaptive_method"]
                del combined[key]["block_size"]
                del combined[key]["C"]
        return combined


def get_video_path():
    video_path = QFileDialog.getOpenFileName(caption="Open Video", filter="Videos (*.mp4 *.avi)")[0]
    return video_path


if __name__ == "__main__":
    freeze_support()
    app = QApplication(sys.argv)
    video_path = get_video_path()
    if video_path:
        print(f"[Video] {video_path}")
        window = MainWindow(video_path)
        window.show()
        app.exec()
