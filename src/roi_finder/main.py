"""Interactive XRF ROI selector and scan-plan generator.

Replace ``load_xrf_data_for_scan`` and ``send_scan_plans`` with the beamline
implementations.  Everything else is intentionally independent of the data
acquisition framework.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import sys

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
from hxntools.CompositeBroker import db
from hxntools.scan_info import get_scan_positions

from xrf_utils import get_all_xrf_roi_data, get_scan_details


@dataclass
class XRFScan:
    """XRF stack is shaped (element, y, x); pixel_size_um is (x, y)."""

    stack: np.ndarray
    element_names: list[str]
    pixel_size_um: tuple[float, float] = (0.25, 0.25)
    origin_um: tuple[float, float] = (0.0, 0.0)


PHANTOM_SCAN_NUMBER = "0000"


def _make_phantom_xrf_scan() -> XRFScan:
    """Build a reproducible three-element phantom stack for testing the GUI."""
    rng = np.random.default_rng(0)
    y, x = np.mgrid[:360, :480]
    stack = []
    for centers in [[(130, 120, 42), (315, 245, 64)],
                    [(165, 150, 58), (330, 242, 35)],
                    [(105, 255, 30), (365, 100, 43)]]:
        image = rng.normal(3, 1.0, x.shape)
        for cx, cy, width in centers:
            image += 90 * np.exp(-((x-cx)**2 + (y-cy)**2) / (2 * width**2))
        stack.append(np.clip(image, 0, None))
    return XRFScan(np.asarray(stack), ["Fe_K", "Cr_K", "Mn_K"])


def _get_scan_geometry(details: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    """Derive (pixel_size_um, origin_um) from the real scan metadata."""
    scan_cfg = details.get("scan") if isinstance(details.get("scan"), dict) else {}
    scan_input = scan_cfg.get("scan_input")
    shape = scan_cfg.get("shape") or details.get("shape")

    if isinstance(scan_input, (list, tuple)) and len(scan_input) >= 6:
        start1 = float(scan_input[0]); end1 = float(scan_input[1]); num1 = int(scan_input[2])
        start2 = float(scan_input[3]); end2 = float(scan_input[4]); num2 = int(scan_input[5])
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            num1 = int(shape[0]) if int(shape[0]) > 0 else num1
            num2 = int(shape[1]) if int(shape[1]) > 0 else num2
        x_step = (end1 - start1) / max(num1 - 1, 1)
        y_step = (end2 - start2) / max(num2 - 1, 1)
        return (x_step, y_step), (start1, start2)

    if not all(key in details for key in ("scan_start1", "scan_end1", "num1")):
        return (0.25, 0.25), (0.0, 0.0)

    x_step = (details["scan_end1"] - details["scan_start1"]) / max(int(details["num1"]) - 1, 1)
    origin_x = details["scan_start1"]
    if all(key in details for key in ("scan_start2", "scan_end2", "num2")):
        y_step = (details["scan_end2"] - details["scan_start2"]) / max(int(details["num2"]) - 1, 1)
        origin_y = details["scan_start2"]
    else:
        y_step, origin_y = x_step, origin_x
    return (x_step, y_step), (origin_x, origin_y)


def load_xrf_data_for_scan(scan_number: str) -> XRFScan:
    """HOOK: Load data for *scan_number* and return an ``XRFScan``.

    Enter "0000" as the scan number to load a phantom (synthetic) dataset for
    testing the GUI before it is connected to databroker/exported XRF files.
    """
    if scan_number.strip() == PHANTOM_SCAN_NUMBER:
        return _make_phantom_xrf_scan()

    else:
        print(f"Attempting to load scan {scan_number!r}...")
        hdr = db[int(scan_number)]
        xrf_stack, element_names = get_all_xrf_roi_data(hdr)
        pixel_size_um, origin_um = _get_scan_geometry(get_scan_details(hdr))
        return XRFScan(xrf_stack, element_names, pixel_size_um, origin_um)

def send_scan_plans(plans: list[dict]) -> None:
    """HOOK: Submit *plans* to QueueServer (or write them to your queue)."""
    # Example: REManagerAPI(...).item_add({"name": "fly2d", "args": ...})
    print("Plans ready for submission:")
    for plan in plans:
        print(plan)


class RealCoordinateAxis(pg.AxisItem):
    """Render axis tick labels in physical coordinates while keeping pixel-space data."""

    def __init__(self, orientation: str, pixel_size_um: tuple[float, float] = (0.25, 0.25), origin_um: tuple[float, float] = (0.0, 0.0)):
        super().__init__(orientation=orientation)
        self.pixel_size_um = pixel_size_um
        self.origin_um = origin_um

    def set_scan_geometry(self, pixel_size_um: tuple[float, float], origin_um: tuple[float, float]) -> None:
        self.pixel_size_um = pixel_size_um
        self.origin_um = origin_um

    def tickStrings(self, values, scale, spacing):
        axis_index = 0 if self.orientation == "bottom" else 1
        labels = []
        for value in values:
            world_coord = self.origin_um[axis_index] + value * self.pixel_size_um[axis_index]
            labels.append(f"{world_coord:.2f}")
        return labels


class ROIViewBox(pg.ViewBox):
    """A ViewBox that creates a movable/resizable RectROI with a right-drag."""

    roiCreated = QtCore.pyqtSignal(object)

    def mouseDragEvent(self, event, axis=None):  # noqa: N802 (pyqtgraph API)
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            event.accept()
            start = self.mapSceneToView(event.buttonDownScenePos())
            current = self.mapSceneToView(event.scenePos())
            x0, y0 = min(start.x(), current.x()), min(start.y(), current.y())
            w, h = abs(current.x() - start.x()), abs(current.y() - start.y())
            if event.isStart():
                self._drawing_roi = pg.RectROI((x0, y0), (0.1, 0.1),
                                                pen=pg.mkPen("#00d8ff", width=2),
                                                removable=True)
                self.addItem(self._drawing_roi)
            if hasattr(self, "_drawing_roi"):
                self._drawing_roi.setPos((x0, y0), update=False)
                self._drawing_roi.setSize((max(w, 0.1), max(h, 0.1)), update=False)
            if event.isFinish() and hasattr(self, "_drawing_roi"):
                roi = self._drawing_roi
                del self._drawing_roi
                if w >= 2 and h >= 2:
                    self.roiCreated.emit(roi)
                else:
                    self.removeItem(roi)
            return
        super().mouseDragEvent(event, axis=axis)


class ROIScanPlanner(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XRF ROI Selector & Scan Planner")
        self.resize(1320, 820)
        self.scan: XRFScan | None = None
        self.rois: list[pg.RectROI] = []
        self._build_ui()
        self.statusBar().showMessage("Enter a scan number and click Load XRF data.")

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        controls = QtWidgets.QWidget()
        controls.setMaximumWidth(330)
        form = QtWidgets.QVBoxLayout(controls)

        load_group = QtWidgets.QGroupBox("Input")
        load_form = QtWidgets.QFormLayout(load_group)
        self.scan_number = QtWidgets.QLineEdit()
        self.scan_number.setPlaceholderText('e.g. 123456 or -1 (use "0000" for phantom test data)')
        self.scan_number.setValidator(QtGui.QIntValidator(-999999999, 999999999, self))
        self.load_button = QtWidgets.QPushButton("Load XRF data")
        self.load_button.clicked.connect(self.load_data)
        load_form.addRow("Scan number", self.scan_number)
        load_form.addRow(self.load_button)
        form.addWidget(load_group)

        selection = QtWidgets.QGroupBox("ROI selection")
        selection_layout = QtWidgets.QVBoxLayout(selection)
        self.auto_radio = QtWidgets.QRadioButton("Auto — find ROIs")
        self.manual_radio = QtWidgets.QRadioButton("Manual — draw ROIs")
        self.auto_radio.setChecked(True)
        self.auto_button = QtWidgets.QPushButton("Find ROIs")
        self.auto_button.clicked.connect(self.find_rois)
        self.clear_button = QtWidgets.QPushButton("Clear ROIs")
        self.clear_button.clicked.connect(self.clear_rois)
        selection_layout.addWidget(self.auto_radio)
        selection_layout.addWidget(self.manual_radio)
        selection_layout.addWidget(self.auto_button)
        selection_layout.addWidget(self.clear_button)
        selection_layout.addWidget(QtWidgets.QLabel("Manual: right-drag on image to draw.\nDrag handles to refine; right-click ROI to remove."))
        form.addWidget(selection)

        parameters = QtWidgets.QGroupBox("Fine-scan parameters")
        params = QtWidgets.QFormLayout(parameters)
        self.step_um = QtWidgets.QDoubleSpinBox(); self.step_um.setRange(0.001, 100); self.step_um.setValue(0.05); self.step_um.setSuffix(" µm")
        self.dwell_s = QtWidgets.QDoubleSpinBox(); self.dwell_s.setRange(0.0001, 100); self.dwell_s.setDecimals(4); self.dwell_s.setValue(0.01); self.dwell_s.setSuffix(" s")
        self.padding_um = QtWidgets.QDoubleSpinBox(); self.padding_um.setRange(0, 100); self.padding_um.setValue(0.25); self.padding_um.setSuffix(" µm")
        params.addRow("Step size", self.step_um)
        params.addRow("Dwell", self.dwell_s)
        params.addRow("ROI padding", self.padding_um)
        form.addWidget(parameters)

        self.generate_button = QtWidgets.QPushButton("Generate scan plans")
        self.generate_button.clicked.connect(self.generate_plans)
        self.send_button = QtWidgets.QPushButton("Send scans")
        self.send_button.clicked.connect(self.send_plans)
        self.send_button.setEnabled(False)
        form.addWidget(self.generate_button)
        form.addWidget(self.send_button)

        info_group = QtWidgets.QGroupBox("Scan info")
        info_layout = QtWidgets.QVBoxLayout(info_group)
        self.scan_info_box = QtWidgets.QTextEdit()
        self.scan_info_box.setReadOnly(True)
        self.scan_info_box.setMinimumHeight(140)
        self.scan_info_box.setPlainText("No scan loaded.\nUse a valid integer scan ID or 0000 for phantom data.")
        info_layout.addWidget(self.scan_info_box)
        form.addWidget(info_group)
        form.addStretch(1)
        layout.addWidget(controls)

        right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        image_box = QtWidgets.QWidget(); image_layout = QtWidgets.QVBoxLayout(image_box)
        image_controls = QtWidgets.QHBoxLayout()
        image_controls.addWidget(QtWidgets.QLabel("Displayed element:"))
        self.element_combo = QtWidgets.QComboBox()
        self.element_combo.currentIndexChanged.connect(self.show_element)
        image_controls.addWidget(self.element_combo)
        image_controls.addStretch(1)
        image_layout.addLayout(image_controls)
        self.view_box = ROIViewBox(lockAspect=True, invertY=True)
        self.view_box.roiCreated.connect(self.add_roi)
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.view_box.addItem(self.image_item)
        self.x_axis = RealCoordinateAxis("bottom")
        self.y_axis = RealCoordinateAxis("left")
        self.plot = pg.PlotWidget(viewBox=self.view_box, enableMenu=False,
                                 axisItems={"bottom": self.x_axis, "left": self.y_axis})
        self.plot.setLabel("bottom", "x", units="µm")
        self.plot.setLabel("left", "y", units="µm")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        image_layout.addWidget(self.plot, stretch=1)
        right.addWidget(image_box)

        plan_box = QtWidgets.QWidget(); plan_layout = QtWidgets.QVBoxLayout(plan_box)
        plan_layout.addWidget(QtWidgets.QLabel("Generated scan plans"))
        self.plan_table = QtWidgets.QTableWidget(0, 7)
        self.plan_table.setHorizontalHeaderLabels(["ROI", "x start", "x stop", "y start", "y stop", "points (x, y)", "est. time"])
        self.plan_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.plan_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        plan_layout.addWidget(self.plan_table)
        right.addWidget(plan_box)
        right.setSizes([560, 230])
        layout.addWidget(right, stretch=1)

    def _update_scan_info(self):
        if self.scan is None:
            self.scan_info_box.setPlainText("No scan loaded.\nUse a valid integer scan ID or 0000 for phantom data.")
            return

        shape = self.scan.stack.shape
        pixel_size_x, pixel_size_y = self.scan.pixel_size_um
        origin_x, origin_y = self.scan.origin_um
        text = (
            f"Shape: {shape[2]} × {shape[1]} px\n"
            f"Elements: {shape[0]}\n"
            f"Element names: {', '.join(self.scan.element_names)}\n\n"
            f"Pixel size: ({pixel_size_x:.3f}, {pixel_size_y:.3f}) µm\n"
            f"Origin: ({origin_x:.3f}, {origin_y:.3f}) µm\n"
            f"Field of view: ({shape[2] * pixel_size_x:.3f}, {shape[1] * pixel_size_y:.3f}) µm"
        )
        self.scan_info_box.setPlainText(text)

    def load_data(self):
        try:
            raw_value = self.scan_number.text().strip()
            if not raw_value or (not raw_value.lstrip("-").isdigit()):
                raise ValueError("Scan number must be an integer.")
            self.scan = load_xrf_data_for_scan(raw_value)
            if self.scan.stack.ndim != 3 or self.scan.stack.shape[0] != len(self.scan.element_names):
                raise ValueError("Expected XRF stack shape (n_elements, y, x) and matching names.")
            self.x_axis.set_scan_geometry(self.scan.pixel_size_um, self.scan.origin_um)
            self.y_axis.set_scan_geometry(self.scan.pixel_size_um, self.scan.origin_um)
            self.element_combo.blockSignals(True)
            self.element_combo.clear(); self.element_combo.addItems(self.scan.element_names)
            self.element_combo.blockSignals(False)
            self.clear_rois(); self.show_element(0)
            self._update_scan_info()
            self.statusBar().showMessage(f"Loaded {self.scan.stack.shape[0]} elements; image size {self.scan.stack.shape[2]} × {self.scan.stack.shape[1]} px.")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Could not load XRF data", str(exc))

    def show_element(self, index: int):
        if self.scan is None or index < 0:
            return
        self.image_item.setImage(self.scan.stack[index], autoLevels=True)
        self.x_axis.set_scan_geometry(self.scan.pixel_size_um, self.scan.origin_um)
        self.y_axis.set_scan_geometry(self.scan.pixel_size_um, self.scan.origin_um)
        self.view_box.autoRange()
        self._update_scan_info()

    def add_roi(self, roi: pg.RectROI):
        self.rois.append(roi)
        roi.sigRemoveRequested.connect(lambda item=roi: self.remove_roi(item))
        roi.sigRegionChangeFinished.connect(lambda: self.send_button.setEnabled(False))
        self.manual_radio.setChecked(True)
        self.send_button.setEnabled(False)
        self.statusBar().showMessage(f"ROI {len(self.rois)} added.")

    def remove_roi(self, roi):
        if roi in self.rois:
            self.rois.remove(roi)
            self.view_box.removeItem(roi)
            self.send_button.setEnabled(False)

    def clear_rois(self):
        for roi in self.rois.copy():
            self.view_box.removeItem(roi)
        self.rois.clear()
        self.plan_table.setRowCount(0)
        self.send_button.setEnabled(False)

    def find_rois(self):
        """Simple connected-component placeholder; replace with your CV method."""
        if self.scan is None:
            self.statusBar().showMessage("Load XRF data first.")
            return
        self.clear_rois()
        image = self.scan.stack[self.element_combo.currentIndex()]
        threshold = np.nanpercentile(image, 92)
        mask = np.isfinite(image) & (image >= threshold)
        for x, y, w, h in self._component_boxes(mask, min_pixels=40):
            roi = pg.RectROI((x, y), (w, h), pen=pg.mkPen("#ffb000", width=2), removable=True)
            self.view_box.addItem(roi); self.add_roi(roi)
        self.auto_radio.setChecked(True)
        self.statusBar().showMessage(f"Found {len(self.rois)} ROI(s) from {self.element_combo.currentText()}.")

    @staticmethod
    def _component_boxes(mask: np.ndarray, min_pixels: int) -> list[tuple[int, int, int, int]]:
        """4-connected components without requiring scipy/opencv."""
        visited = np.zeros_like(mask, dtype=bool); height, width = mask.shape; boxes = []
        for sy, sx in np.argwhere(mask):
            if visited[sy, sx]: continue
            todo = deque([(sy, sx)]); visited[sy, sx] = True; count = 0; xs = []; ys = []
            while todo:
                y, x = todo.popleft(); count += 1; xs.append(x); ys.append(y)
                for ny, nx in ((y-1,x), (y+1,x), (y,x-1), (y,x+1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True; todo.append((ny, nx))
            if count >= min_pixels:
                boxes.append((min(xs), min(ys), max(xs)-min(xs)+1, max(ys)-min(ys)+1))
        return boxes

    def generate_plans(self):
        if self.scan is None or not self.rois:
            self.statusBar().showMessage("Load data and select at least one ROI.")
            return
        self.plans = []
        px, py = self.scan.pixel_size_um; ox, oy = self.scan.origin_um; pad = self.padding_um.value()
        ny, nx = self.scan.stack.shape[1:]
        for i, roi in enumerate(self.rois, start=1):
            pos, size = roi.pos(), roi.size()
            x0 = max(0, pos.x()); x1 = min(nx, pos.x()+size.x())
            y0 = max(0, pos.y()); y1 = min(ny, pos.y()+size.y())
            plan = {"roi": i, "x_start_um": ox + x0*px-pad, "x_stop_um": ox + x1*px+pad,
                    "y_start_um": oy + y0*py-pad, "y_stop_um": oy + y1*py+pad,
                    "step_um": self.step_um.value(), "dwell_s": self.dwell_s.value()}
            plan["num_x"] = max(2, round((plan["x_stop_um"]-plan["x_start_um"])/plan["step_um"])+1)
            plan["num_y"] = max(2, round((plan["y_stop_um"]-plan["y_start_um"])/plan["step_um"])+1)
            plan["estimated_s"] = plan["num_x"]*plan["num_y"]*plan["dwell_s"]
            self.plans.append(plan)
        self._show_plans(); self.send_button.setEnabled(True)
        self.statusBar().showMessage(f"Generated {len(self.plans)} scan plan(s).")

    def _show_plans(self):
        self.plan_table.setRowCount(len(self.plans))
        for row, p in enumerate(self.plans):
            values = [p["roi"], f'{p["x_start_um"]:.3f} µm', f'{p["x_stop_um"]:.3f} µm',
                      f'{p["y_start_um"]:.3f} µm', f'{p["y_stop_um"]:.3f} µm',
                      f'{p["num_x"]}, {p["num_y"]}', f'{p["estimated_s"]/60:.1f} min']
            for col, value in enumerate(values): self.plan_table.setItem(row, col, QtWidgets.QTableWidgetItem(str(value)))

    def send_plans(self):
        try:
            send_scan_plans(self.plans)
            self.statusBar().showMessage(f"Submitted {len(self.plans)} plan(s).")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Could not send plans", str(exc))


if __name__ == "__main__":
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    window = ROIScanPlanner(); window.show()
    sys.exit(app.exec())
