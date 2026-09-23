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

from qserver_utils import send_fly2d_recover_and_scan
from xrf_utils import get_all_xrf_roi_data, get_scan_details



@dataclass
class XRFScan:
    """XRF stack is shaped (element, y, x); pixel_size_um is (x, y)."""

    stack: np.ndarray
    element_names: list[str]
    pixel_size_um: tuple[float, float] = (0.25, 0.25)
    origin_um: tuple[float, float] = (0.0, 0.0)


PHANTOM_SCAN_NUMBER = "0000"
SCANNER_RANGE_LIMIT_UM = 14.0
MAX_SCAN_POINTS = 64_000
MANUAL_ROI_COLOR = "#ff4fd8"
AUTO_ROI_COLOR = "#ff6b6b"

PLAN_TABLE_COLUMNS = [
    "Use",
    "ROI",
    "x start",
    "x stop",
    "y start",
    "y stop",
    "points (x, y)",
    "est. time",
    "range",
]

DETECTOR_PRESETS = {
    "dets_fast": ["fs", "eiger2", "xspress3"],
    "dets_fast_merlin": ["fs", "xspress3", "merlin1", "eiger2"],
    "dets_fast_fs": ["fs", "xspress3"],
}


def _plan_within_scanner_limits(plan: dict, limit_um: float = SCANNER_RANGE_LIMIT_UM) -> bool:
    return all(
        abs(float(plan[key])) <= limit_um
        for key in ("x_start_um", "x_stop_um", "y_start_um", "y_stop_um")
    )


def _plan_total_points(plan: dict) -> int:
    return int(plan.get("num_x", 0)) * int(plan.get("num_y", 0))


def _plan_within_point_limit(plan: dict, max_points: int = MAX_SCAN_POINTS) -> bool:
    return _plan_total_points(plan) <= max_points


def _plan_is_valid(plan: dict, limit_um: float = SCANNER_RANGE_LIMIT_UM, max_points: int = MAX_SCAN_POINTS) -> bool:
    return _plan_within_scanner_limits(plan, limit_um=limit_um) and _plan_within_point_limit(plan, max_points=max_points)


def _plan_range_status(plan: dict, limit_um: float = SCANNER_RANGE_LIMIT_UM, max_points: int = MAX_SCAN_POINTS) -> str:
    if not _plan_within_scanner_limits(plan, limit_um=limit_um):
        return f"Outside ±{limit_um:.0f} µm"
    if not _plan_within_point_limit(plan, max_points=max_points):
        return f"Too many points (> {max_points:,})"
    return "OK"


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


def _roi_padding_um(roi_size_px: tuple[float, float], pixel_size_um: tuple[float, float], padding_fraction: float) -> tuple[float, float]:
    """Return padding in microns as a fraction of the ROI size."""
    width_px, height_px = roi_size_px
    pad_x_um = max(width_px, 0.0) * pixel_size_um[0] * max(padding_fraction, 0.0)
    pad_y_um = max(height_px, 0.0) * pixel_size_um[1] * max(padding_fraction, 0.0)
    return pad_x_um, pad_y_um


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

def send_scan_plans(plans: list[dict], sid: str | int | None = None,
                   dets: str | list[str] | None = None,
                   mot1: str = "zpssx", mot2: str = "zpssy") -> None:
    """Submit generated ROI plans through the queue-server recovery + fly2d plan."""
    if not plans:
        return
    if sid is None:
        raise ValueError("Scan ID is required to recover the motor positions before the fly scan.")

    invalid_rois = [str(plan.get("roi", "?")) for plan in plans if not _plan_is_valid(plan)]
    if invalid_rois:
        raise ValueError(
            "Selected scan plan(s) violate scanner limits or point limits: " + ", ".join(invalid_rois)
        )

    for plan in plans:
        send_fly2d_recover_and_scan(
            label=f"roi_{plan['roi']}",
            roi_positions=int(sid),
            dets=dets,
            mot1=mot1,
            mot1_s=plan["x_start_um"],
            mot1_e=plan["x_stop_um"],
            mot1_n=plan["num_x"],
            mot2=mot2,
            mot2_s=plan["y_start_um"],
            mot2_e=plan["y_stop_um"],
            mot2_n=plan["num_y"],
            exp_t=plan["dwell_s"],
        )


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
                                                pen=pg.mkPen(MANUAL_ROI_COLOR, width=2),
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
        self.plans: list[dict] = []
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
        self.padding_pct = QtWidgets.QDoubleSpinBox(); self.padding_pct.setRange(0, 100); self.padding_pct.setDecimals(1); self.padding_pct.setSingleStep(0.5); self.padding_pct.setValue(10.0); self.padding_pct.setSuffix(" %")
        self.detector_system = QtWidgets.QComboBox()
        self.detector_system.addItems(["dets_fast", "dets_fast_merlin", "dets_fast_fs"])
        self.detector_system.setCurrentText("dets_fast")
        params.addRow("Step size", self.step_um)
        params.addRow("Dwell", self.dwell_s)
        params.addRow("ROI padding", self.padding_pct)
        params.addRow("Detector system", self.detector_system)
        self.auto_update_table = QtWidgets.QCheckBox("Auto-update table when parameters change")
        self.auto_update_table.setChecked(False)
        params.addRow(self.auto_update_table)
        self.step_um.valueChanged.connect(self._on_scan_param_changed)
        self.dwell_s.valueChanged.connect(self._on_scan_param_changed)
        self.padding_pct.valueChanged.connect(self._on_scan_param_changed)
        form.addWidget(parameters)

        self.generate_button = QtWidgets.QPushButton("Generate scan plans")
        self.generate_button.clicked.connect(self.generate_plans)
        self.update_button = QtWidgets.QPushButton("Update table from params")
        self.update_button.clicked.connect(self.generate_plans)
        self.send_button = QtWidgets.QPushButton("Send scans")
        self.send_button.clicked.connect(self.send_plans)
        self.send_button.setEnabled(False)
        form.addWidget(self.generate_button)
        form.addWidget(self.update_button)
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
        image_display = QtWidgets.QHBoxLayout()
        self.view_box = ROIViewBox(lockAspect=True, invertY=True)
        self.view_box.roiCreated.connect(self.add_roi)
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.view_box.addItem(self.image_item)
        self.hover_label = pg.TextItem(text="", color="w", fill="k")
        self.hover_label.setVisible(False)
        self.view_box.addItem(self.hover_label)
        self.x_axis = RealCoordinateAxis("bottom")
        self.y_axis = RealCoordinateAxis("left")
        self.plot = pg.PlotWidget(viewBox=self.view_box, enableMenu=False,
                                 axisItems={"bottom": self.x_axis, "left": self.y_axis})
        self.plot.setLabel("bottom", "x", units="µm")
        self.plot.setLabel("left", "y", units="µm")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.scene().sigMouseMoved.connect(self._update_hover_coordinates)
        self.image_histogram = pg.HistogramLUTWidget()
        self.image_histogram.setImageItem(self.image_item)
        self.image_histogram.setMaximumWidth(140)
        viridis = pg.colormap.get("viridis")
        self.image_item.setColorMap(viridis)
        self.image_histogram.gradient.setColorMap(viridis)
        image_display.addWidget(self.plot, stretch=1)
        image_display.addWidget(self.image_histogram)
        image_layout.addLayout(image_display, stretch=1)
        right.addWidget(image_box)

        plan_box = QtWidgets.QWidget(); plan_layout = QtWidgets.QVBoxLayout(plan_box)
        plan_layout.addWidget(QtWidgets.QLabel("Generated scan plans"))
        self.plan_table = QtWidgets.QTableWidget(0, len(PLAN_TABLE_COLUMNS))
        self.plan_table.setHorizontalHeaderLabels(PLAN_TABLE_COLUMNS)
        self.plan_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.plan_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.AllEditTriggers)
        self.plan_table.itemChanged.connect(self._on_plan_table_item_changed)
        plan_layout.addWidget(self.plan_table)
        right.addWidget(plan_box)
        right.setSizes([560, 230])
        layout.addWidget(right, stretch=1)

    def _update_hover_coordinates(self, pos):
        if self.scan is None:
            self.hover_label.setVisible(False)
            return
        view_pos = self.view_box.mapSceneToView(pos)
        x_px, y_px = view_pos.x(), view_pos.y()
        if not (0 <= x_px < self.scan.stack.shape[2] and 0 <= y_px < self.scan.stack.shape[1]):
            self.hover_label.setVisible(False)
            return
        px, py = self.scan.pixel_size_um
        ox, oy = self.scan.origin_um
        x_um = ox + x_px * px
        y_um = oy + y_px * py
        self.statusBar().showMessage(f"Cursor: x={x_um:.3f} µm, y={y_um:.3f} µm  |  px=({x_px:.1f}, {y_px:.1f})")
        self.hover_label.setText(f"x={x_um:.2f} µm\ny={y_um:.2f} µm")
        self.hover_label.setPos(x_px + 2, y_px + 2)
        self.hover_label.setVisible(True)

    def _update_scan_info(self):
        if self.scan is None:
            self.scan_info_box.setPlainText("No scan loaded.\nUse a valid integer scan ID or 0000 for phantom data.")
            return

        shape = self.scan.stack.shape
        pixel_size_x, pixel_size_y = self.scan.pixel_size_um
        origin_x, origin_y = self.scan.origin_um
        scan_input = getattr(self, "scan_input", None)
        if scan_input is None:
            scan_input = "n/a"
        text = (
            f"Scan input: {scan_input}\n"
            f"Shape: {shape[2]} × {shape[1]} px\n"
            f"Elements: {shape[0]}\n"
            f"Element names: {', '.join(self.scan.element_names)}\n\n"
            f"Pixel size: ({pixel_size_x:.3f}, {pixel_size_y:.3f}) µm\n"
            f"Origin: ({origin_x:.3f}, {origin_y:.3f}) µm\n"
            f"Field of view: ({shape[2] * pixel_size_x:.3f}, {shape[1] * pixel_size_y:.3f}) µm"
        )
        self.scan_info_box.setPlainText(text)

    def _on_scan_param_changed(self, *_):
        if self.auto_update_table.isChecked():
            self.generate_plans()
        elif self.plan_table.rowCount() > 0:
            self.statusBar().showMessage("Parameters changed. Use Update table from params to refresh the plans.")

    def _plan_selection_states(self) -> list[bool]:
        states: list[bool] = []
        for row in range(self.plan_table.rowCount()):
            item = self.plan_table.item(row, 0)
            states.append(item is not None and item.checkState() == QtCore.Qt.CheckState.Checked)
        return states

    def _make_plan_item(self, text: str, *, editable: bool = False, checkable: bool = False, checked: bool = True) -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem(text)
        flags = QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled
        if editable:
            flags |= QtCore.Qt.ItemFlag.ItemIsEditable
        if checkable:
            flags |= QtCore.Qt.ItemFlag.ItemIsUserCheckable
        item.setFlags(flags)
        if checkable:
            item.setCheckState(QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)
        return item

    @staticmethod
    def _parse_float_item(item: QtWidgets.QTableWidgetItem | None) -> float | None:
        if item is None:
            return None
        text = item.text().replace("µm", "").strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _plan_from_row(self, row: int) -> tuple[dict, list[str]]:
        errors: list[str] = []

        roi_item = self.plan_table.item(row, 1)
        roi_text = roi_item.text().strip() if roi_item is not None else str(row + 1)
        try:
            roi_index = int(roi_text)
        except ValueError:
            roi_index = row + 1
            errors.append(f"ROI {row + 1}: invalid ROI label {roi_text!r}")

        x_start = self._parse_float_item(self.plan_table.item(row, 2))
        x_stop = self._parse_float_item(self.plan_table.item(row, 3))
        y_start = self._parse_float_item(self.plan_table.item(row, 4))
        y_stop = self._parse_float_item(self.plan_table.item(row, 5))
        step_um = self.step_um.value()
        dwell_s = self.dwell_s.value()

        selected_item = self.plan_table.item(row, 0)
        selected = selected_item is not None and selected_item.checkState() == QtCore.Qt.CheckState.Checked

        plan = {
            "roi": roi_index,
            "selected": selected,
            "step_um": step_um,
            "dwell_s": dwell_s,
            "x_start_um": x_start if x_start is not None else 0.0,
            "x_stop_um": x_stop if x_stop is not None else 0.0,
            "y_start_um": y_start if y_start is not None else 0.0,
            "y_stop_um": y_stop if y_stop is not None else 0.0,
        }

        if None in (x_start, x_stop, y_start, y_stop):
            errors.append(f"ROI {roi_index}: one or more coordinate values are invalid")
            plan["num_x"] = 0
            plan["num_y"] = 0
            plan["estimated_s"] = 0.0
            plan["within_limits"] = False
            plan["range_status"] = "Invalid value"
            return plan, errors

        span_x_um = abs(x_stop - x_start)
        span_y_um = abs(y_stop - y_start)
        plan["num_x"] = max(2, round(span_x_um / max(step_um, 1e-12)) + 1)
        plan["num_y"] = max(2, round(span_y_um / max(step_um, 1e-12)) + 1)
        plan["estimated_s"] = plan["num_x"] * plan["num_y"] * dwell_s
        plan["within_limits"] = _plan_is_valid(plan)
        plan["range_status"] = _plan_range_status(plan)
        return plan, errors

    def _refresh_plan_row(self, row: int) -> None:
        if row < 0 or row >= self.plan_table.rowCount():
            return

        plan, errors = self._plan_from_row(row)
        status_text = "; ".join(errors) if errors else plan["range_status"]
        is_valid = plan.get("within_limits", False) and not errors
        blocker = QtCore.QSignalBlocker(self.plan_table)
        try:
            points_item = self.plan_table.item(row, 6)
            if points_item is not None:
                points_item.setText(f'{plan["num_x"]}, {plan["num_y"]}' if not errors else "—")
            time_item = self.plan_table.item(row, 7)
            if time_item is not None:
                time_item.setText(f'{plan["estimated_s"]/60:.1f} min' if not errors else "—")
            status_item = self.plan_table.item(row, 8)
            if status_item is not None:
                status_item.setText(status_text)
                status_item.setBackground(QtGui.QColor("#1f7a1f") if is_valid else QtGui.QColor("#8a1f11"))
                status_item.setForeground(QtGui.QColor("white"))
        finally:
            del blocker
        self._update_send_button_state()

    def _refresh_plan_table(self) -> None:
        for row in range(self.plan_table.rowCount()):
            self._refresh_plan_row(row)

    def _update_send_button_state(self) -> None:
        has_selected = False
        for row in range(self.plan_table.rowCount()):
            item = self.plan_table.item(row, 0)
            if item is not None and item.checkState() == QtCore.Qt.CheckState.Checked:
                has_selected = True
                break
        self.send_button.setEnabled(self.scan is not None and has_selected)

    def _on_plan_table_item_changed(self, item: QtWidgets.QTableWidgetItem):
        if item is None:
            return
        if item.column() == 0:
            self._update_send_button_state()
            return
        if item.column() in {2, 3, 4, 5}:
            self._refresh_plan_row(item.row())

    def load_data(self):
        try:
            raw_value = self.scan_number.text().strip()
            if not raw_value or (not raw_value.lstrip("-").isdigit()):
                raise ValueError("Scan number must be an integer.")
            self.scan_input = raw_value
            self.scan = load_xrf_data_for_scan(raw_value)
            self.plans = []
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
        self.plans = []
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
            roi = pg.RectROI((x, y), (w, h), pen=pg.mkPen(AUTO_ROI_COLOR, width=2), removable=True)
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
        preserved_selection = self._plan_selection_states()
        self.plans = []
        px, py = self.scan.pixel_size_um; ox, oy = self.scan.origin_um
        padding_fraction = self.padding_pct.value() / 100.0
        ny, nx = self.scan.stack.shape[1:]
        for i, roi in enumerate(self.rois, start=1):
            pos, size = roi.pos(), roi.size()
            x0 = max(0, pos.x()); x1 = min(nx, pos.x()+size.x())
            y0 = max(0, pos.y()); y1 = min(ny, pos.y()+size.y())
            roi_w_px = max(x1 - x0, 1.0)
            roi_h_px = max(y1 - y0, 1.0)
            pad_x_um, pad_y_um = _roi_padding_um((roi_w_px, roi_h_px), (px, py), padding_fraction)
            plan = {"roi": i, "x_start_um": ox + x0*px-pad_x_um, "x_stop_um": ox + x1*px+pad_x_um,
                    "y_start_um": oy + y0*py-pad_y_um, "y_stop_um": oy + y1*py+pad_y_um,
                    "step_um": self.step_um.value(), "dwell_s": self.dwell_s.value()}
            plan["num_x"] = max(2, round((plan["x_stop_um"]-plan["x_start_um"])/plan["step_um"])+1)
            plan["num_y"] = max(2, round((plan["y_stop_um"]-plan["y_start_um"])/plan["step_um"])+1)
            plan["estimated_s"] = plan["num_x"]*plan["num_y"]*plan["dwell_s"]
            plan["selected"] = preserved_selection[i - 1] if i - 1 < len(preserved_selection) else True
            plan["within_limits"] = _plan_is_valid(plan)
            plan["range_status"] = _plan_range_status(plan)
            self.plans.append(plan)
        self._show_plans()
        invalid_count = sum(1 for plan in self.plans if not plan["within_limits"])
        if invalid_count:
            self.statusBar().showMessage(
                f"Generated {len(self.plans)} scan plan(s); {invalid_count} exceed the scan limits or the {MAX_SCAN_POINTS:,} point cap."
            )
        else:
            self.statusBar().showMessage(f"Generated {len(self.plans)} scan plan(s).")

    def _show_plans(self):
        blocker = QtCore.QSignalBlocker(self.plan_table)
        try:
            self.plan_table.setRowCount(len(self.plans))
            for row, plan in enumerate(self.plans):
                self.plan_table.setItem(row, 0, self._make_plan_item("", checkable=True, checked=plan.get("selected", True)))
                self.plan_table.setItem(row, 1, self._make_plan_item(str(plan["roi"])))
                self.plan_table.setItem(row, 2, self._make_plan_item(f'{plan["x_start_um"]:.3f}', editable=True))
                self.plan_table.setItem(row, 3, self._make_plan_item(f'{plan["x_stop_um"]:.3f}', editable=True))
                self.plan_table.setItem(row, 4, self._make_plan_item(f'{plan["y_start_um"]:.3f}', editable=True))
                self.plan_table.setItem(row, 5, self._make_plan_item(f'{plan["y_stop_um"]:.3f}', editable=True))
                self.plan_table.setItem(row, 6, self._make_plan_item(f'{plan["num_x"]}, {plan["num_y"]}'))
                self.plan_table.setItem(row, 7, self._make_plan_item(f'{plan["estimated_s"]/60:.1f} min'))
                status_item = self._make_plan_item(plan.get("range_status", _plan_range_status(plan)))
                status_item.setBackground(QtGui.QColor("#1f7a1f") if plan.get("within_limits", False) else QtGui.QColor("#8a1f11"))
                status_item.setForeground(QtGui.QColor("white"))
                self.plan_table.setItem(row, 8, status_item)
        finally:
            del blocker
        self._update_send_button_state()
        self._refresh_plan_table()

    def send_plans(self):
        plans: list[dict] = []
        invalid_messages: list[str] = []
        for row in range(self.plan_table.rowCount()):
            plan, errors = self._plan_from_row(row)
            if not plan["selected"]:
                continue
            if errors:
                invalid_messages.extend(errors)
                continue
            if not plan["within_limits"]:
                invalid_messages.append(f"ROI {plan['roi']}: {plan['range_status']}")
                continue
            plans.append(plan)

        if invalid_messages:
            QtWidgets.QMessageBox.warning(
                self,
                "Cannot send selected scans",
                "\n".join(invalid_messages),
            )
            self.statusBar().showMessage("Selected scan plan(s) were not sent.")
            return

        if not plans:
            self.statusBar().showMessage("No selected scan plans to send.")
            return

        try:
            chosen = self.detector_system.currentText()
            dets = DETECTOR_PRESETS.get(chosen, [])
            send_scan_plans(
                plans,
                sid=self.scan_input if hasattr(self, "scan_input") else None,
                dets=dets,
                mot1="zpssx",
                mot2="zpssy",
            )
            self.statusBar().showMessage(f"Submitted {len(plans)} plan(s).")
        except Exception:
            import traceback
            QtWidgets.QMessageBox.critical(
                self,
                "Could not send plans",
                traceback.format_exc(),
            )


if __name__ == "__main__":
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    window = ROIScanPlanner(); window.show()
    sys.exit(app.exec())
