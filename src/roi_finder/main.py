"""Interactive XRF ROI selector and scan-plan generator.

Replace ``load_xrf_data_for_scan`` and ``send_scan_plans`` with the beamline
implementations.  Everything else is intentionally independent of the data
acquisition framework.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pyqtgraph as pg
import tifffile
from PyQt6 import QtCore, QtGui, QtWidgets
from hxntools.CompositeBroker import db
from hxntools.scan_info import get_scan_positions

from qserver_utils import send_fly2d_recover_and_scan
from xrf_utils import get_all_xrf_roi_data, get_real_scan_geometry



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
USWDS_INK = "#1b1b1b"
USWDS_BASE_LIGHTEST = "#f0f0f0"
USWDS_BASE_LIGHTER = "#dfe1e2"
USWDS_BASE_LIGHT = "#a9aeb1"
USWDS_BASE = "#71767a"
USWDS_BASE_DARK = "#565c65"
USWDS_BASE_DARKER = "#3d4551"
USWDS_PRIMARY = "#005ea2"
USWDS_PRIMARY_VIVID = "#0050d8"
USWDS_PRIMARY_DARK = "#1a4480"
USWDS_SECONDARY = "#d83933"
USWDS_ACCENT_COOL = "#00bde3"
USWDS_ACCENT_COOL_DARK = "#28a0cb"
USWDS_WARNING = "#ffbe2e"
USWDS_SUCCESS = "#00a91c"
USWDS_ERROR = "#d54309"
MANUAL_ROI_COLOR = USWDS_PRIMARY_VIVID
AUTO_ROI_COLOR = USWDS_WARNING
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
EXTERNAL_IMAGE_FILTER = "Supported images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
EXTERNAL_CONFIG_FILTER = "JSON files (*.json)"

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
        return f"Too many points (> {max_points:,}); increase step size"
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


def _roi_padding_um(roi_size_px: tuple[float, float], pixel_size_um: tuple[float, float], padding_fraction: float) -> tuple[float, float]:
    """Return padding in microns as a fraction of the ROI size."""
    width_px, height_px = roi_size_px
    pad_x_um = max(width_px, 0.0) * pixel_size_um[0] * max(padding_fraction, 0.0)
    pad_y_um = max(height_px, 0.0) * pixel_size_um[1] * max(padding_fraction, 0.0)
    return pad_x_um, pad_y_um


def _qimage_to_array(image: QtGui.QImage) -> np.ndarray:
    if image.isNull():
        raise ValueError("One or more image files could not be read.")

    converted = image.convertToFormat(QtGui.QImage.Format.Format_Grayscale8)
    height = converted.height()
    width = converted.width()
    bytes_per_line = converted.bytesPerLine()
    buffer = converted.bits()
    buffer.setsize(height * bytes_per_line)
    array = np.frombuffer(buffer, dtype=np.uint8, count=height * bytes_per_line)
    return array.reshape((height, bytes_per_line))[:, :width].astype(np.float32, copy=True)


def _normalize_external_image_array(data: np.ndarray, image_path: Path) -> list[np.ndarray]:
    array = np.asarray(data)
    if array.size == 0:
        raise ValueError(f"Image file is empty: {image_path.name}")

    array = np.squeeze(array)
    if array.ndim == 2:
        return [array.astype(np.float32, copy=False)]
    if array.ndim == 3:
        if array.shape[-1] in {3, 4}:
            rgb = array[..., :3].astype(np.float32, copy=False)
            grayscale = np.tensordot(rgb, np.array([0.299, 0.587, 0.114], dtype=np.float32), axes=([-1], [0]))
            return [grayscale.astype(np.float32, copy=False)]
        return [layer.astype(np.float32, copy=False) for layer in array]
    raise ValueError(
        f"Unsupported image dimensions in {image_path.name}: expected 2D image or 3D TIFF array, got shape {array.shape}"
    )


def _load_external_image_layers(image_path: Path) -> list[tuple[str, np.ndarray]]:
    suffix = image_path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        layers = _normalize_external_image_array(tifffile.imread(image_path), image_path)
    else:
        layers = [_qimage_to_array(QtGui.QImage(str(image_path)))]

    if len(layers) == 1:
        return [(image_path.stem, layers[0])]
    return [
        (f"{image_path.stem}_{index + 1:03d}", layer)
        for index, layer in enumerate(layers)
    ]


def load_xrf_data_from_files(
    file_paths: list[str],
    pixel_size_um: tuple[float, float],
    origin_um: tuple[float, float],
) -> XRFScan:
    image_paths = [Path(file_path).expanduser() for file_path in file_paths if file_path.strip()]
    if not image_paths:
        raise ValueError("Select one or more image files first.")

    for image_path in image_paths:
        if not image_path.is_file():
            raise ValueError(f"Image file does not exist: {image_path}")
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {image_path.name}")

    stack: list[np.ndarray] = []
    element_names: list[str] = []
    expected_shape: tuple[int, int] | None = None
    for image_path in image_paths:
        for element_name, image in _load_external_image_layers(image_path):
            if expected_shape is None:
                expected_shape = image.shape
            elif image.shape != expected_shape:
                raise ValueError("All images and TIFF array slices in the folder must have the same pixel dimensions.")
            stack.append(image)
            element_names.append(element_name)

    return XRFScan(np.asarray(stack), element_names, pixel_size_um, origin_um)


def load_xrf_data_from_file(
    file_path: str,
    pixel_size_um: tuple[float, float],
    origin_um: tuple[float, float],
) -> XRFScan:
    return load_xrf_data_from_files([file_path], pixel_size_um, origin_um)


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
        pixel_size_um, origin_um = get_real_scan_geometry(hdr)
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
        self.resize(1560, 860)
        self.setMinimumSize(1360, 760)
        self.scan: XRFScan | None = None
        self.scan_source_label: str | None = None
        self.recovery_scan_id: str | None = None
        self.rois: list[pg.RectROI] = []
        self.plans: list[dict] = []
        self._build_ui()
        self._apply_uswds_theme()
        self.statusBar().showMessage("Load beamline XRF data or import external images to begin.")

    def _apply_uswds_theme(self) -> None:
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(USWDS_BASE_LIGHTEST))
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(USWDS_INK))
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#ffffff"))
        palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(USWDS_BASE_LIGHTEST))
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(USWDS_INK))
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(USWDS_BASE_LIGHTEST))
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(USWDS_INK))
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(USWDS_PRIMARY))
        palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#ffffff"))
        palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor(USWDS_PRIMARY_VIVID))
        self.setPalette(palette)

        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{
                background-color: {USWDS_BASE_LIGHTEST};
                color: {USWDS_INK};
                font-size: 11px;
            }}
            QGroupBox {{
                background-color: #ffffff;
                border: 1px solid {USWDS_BASE_LIGHTER};
                border-radius: 8px;
                margin-top: 10px;
                padding: 12px 10px 10px 10px;
                font-weight: 600;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: {USWDS_PRIMARY_DARK};
                background-color: #ffffff;
            }}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QTableWidget, QTabWidget::pane {{
                background-color: #ffffff;
                border: 1px solid {USWDS_BASE_LIGHT};
                border-radius: 6px;
                color: {USWDS_INK};
            }}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                min-height: 26px;
                padding: 3px 7px;
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QTextEdit:focus {{
                border: 2px solid {USWDS_PRIMARY_VIVID};
            }}
            QPushButton {{
                background-color: {USWDS_PRIMARY};
                color: #ffffff;
                border: 1px solid {USWDS_PRIMARY_DARK};
                border-radius: 6px;
                min-height: 28px;
                padding: 4px 10px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {USWDS_PRIMARY_VIVID};
            }}
            QPushButton:pressed {{
                background-color: {USWDS_PRIMARY_DARK};
            }}
            QPushButton:disabled {{
                background-color: {USWDS_BASE_LIGHT};
                border-color: {USWDS_BASE};
                color: #ffffff;
            }}
            QTabBar::tab {{
                background-color: {USWDS_BASE_LIGHTER};
                color: {USWDS_BASE_DARKER};
                border: 1px solid {USWDS_BASE_LIGHT};
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 6px 10px;
                margin-right: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: #ffffff;
                color: {USWDS_PRIMARY_DARK};
            }}
            QHeaderView::section {{
                background-color: {USWDS_PRIMARY_DARK};
                color: #ffffff;
                padding: 5px;
                border: none;
                font-weight: 600;
            }}
            QTableWidget {{
                gridline-color: {USWDS_BASE_LIGHTER};
                selection-background-color: {USWDS_ACCENT_COOL};
                selection-color: {USWDS_INK};
            }}
            QTableWidget::item {{
                padding: 4px;
            }}
            QRadioButton, QCheckBox {{
                spacing: 8px;
            }}
            QStatusBar {{
                background-color: {USWDS_BASE_DARKER};
                color: #ffffff;
            }}
            QLabel {{
                color: {USWDS_BASE_DARKER};
            }}
            """
        )

        self.plot.setBackground("#ffffff")
        plot_item = self.plot.getPlotItem()
        plot_item.getAxis("bottom").setTextPen(pg.mkPen(USWDS_BASE_DARKER))
        plot_item.getAxis("left").setTextPen(pg.mkPen(USWDS_BASE_DARKER))
        plot_item.getAxis("bottom").setPen(pg.mkPen(USWDS_BASE_DARK))
        plot_item.getAxis("left").setPen(pg.mkPen(USWDS_BASE_DARK))
        self.hover_label.setColor(USWDS_INK)
        self.image_histogram.setBackground(USWDS_BASE_LIGHTEST)

    def _make_field_block(self, label_text: str, control: QtWidgets.QWidget) -> QtWidgets.QWidget:
        block = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        label = QtWidgets.QLabel(label_text)
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addWidget(control)
        return block

    def _make_field_row(self, label_text: str, control: QtWidgets.QWidget) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        label = QtWidgets.QLabel(label_text)
        label.setMinimumWidth(96)
        layout.addWidget(label)
        layout.addWidget(control, 1)
        return row

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        controls = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(controls)
        form.setSpacing(10)
        form.setContentsMargins(2, 2, 8, 2)

        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidget(controls)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setMinimumWidth(380)
        controls_scroll.setMaximumWidth(440)

        load_tabs = QtWidgets.QTabWidget()
        load_tabs.setMinimumHeight(320)

        beamline_tab = QtWidgets.QWidget()
        beamline_layout = QtWidgets.QVBoxLayout(beamline_tab)
        beamline_layout.setContentsMargins(8, 8, 8, 8)
        beamline_layout.setSpacing(10)
        self.scan_number = QtWidgets.QLineEdit()
        self.scan_number.setPlaceholderText('e.g. 123456 or -1 (use "0000" for phantom test data)')
        self.scan_number.setValidator(QtGui.QIntValidator(-999999999, 999999999, self))
        self.load_button = QtWidgets.QPushButton("Load XRF data")
        self.load_button.clicked.connect(self.load_beamline_data)
        self.load_button.setMinimumHeight(34)
        beamline_layout.addWidget(self._make_field_block("Scan number", self.scan_number))
        beamline_layout.addWidget(self.load_button)
        beamline_layout.addStretch(1)
        load_tabs.addTab(beamline_tab, "Beamline XRF")

        external_tab = QtWidgets.QWidget()
        external_layout = QtWidgets.QVBoxLayout(external_tab)
        external_layout.setContentsMargins(8, 8, 8, 8)
        external_layout.setSpacing(10)
        self.external_files = QtWidgets.QLineEdit()
        self.external_files.setPlaceholderText("Choose a SEM, optical, or XRF image file")
        self.external_files.setReadOnly(True)
        self.external_browse_button = QtWidgets.QPushButton("Choose image…")
        self.external_browse_button.clicked.connect(self.browse_and_load_external_data)
        files_row = QtWidgets.QWidget()
        files_layout = QtWidgets.QVBoxLayout(files_row)
        files_layout.setContentsMargins(0, 0, 0, 0)
        files_layout.setSpacing(6)
        files_layout.addWidget(self.external_files)
        files_layout.addWidget(self.external_browse_button)
        self.external_browse_button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        self.external_pixel_x = QtWidgets.QDoubleSpinBox(); self.external_pixel_x.setRange(0.000001, 1000000); self.external_pixel_x.setDecimals(6); self.external_pixel_x.setValue(0.25); self.external_pixel_x.setSuffix(" µm")
        self.external_pixel_y = QtWidgets.QDoubleSpinBox(); self.external_pixel_y.setRange(0.000001, 1000000); self.external_pixel_y.setDecimals(6); self.external_pixel_y.setValue(0.25); self.external_pixel_y.setSuffix(" µm")
        self.external_origin_x = QtWidgets.QDoubleSpinBox(); self.external_origin_x.setRange(-1000000, 1000000); self.external_origin_x.setDecimals(6); self.external_origin_x.setValue(0.0); self.external_origin_x.setSuffix(" µm")
        self.external_origin_y = QtWidgets.QDoubleSpinBox(); self.external_origin_y.setRange(-1000000, 1000000); self.external_origin_y.setDecimals(6); self.external_origin_y.setValue(0.0); self.external_origin_y.setSuffix(" µm")
        self.external_recovery_scan = QtWidgets.QLineEdit()
        self.external_recovery_scan.setPlaceholderText("Optional: scan ID used only for queue-server recovery before sending")
        self.external_import_json_button = QtWidgets.QPushButton("Import JSON…")
        self.external_import_json_button.clicked.connect(self.import_external_config)
        self.external_export_json_button = QtWidgets.QPushButton("Export JSON…")
        self.external_export_json_button.clicked.connect(self.export_external_config)
        json_row = QtWidgets.QWidget()
        json_layout = QtWidgets.QVBoxLayout(json_row)
        json_layout.setContentsMargins(0, 0, 0, 0)
        json_layout.setSpacing(6)
        json_layout.addWidget(self.external_import_json_button)
        json_layout.addWidget(self.external_export_json_button)
        external_note = QtWidgets.QLabel("Load one image at a time. TIFF arrays still expand into one layer per slice. Supported files: PNG, JPG, TIFF, BMP.")
        external_note.setWordWrap(True)
        self.external_browse_button.setMinimumHeight(34)
        self.external_import_json_button.setMinimumHeight(30)
        self.external_export_json_button.setMinimumHeight(30)
        external_layout.addWidget(self._make_field_block("Image file", files_row))
        external_layout.addWidget(self._make_field_row("Pixel size x", self.external_pixel_x))
        external_layout.addWidget(self._make_field_row("Pixel size y", self.external_pixel_y))
        external_layout.addWidget(self._make_field_row("Origin x", self.external_origin_x))
        external_layout.addWidget(self._make_field_row("Origin y", self.external_origin_y))
        external_layout.addWidget(self._make_field_block("Recovery scan ID", self.external_recovery_scan))
        external_layout.addWidget(self._make_field_block("Saved settings", json_row))
        external_layout.addWidget(external_note)
        external_layout.addStretch(1)
        load_tabs.addTab(external_tab, "External Images")

        form.addWidget(load_tabs)

        selection = QtWidgets.QGroupBox("ROI selection")
        selection_layout = QtWidgets.QVBoxLayout(selection)
        selection_layout.setSpacing(10)
        self.auto_radio = QtWidgets.QRadioButton("Auto — find ROIs")
        self.manual_radio = QtWidgets.QRadioButton("Manual — draw ROIs")
        self.auto_radio.setChecked(True)
        self.auto_button = QtWidgets.QPushButton("Find ROIs")
        self.auto_button.clicked.connect(self.find_rois)
        self.clear_button = QtWidgets.QPushButton("Clear ROIs")
        self.clear_button.clicked.connect(self.clear_rois)
        self.auto_button.setMinimumHeight(36)
        self.clear_button.setMinimumHeight(36)
        selection_layout.addWidget(self.auto_radio)
        selection_layout.addWidget(self.manual_radio)
        selection_layout.addWidget(self.auto_button)
        selection_layout.addWidget(self.clear_button)
        selection_note = QtWidgets.QLabel("Manual: right-drag on image to draw.\nDrag handles to refine; right-click ROI to remove.")
        selection_note.setWordWrap(True)
        selection_layout.addWidget(selection_note)
        form.addWidget(selection)
        form.addStretch(1)
        layout.addWidget(controls_scroll)

        right_controls = QtWidgets.QWidget()
        right_form = QtWidgets.QVBoxLayout(right_controls)
        right_form.setSpacing(10)
        right_form.setContentsMargins(2, 2, 8, 2)
        right_controls_scroll = QtWidgets.QScrollArea()
        right_controls_scroll.setWidget(right_controls)
        right_controls_scroll.setWidgetResizable(True)
        right_controls_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        right_controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_controls_scroll.setMinimumWidth(300)
        right_controls_scroll.setMaximumWidth(360)

        parameters = QtWidgets.QGroupBox("Fine-scan parameters")
        params = QtWidgets.QVBoxLayout(parameters)
        params.setSpacing(10)
        self.step_um = QtWidgets.QDoubleSpinBox(); self.step_um.setRange(0.001, 100); self.step_um.setValue(0.05); self.step_um.setSuffix(" µm")
        self.dwell_s = QtWidgets.QDoubleSpinBox(); self.dwell_s.setRange(0.0001, 100); self.dwell_s.setDecimals(4); self.dwell_s.setValue(0.01); self.dwell_s.setSuffix(" s")
        self.padding_pct = QtWidgets.QDoubleSpinBox(); self.padding_pct.setRange(0, 100); self.padding_pct.setDecimals(1); self.padding_pct.setSingleStep(0.5); self.padding_pct.setValue(10.0); self.padding_pct.setSuffix(" %")
        self.detector_system = QtWidgets.QComboBox()
        self.detector_system.addItems(["dets_fast", "dets_fast_merlin", "dets_fast_fs"])
        self.detector_system.setCurrentText("dets_fast")
        params.addWidget(self._make_field_block("Step size", self.step_um))
        params.addWidget(self._make_field_block("Dwell", self.dwell_s))
        params.addWidget(self._make_field_block("ROI padding", self.padding_pct))
        params.addWidget(self._make_field_block("Detector system", self.detector_system))
        self.auto_update_table = QtWidgets.QCheckBox("Auto-update table when parameters change")
        self.auto_update_table.setChecked(False)
        params.addWidget(self.auto_update_table)
        self.step_um.valueChanged.connect(self._on_scan_param_changed)
        self.dwell_s.valueChanged.connect(self._on_scan_param_changed)
        self.padding_pct.valueChanged.connect(self._on_scan_param_changed)
        right_form.addWidget(parameters)

        self.generate_button = QtWidgets.QPushButton("Generate scan plans")
        self.generate_button.clicked.connect(self.generate_plans)
        self.update_button = QtWidgets.QPushButton("Update table from params")
        self.update_button.clicked.connect(self.generate_plans)
        self.send_button = QtWidgets.QPushButton("Send scans")
        self.send_button.clicked.connect(self.send_plans)
        self.send_button.setEnabled(False)
        right_form.addWidget(self.generate_button)
        right_form.addWidget(self.update_button)
        right_form.addWidget(self.send_button)

        info_group = QtWidgets.QGroupBox("Scan info")
        info_layout = QtWidgets.QVBoxLayout(info_group)
        self.scan_info_box = QtWidgets.QTextEdit()
        self.scan_info_box.setReadOnly(True)
        self.scan_info_box.setMinimumHeight(120)
        self.scan_info_box.setPlainText("No scan loaded.\nLoad a beamline scan or import external images with manual geometry.")
        info_layout.addWidget(self.scan_info_box)
        right_form.addWidget(info_group)
        right_form.addStretch(1)

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
        self.plan_table.verticalHeader().setDefaultSectionSize(28)
        self.plan_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.AllEditTriggers)
        self.plan_table.itemChanged.connect(self._on_plan_table_item_changed)
        plan_layout.addWidget(self.plan_table)
        right.addWidget(plan_box)
        right.setSizes([620, 220])
        layout.addWidget(right, stretch=1)
        layout.addWidget(right_controls_scroll)

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
            self.scan_info_box.setPlainText("No scan loaded.\nLoad a beamline scan or import external images with manual geometry.")
            return

        shape = self.scan.stack.shape
        pixel_size_x, pixel_size_y = self.scan.pixel_size_um
        origin_x, origin_y = self.scan.origin_um
        source_label = self.scan_source_label or "n/a"
        recovery_scan_id = self.recovery_scan_id or "not set"
        text = (
            f"Data source: {source_label}\n"
            f"Recovery scan ID: {recovery_scan_id}\n"
            f"Shape: {shape[2]} × {shape[1]} px\n"
            f"Elements: {shape[0]}\n"
            f"Element names: {', '.join(self.scan.element_names)}\n\n"
            f"Pixel size: ({pixel_size_x:.3f}, {pixel_size_y:.3f}) µm\n"
            f"Origin: ({origin_x:.3f}, {origin_y:.3f}) µm\n"
            f"Field of view: ({shape[2] * pixel_size_x:.3f}, {shape[1] * pixel_size_y:.3f}) µm"
        )
        self.scan_info_box.setPlainText(text)

    def _set_scan(self, scan: XRFScan, source_label: str, recovery_scan_id: str | None) -> None:
        self.scan = scan
        self.scan_source_label = source_label
        self.recovery_scan_id = recovery_scan_id.strip() if recovery_scan_id else None
        self.plans = []
        if self.scan.stack.ndim != 3 or self.scan.stack.shape[0] != len(self.scan.element_names):
            raise ValueError("Expected image stack shape (n_elements, y, x) and matching names.")
        self.x_axis.set_scan_geometry(self.scan.pixel_size_um, self.scan.origin_um)
        self.y_axis.set_scan_geometry(self.scan.pixel_size_um, self.scan.origin_um)
        self.element_combo.blockSignals(True)
        self.element_combo.clear(); self.element_combo.addItems(self.scan.element_names)
        self.element_combo.blockSignals(False)
        self.clear_rois(); self.show_element(0)
        self._update_scan_info()

    @staticmethod
    def _serialize_external_files(file_paths: list[str]) -> str:
        return file_paths[0] if file_paths else ""

    def _selected_external_files(self) -> list[str]:
        raw_value = self.external_files.text().strip()
        if not raw_value:
            return []
        return [raw_value]

    def _set_selected_external_files(self, file_paths: list[str]) -> None:
        selected_file = file_paths[:1]
        self.external_files.setText(self._serialize_external_files(selected_file))
        self.external_files.setToolTip(selected_file[0] if selected_file else "")

    def _external_config(self) -> dict:
        return {
            "file_paths": self._selected_external_files(),
            "pixel_size_um": {
                "x": self.external_pixel_x.value(),
                "y": self.external_pixel_y.value(),
            },
            "origin_um": {
                "x": self.external_origin_x.value(),
                "y": self.external_origin_y.value(),
            },
            "recovery_scan_id": self.external_recovery_scan.text().strip(),
        }

    def _apply_external_config(self, config: dict) -> None:
        if not isinstance(config, dict):
            raise ValueError("External image JSON must contain a JSON object.")

        file_paths = config.get("file_paths")
        pixel_size = config.get("pixel_size_um")
        origin = config.get("origin_um")

        if not isinstance(file_paths, list) or not all(isinstance(path, str) for path in file_paths):
            raise ValueError("JSON field 'file_paths' must be a list of file paths.")
        if len(file_paths) > 1:
            raise ValueError("JSON settings now support only one external image at a time.")
        if not isinstance(pixel_size, dict) or not all(axis in pixel_size for axis in ("x", "y")):
            raise ValueError("JSON field 'pixel_size_um' must contain 'x' and 'y'.")
        if not isinstance(origin, dict) or not all(axis in origin for axis in ("x", "y")):
            raise ValueError("JSON field 'origin_um' must contain 'x' and 'y'.")

        self._set_selected_external_files(file_paths)
        self.external_pixel_x.setValue(float(pixel_size["x"]))
        self.external_pixel_y.setValue(float(pixel_size["y"]))
        self.external_origin_x.setValue(float(origin["x"]))
        self.external_origin_y.setValue(float(origin["y"]))
        self.external_recovery_scan.setText(str(config.get("recovery_scan_id", "")))

    def browse_and_load_external_data(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select external image file",
            "",
            EXTERNAL_IMAGE_FILTER,
        )
        if not file_path:
            return
        self._set_selected_external_files([file_path])
        self.load_external_data()

    def import_external_config(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import external image settings",
            "",
            EXTERNAL_CONFIG_FILTER,
        )
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as stream:
                config = json.load(stream)
            self._apply_external_config(config)
            self.statusBar().showMessage(f"Loaded external image settings from {Path(file_path).name}.")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Could not import settings", str(exc))

    def export_external_config(self):
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export external image settings",
            "external-image-settings.json",
            EXTERNAL_CONFIG_FILTER,
        )
        if not file_path:
            return
        try:
            target_path = Path(file_path)
            config = self._external_config()
            with target_path.open("w", encoding="utf-8") as stream:
                json.dump(config, stream, indent=2)
            self.statusBar().showMessage(f"Saved external image settings to {target_path.name}.")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Could not export settings", str(exc))

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
                status_item.setBackground(QtGui.QColor(USWDS_SUCCESS) if is_valid else QtGui.QColor(USWDS_ERROR))
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

    def load_beamline_data(self):
        try:
            raw_value = self.scan_number.text().strip()
            if not raw_value or (not raw_value.lstrip("-").isdigit()):
                raise ValueError("Scan number must be an integer.")
            scan = load_xrf_data_for_scan(raw_value)
            self._set_scan(scan, f"Beamline XRF scan {raw_value}", raw_value)
            self.statusBar().showMessage(f"Loaded {self.scan.stack.shape[0]} elements; image size {self.scan.stack.shape[2]} × {self.scan.stack.shape[1]} px.")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Could not load XRF data", str(exc))

    def load_data(self):
        self.load_beamline_data()

    def load_external_data(self):
        try:
            file_paths = self._selected_external_files()
            if not file_paths:
                raise ValueError("Choose an image file first.")

            pixel_size_um = (self.external_pixel_x.value(), self.external_pixel_y.value())
            origin_um = (self.external_origin_x.value(), self.external_origin_y.value())
            recovery_scan_id = self.external_recovery_scan.text().strip() or None
            scan = load_xrf_data_from_file(file_paths[0], pixel_size_um, origin_um)
            source_label = f"External image {Path(file_paths[0]).name}"
            self._set_scan(scan, source_label, recovery_scan_id)
            self.statusBar().showMessage(
                f"Loaded {self.scan.stack.shape[0]} image layer(s) from {Path(file_paths[0]).name}; image size {self.scan.stack.shape[2]} × {self.scan.stack.shape[1]} px."
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Could not load external images", str(exc))

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
                status_item.setBackground(QtGui.QColor(USWDS_SUCCESS) if plan.get("within_limits", False) else QtGui.QColor(USWDS_ERROR))
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
                sid=self.recovery_scan_id,
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
