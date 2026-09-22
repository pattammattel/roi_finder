from PyQt6 import QtWidgets

from src.roi_finder.main import ROIScanPlanner, _roi_padding_um


def test_roi_padding_uses_relative_fraction_of_roi_size():
    pad_x_um, pad_y_um = _roi_padding_um((100, 80), (0.25, 0.50), 0.10)
    assert pad_x_um == 2.5
    assert pad_y_um == 4.0


def test_gui_initializes_without_viewbox_mouse_tracking_error():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = ROIScanPlanner()
    window.close()
    app.processEvents()