from src.roi_finder.main import _roi_padding_um


def test_roi_padding_uses_relative_fraction_of_roi_size():
    pad_x_um, pad_y_um = _roi_padding_um((100, 80), (0.25, 0.50), 0.10)
    assert pad_x_um == 2.5
    assert pad_y_um == 4.0