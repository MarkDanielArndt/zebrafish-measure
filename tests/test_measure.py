import cv2
import numpy as np
import pytest

from zebrafish_measure import (
    compute_eye_diameters,
    compute_eye_metrics,
    compute_tube_metrics,
    tube_length_border2border,
)


def _rect(shape, p1, p2):
    m = np.zeros(shape, np.uint8)
    cv2.rectangle(m, p1, p2, 255, -1)
    return m


def _disc(r=20, shape=(100, 100)):
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, (shape[1] // 2, shape[0] // 2), r, 255, -1)
    return m


def test_tube_metrics_unit_spacing():
    out = compute_tube_metrics(_rect((100, 200), (40, 40), (139, 59)))
    assert out["length"] == pytest.approx(100, abs=3)
    assert out["width"] == pytest.approx(20, abs=3)
    assert out["area"] == pytest.approx(2000, rel=0.05)


def test_tube_metrics_scaled_spacing():
    out = compute_tube_metrics(_rect((100, 200), (40, 40), (139, 59)), spacing=(2.0, 3.0))
    assert out["length"] == pytest.approx(300, rel=0.05)
    assert out["width"] == pytest.approx(40, rel=0.1)
    assert out["area"] == pytest.approx(2000 * 6, rel=0.05)


def test_tube_metrics_empty_and_none():
    for m in (np.zeros((50, 50), np.uint8), None):
        out = compute_tube_metrics(m)
        assert out["length"] == 0.0 and out["length_line"] is None


def test_eye_metrics_disc_area():
    out = compute_eye_metrics(_disc(20))
    assert out["eye_area"] == pytest.approx(np.pi * 20 ** 2, rel=0.05)
    assert out["eye_diameter"] == pytest.approx(40, abs=3)
    assert out["eye_centroid"] == pytest.approx([50, 50], abs=1)


def test_eye_metrics_spacing_and_empty():
    out = compute_eye_metrics(_disc(20), spacing=(0.5, 0.5))
    assert out["eye_area"] == pytest.approx(np.pi * 20 ** 2 * 0.25, rel=0.05)
    empty = compute_eye_metrics(np.zeros((9, 9)))
    assert empty["eye_area"] == 0.0 and empty["eye_centroid"] is None


def test_eye_diameters_disc():
    out = compute_eye_diameters(_disc(20))
    assert out["eye_width_um"] == pytest.approx(40, abs=3)
    assert out["eye_height_um"] == pytest.approx(40, abs=3)
    assert compute_eye_diameters(np.zeros((9, 9)))["eye_width_line"] is None


def test_tube_length_border2border_straight_bar():
    bar = _rect((120, 300), (20, 50), (279, 69))  # 260 px long
    length, straight = tube_length_border2border(bar > 0)
    assert length == pytest.approx(260, rel=0.05)
    assert straight >= 255
    length2, _ = tube_length_border2border(bar > 0, spacing=(2.0, 2.0))
    assert length2 == pytest.approx(2 * length, rel=0.05)
