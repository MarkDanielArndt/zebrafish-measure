import threading

import numpy as np

import zebrafish_measure as zm
from zebrafish_measure.length import _fish_axis_local_width


def test_all_names_exported():
    for name in zm.__all__:
        assert hasattr(zm, name), name


def _run_with_timeout(fn, timeout=5):
    box = {}

    def target():
        box["r"] = fn()

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), "function hung"
    return box["r"]


def test_local_width_empty_mask_does_not_hang():
    mask = np.zeros((10, 10), bool)
    assert _run_with_timeout(lambda: _fish_axis_local_width(mask, 1.0, 0.0)) == (0.0, None, None)


def test_local_width_single_pixel_does_not_hang():
    mask = np.zeros((10, 10), bool)
    mask[4, 5] = True
    assert _run_with_timeout(lambda: _fish_axis_local_width(mask, 1.0, 0.0)) == (0.0, None, None)
