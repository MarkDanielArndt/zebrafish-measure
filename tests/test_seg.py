import sys
import types

import cv2
import numpy as np
import pytest
import torch

from zebrafish_measure import adjust_mask_boundary, fill_holes, grow_mask
from zebrafish_measure import seg


def test_fill_holes_ring():
    m = np.zeros((60, 60), np.uint8)
    cv2.circle(m, (30, 30), 20, 255, 3)
    filled = fill_holes(m)
    assert filled[30, 30] == 255
    assert (filled > 0).sum() > (m > 0).sum()


def test_grow_mask():
    m = np.zeros((60, 60), np.uint8)
    m[25:35, 25:35] = 255
    g = grow_mask(m, iterations=2)
    assert (g > 0).sum() > (m > 0).sum()
    assert g.shape == m.shape


def test_adjust_mask_boundary_direction():
    m = np.zeros((80, 80), np.uint8)
    cv2.circle(m, (40, 40), 20, 255, -1)
    base = (m > 0).sum()
    assert (adjust_mask_boundary(m, 2.0) > 0).sum() > base
    assert (adjust_mask_boundary(m, -2.0) > 0).sum() < base
    assert (adjust_mask_boundary(m, 0) > 0).sum() == base
    assert not adjust_mask_boundary(np.zeros((10, 10), np.uint8), 1.0).any()


def test_load_unet_uses_no_pretrained_weights(monkeypatch, tmp_path):
    calls = {}

    class FakeModel(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            calls.update(kwargs)
            self.lin = torch.nn.Linear(1, 1)

    ckpt = tmp_path / "fake.pth"
    torch.save(FakeModel().state_dict(), ckpt)
    calls.clear()

    monkeypatch.setitem(seg._MODEL_TYPES, "Unet", FakeModel)
    monkeypatch.setattr(seg, "_UNET_CACHE", {})
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda *a, **k: pytest.fail("download attempted")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    model = seg._load_unet_model(model_path=str(ckpt), encoder_name="vgg16", model_type="Unet")
    assert model is not None
    assert calls["encoder_weights"] is None
    assert calls["encoder_name"] == "vgg16"
