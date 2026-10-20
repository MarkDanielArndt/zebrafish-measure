import cv2
import numpy as np
import pytest
import torch
from torch import nn

from zebrafish_measure import apply_mask, classification_curvature, preprocess_masked_image


class StubModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor([logits], dtype=torch.float32)

    def forward(self, x):
        return self.logits.repeat(x.shape[0], 1)


@pytest.fixture
def img_mask():
    rng = np.random.default_rng(0)
    img = rng.integers(50, 255, (128, 128, 3), dtype=np.uint8)
    mask = np.zeros((128, 128), np.uint8)
    cv2.ellipse(mask, (64, 64), (40, 15), 0, 0, 360, 255, -1)
    return img, mask


def test_apply_and_preprocess(img_mask):
    img, mask = img_mask
    masked = apply_mask(img, mask)
    assert masked.shape == img.shape
    assert masked[0, 0].sum() == 0
    out = preprocess_masked_image(masked)
    assert out.shape == (256, 256, 3)
    assert preprocess_masked_image(np.zeros((10, 10, 3), np.uint8)) is None


def test_classification_argmax(img_mask):
    img, mask = img_mask
    model = StubModel([0.0, 0.0, 5.0, 0.0])
    _, cur = classification_curvature(img, mask, model, False, 0.0)
    assert cur.device.type == "cpu"
    assert cur.shape == (1,)
    assert cur.numpy()[0] == 3


def test_classification_threshold(img_mask):
    img, mask = img_mask
    model = StubModel([0.0, 0.0, 5.0, 0.0])
    _, cur = classification_curvature(img, mask, model, True, 0.999999)
    assert cur.item() == 5
    _, cur = classification_curvature(img, mask, model, True, 0.5)
    assert cur.item() == 3


def test_classification_empty_mask(img_mask):
    img, _ = img_mask
    crop, cur = classification_curvature(img, np.zeros((128, 128), np.uint8), StubModel([1, 0, 0, 0]), False, 0)
    assert crop is None
    assert cur.device.type == "cpu" and cur.item() == 5
