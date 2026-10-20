import sys
import types

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from zebrafish_measure import classification_curvature, length, select_torch_device


class TinyClassifier(nn.Module):
    """Tiny stand-in with FishClassifier's constructor signature."""

    created = []

    def __init__(self, num_classes, dense_layer_size, dropout_rate,
                 model_name="resnet101", pretrained=True):
        super().__init__()
        TinyClassifier.created.append({"model_name": model_name, "pretrained": pretrained})
        self.fc1 = nn.Linear(3, dense_layer_size)
        self.fc2 = nn.Linear(dense_layer_size, num_classes)

    def forward(self, x):
        x = x.mean(dim=(2, 3))
        return self.fc2(torch.relu(self.fc1(x)))


class Evil:
    pass


@pytest.fixture
def tiny(monkeypatch):
    TinyClassifier.created.clear()
    monkeypatch.setattr(length, "_get_fish_classifier_class", lambda: TinyClassifier)
    return TinyClassifier


@pytest.fixture
def no_hf(monkeypatch):
    fake = types.ModuleType("huggingface_hub")

    def _fail(*a, **k):
        pytest.fail("hf_hub_download must not be called in local mode")

    fake.hf_hub_download = _fail
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)


@pytest.fixture
def ckpt(tmp_path, tiny):
    path = tmp_path / "m.pth"
    torch.save(TinyClassifier(4, 512, 0.2).state_dict(), path)
    return path


def test_local_load(ckpt, tiny, no_hf):
    model = length.load_model(str(ckpt))
    assert isinstance(model, TinyClassifier)
    assert not model.training
    assert next(model.parameters()).device.type == "cpu"
    assert tiny.created[-1]["pretrained"] is False
    assert tiny.created[-1]["model_name"] == "convnext_base"


def test_local_load_unwraps_and_strips_prefix(tmp_path, tiny, no_hf):
    sd = {"module." + k: v for k, v in TinyClassifier(4, 512, 0.2).state_dict().items()}
    path = tmp_path / "w.pth"
    torch.save({"state_dict": sd}, path)
    assert isinstance(length.load_model(str(path)), TinyClassifier)


def test_missing_path(tmp_path, tiny, no_hf):
    with pytest.raises(FileNotFoundError, match="Curvature model not found"):
        length.load_model(str(tmp_path / "nope.pth"))


def test_unsafe_pickle_rejected(tmp_path, tiny, no_hf):
    path = tmp_path / "evil.pth"
    torch.save(Evil(), path)
    with pytest.raises(RuntimeError, match="safe loading"):
        length.load_model(str(path))


def test_missing_keys(tmp_path, tiny, no_hf):
    sd = TinyClassifier(4, 512, 0.2).state_dict()
    del sd["fc2.weight"]
    path = tmp_path / "bad.pth"
    torch.save(sd, path)
    with pytest.raises(RuntimeError, match="missing required keys"):
        length.load_model(str(path))


def test_wrong_keys(tmp_path, tiny, no_hf):
    path = tmp_path / "wrong.pth"
    torch.save({"foo": torch.zeros(1)}, path)
    with pytest.raises(RuntimeError, match="missing required keys"):
        length.load_model(str(path))


def test_device_cpu_honoured(ckpt, tiny, no_hf, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("select_torch_device must not be used when device is given")

    monkeypatch.setattr(length, "select_torch_device", boom)
    model = length.load_model(str(ckpt), device="cpu")
    assert next(model.parameters()).device.type == "cpu"


def test_local_mode_uses_probe(ckpt, tiny, no_hf, monkeypatch):
    seen = {}

    def fake_select(torch_mod, probe_model=None, **kw):
        seen["probe"] = probe_model
        return torch.device("cpu")

    monkeypatch.setattr(length, "select_torch_device", fake_select)
    model = length.load_model(str(ckpt))
    assert seen["probe"] is model


# ---- select_torch_device (ported from the Slicer test-suite) ----

class _FakeModel:
    def __init__(self, cuda_functional=True):
        self.cuda_functional = cuda_functional
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self

    def __call__(self, x):
        if self.device == "cuda" and not self.cuda_functional:
            raise RuntimeError("CUDA error: no kernel image is available for execution on the device")
        return x


class _NoGradCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeTorch:
    class cuda:
        pass

    def zeros(self, *args, device=None, **kwargs):
        return f"tensor(device={device})"

    def device(self, name):
        return f"device:{name}"

    def no_grad(self):
        return _NoGradCtx()


def _fake_torch(cuda_available):
    fake = _FakeTorch()
    fake.cuda.is_available = lambda: cuda_available
    return fake


def test_select_no_cuda_returns_cpu():
    assert select_torch_device(_fake_torch(False), probe_model=_FakeModel()) == "device:cpu"


def test_select_no_probe_model():
    assert select_torch_device(_fake_torch(True)) == "device:cuda"


def test_select_functional_cuda():
    model = _FakeModel(cuda_functional=True)
    assert select_torch_device(_fake_torch(True), probe_model=model) == "device:cuda"
    assert model.device == "cuda"


def test_select_broken_cuda_falls_back():
    model = _FakeModel(cuda_functional=False)
    assert select_torch_device(_fake_torch(True), probe_model=model) == "device:cpu"
    assert model.device == "cpu"


# ---- classification_curvature device handling ----

@pytest.fixture
def img_mask():
    rng = np.random.default_rng(0)
    img = rng.integers(50, 255, (128, 128, 3), dtype=np.uint8)
    mask = np.zeros((128, 128), np.uint8)
    cv2.ellipse(mask, (64, 64), (40, 15), 0, 0, 360, 255, -1)
    return img, mask


def test_curvature_uses_model_device(img_mask, monkeypatch):
    monkeypatch.setattr(length, "_get_device",
                        lambda: pytest.fail("_get_device used although model has parameters"))
    model = TinyClassifier(4, 8, 0.0).eval()
    img, mask = img_mask
    crop, cur = classification_curvature(img, mask, model, False, 0.0)
    assert crop is not None
    assert cur.device.type == "cpu"
    assert 1 <= int(cur[0]) <= 4
    _, cur = classification_curvature(img, mask, model, True, 1.1)
    assert cur.device.type == "cpu" and int(cur[0]) == 5


def test_curvature_empty_mask(img_mask):
    img, _ = img_mask
    model = TinyClassifier(4, 8, 0.0)
    crop, cur = classification_curvature(img, np.zeros((128, 128), np.uint8), model, False, 0.0)
    assert crop is None
    assert cur.device.type == "cpu" and cur.tolist() == [5]


# ---- FishClassifier pretrained passthrough ----

def test_fishclassifier_pretrained_flag(monkeypatch):
    import timm

    calls = []

    def fake_create(name, pretrained=None, num_classes=None, **kw):
        calls.append((name, pretrained, num_classes))
        return nn.Sequential(nn.Conv2d(3, 5, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten())

    monkeypatch.setattr(timm, "create_model", fake_create)
    monkeypatch.setattr(length, "_FishClassifier", None)
    cls = length._get_fish_classifier_class()
    try:
        cls(4, 8, 0.1, model_name="x", pretrained=False)
        cls(4, 8, 0.1, model_name="x")
    finally:
        length._FishClassifier = None
    assert calls == [("x", False, 0), ("x", True, 0)]
