import io
import pickle
import sys
import types

import pytest
import torch

import zebrafish_measure
from zebrafish_measure import length, seg


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 4)

    def forward(self, x):
        return self.lin(x.mean(dim=(2, 3)))


def test_fish_classifier_class_pickles():
    cls = zebrafish_measure.FishClassifier
    assert cls.__qualname__ == "FishClassifier"
    assert pickle.loads(pickle.dumps(cls)) is cls


def test_fish_classifier_instance_torch_save_load(monkeypatch):
    import timm
    monkeypatch.setattr(timm, "create_model", lambda *a, **k: _Tiny())
    cls = zebrafish_measure.FishClassifier
    inst = cls(num_classes=2, dense_layer_size=5, dropout_rate=0.1)
    buf = io.BytesIO()
    torch.save(inst, buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=False)
    assert isinstance(loaded, zebrafish_measure.FishClassifier)


def test_dir_lists_fish_classifier():
    assert "FishClassifier" in dir(zebrafish_measure)
    assert "FishClassifier" in dir(length)


def test_unet_cache_not_shared_with_local_only(monkeypatch, tmp_path):
    class FakeModel(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.lin = torch.nn.Linear(1, 1)

    ckpt = tmp_path / "ok.pth"
    torch.save(FakeModel().state_dict(), ckpt)
    monkeypatch.setitem(seg._MODEL_TYPES, "Unet", FakeModel)
    monkeypatch.setattr(seg, "_UNET_CACHE", {})
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda *a, **k: pytest.fail("download attempted")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    m1 = seg._load_unet_model(model_path=str(ckpt), device="cpu")
    assert seg._load_unet_model(model_path=str(ckpt), device="cpu") is m1
    with pytest.raises(RuntimeError):
        seg._load_unet_model(model_path=str(tmp_path / "missing.pth"), device="cpu", local_only=True)
    # same path in local_only mode must not reuse the default-mode entry
    m2 = seg._load_unet_model(model_path=str(ckpt), device="cpu", local_only=True)
    assert m2 is not m1
