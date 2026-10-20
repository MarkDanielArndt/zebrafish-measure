import itertools
import sys
import types

import cv2
import numpy as np
import pytest
import torch

from zebrafish_measure import seg


class FakeModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    for name in ("Unet", "FPN", "Segformer"):
        monkeypatch.setitem(seg._MODEL_TYPES, name, FakeModel)
    monkeypatch.setattr(seg, "_UNET_CACHE", {})

    def fail(*a, **k):
        pytest.fail("Hugging Face download attempted")

    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = fail
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)


@pytest.fixture
def ckpt(tmp_path):
    p = tmp_path / "ok.pth"
    torch.save(FakeModel().state_dict(), p)
    return str(p)


@pytest.fixture
def img_files(tmp_path):
    files = []
    for i in range(2):
        f = tmp_path / f"img{i}.png"
        cv2.imwrite(str(f), np.full((40, 50, 3), 100 + i, np.uint8))
        files.append(str(f))
    return files


@pytest.fixture
def fake_seg(monkeypatch):
    calls = []

    def fake_segment_fish(image, model, biggest_only=True):
        calls.append({"biggest_only": biggest_only, "device": image.device.type})
        m = np.zeros((16, 16), np.uint8)
        m[4:12, 4:12] = 255
        return m, np.ones((16, 16), np.float32)

    monkeypatch.setattr(seg, "segment_fish", fake_segment_fish)
    return calls


def _all_paths(ckpt):
    return dict(
        body_model_path=ckpt,
        eye_model_path=ckpt,
        edema_model_path=ckpt,
        swimbladder_model_path=ckpt,
    )


def test_body_model_path_used_without_hf(ckpt, img_files, fake_seg):
    out = seg.segmentation_pipeline(
        file_list=img_files, target_size=(16, 16), body_model_path=ckpt, device="cpu"
    )
    assert len(out) == 3
    assert any(k[0] == ckpt for k in seg._UNET_CACHE)


@pytest.mark.parametrize("eyes,edema,swim", list(itertools.product([False, True], repeat=3)))
def test_return_arity_and_order(ckpt, img_files, fake_seg, eyes, edema, swim):
    out = seg.segmentation_pipeline(
        file_list=img_files, target_size=(16, 16), include_eyes=eyes,
        include_edema=edema, include_swimbladder=swim, device="cpu", **_all_paths(ckpt),
    )
    assert isinstance(out, tuple)
    assert len(out) == 3 + eyes + edema + swim
    assert out[0][0].shape == (40, 50, 3)
    assert all(len(x) == 2 for x in out)
    # eyes/edema/swim masks are distinguishable by segment_fish biggest_only calls
    biggest = [c["biggest_only"] for c in fake_seg]
    expected_per_img = [True] * eyes + [False] * edema + [True] * swim
    assert biggest == ([True] + expected_per_img) * 2


def test_edema_only_without_eyes_is_four_tuple(ckpt, img_files, fake_seg):
    out = seg.segmentation_pipeline(
        file_list=img_files, target_size=(16, 16), include_edema=True,
        device="cpu", **_all_paths(ckpt),
    )
    assert len(out) == 4
    assert out[3][0].shape == (16, 16)


def test_eyes_false_does_not_load_eye_model(ckpt, img_files, fake_seg, monkeypatch):
    labels = []
    orig = seg._load_unet_model

    def spy(*a, **k):
        labels.append(k.get("label"))
        return orig(*a, **k)

    monkeypatch.setattr(seg, "_load_unet_model", spy)
    seg.segmentation_pipeline(
        file_list=img_files, target_size=(16, 16), include_edema=True,
        device="cpu", **_all_paths(ckpt),
    )
    assert "eye model" not in labels


def test_local_only_missing_body_raises(tmp_path):
    with pytest.raises(RuntimeError, match="not found"):
        seg._load_unet_model(model_path=str(tmp_path / "nope.pth"), repo_id="r", filename="f",
                             label="body model", local_only=True)
    with pytest.raises(RuntimeError, match="not found"):
        seg._load_unet_model(model_path=None, repo_id="r", filename="f", local_only=True)


@pytest.mark.parametrize("opt", ["eyes", "edema", "swim"])
def test_local_only_missing_optional_raises(ckpt, img_files, fake_seg, tmp_path, opt):
    kw = dict(body_model_path=ckpt)
    if opt == "eyes":
        kw.update(include_eyes=True, eye_model_path=str(tmp_path / "x.pth"))
    elif opt == "edema":
        kw.update(include_edema=True)
    else:
        kw.update(include_swimbladder=True, swimbladder_model_path=str(tmp_path / "x.pth"))
    with pytest.raises(RuntimeError):
        seg.segmentation_pipeline(file_list=img_files, target_size=(16, 16),
                                  local_only=True, device="cpu", **kw)


def test_local_only_corrupt_checkpoint_raises(tmp_path):
    bad = tmp_path / "bad.pth"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(RuntimeError, match="Re-download"):
        seg._load_unet_model(model_path=str(bad), local_only=True, device="cpu")


def test_local_only_uses_weights_only(ckpt, monkeypatch):
    seen = {}
    real = torch.load

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(torch, "load", spy)
    m = seg._load_unet_model(model_path=ckpt, local_only=True, device="cpu")
    assert m is not None
    assert seen["weights_only"] is True
    assert seen["map_location"] == "cpu"


def test_default_mode_missing_returns_none_and_empty_masks(ckpt, img_files, fake_seg, tmp_path):
    hub = sys.modules["huggingface_hub"]
    hub.hf_hub_download = lambda **k: (_ for _ in ()).throw(OSError("offline"))
    assert seg._load_unet_model(model_path=str(tmp_path / "n.pth"), label="x") is None
    out = seg.segmentation_pipeline(
        file_list=img_files, target_size=(16, 16), include_eyes=True, include_edema=True,
        include_swimbladder=True, body_model_path=ckpt, device="cpu",
    )
    assert len(out) == 6
    for masks in out[3:]:
        assert all(not m.any() and m.shape == (16, 16) for m in masks)


def test_swimbladder_options(ckpt, img_files, fake_seg, monkeypatch):
    offsets = []

    def rec(mask, offset):
        offsets.append(offset)
        return mask

    monkeypatch.setattr(seg, "adjust_mask_boundary", rec)
    common = dict(file_list=img_files, target_size=(16, 16), include_swimbladder=True,
                  device="cpu", body_model_path=ckpt, swimbladder_model_path=ckpt)
    seg.segmentation_pipeline(**common)
    assert offsets == [-0.73, -0.73]
    assert [c["biggest_only"] for c in fake_seg][1::2] == [True, True]

    offsets.clear()
    fake_seg.clear()
    seg.segmentation_pipeline(swimbladder_biggest_only=False,
                              swimbladder_boundary_offset_px=None, **common)
    assert offsets == []
    assert [c["biggest_only"] for c in fake_seg][1::2] == [False, False]

    seg.segmentation_pipeline(swimbladder_boundary_offset_px=0, **common)
    assert offsets == []


def test_device_honoured_and_cache_key(ckpt, monkeypatch):
    m = seg._load_unet_model(model_path=ckpt, device="cpu")
    assert next(m.parameters()).device.type == "cpu"
    m2 = seg._load_unet_model(model_path=ckpt, device=torch.device("cpu"))
    assert m2 is m
    assert all(k[-1] == "cpu" for k in seg._UNET_CACHE)
    # local_only loads on cpu then moves, so a "meta" device works as a stand-in for cuda
    m3 = seg._load_unet_model(model_path=ckpt, local_only=True, device="meta")
    assert m3 is not m
    assert {k[-1] for k in seg._UNET_CACHE} == {"cpu", "meta"}


def test_pipeline_resolves_load_unet_model_at_call_time(ckpt, img_files, fake_seg, monkeypatch):
    calls = []

    def patched(*a, **k):
        calls.append((a, k))
        return FakeModel()

    monkeypatch.setattr(seg, "_load_unet_model", patched)
    seg.segmentation_pipeline(
        file_list=img_files, target_size=(16, 16), include_eyes=True, include_edema=True,
        include_swimbladder=True, local_only=True, device="cpu", body_model_path="/b.pth",
    )
    assert len(calls) == 4
    for a, k in calls:
        assert a == ()
        assert k["local_only"] is True
        assert k["device"] == "cpu"
    assert calls[0][1]["model_path"] == "/b.pth"
