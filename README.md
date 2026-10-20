# zebrafish-measure

Segmentation, measurement, and scale-bar calibration functions for zebrafish
larvae images, factored out of the [Zebrafish_webapp](../Zebrafish_webapp)
Gradio app so they can be shared with the
[SlicerZebrafishEmbryoAnalyzer](../SlicerZebrafishEmbryoAnalyzer) 3D Slicer
extension without depending on Gradio.

## Install

```bash
pip install -e .
```

## Modules

- `zebrafish_measure.seg` — `segmentation_pipeline`: runs the body/eye/edema/
  swim-bladder Unet/FPN/Segformer models (downloaded from Hugging Face Hub)
  over a folder or list of images.
- `zebrafish_measure.seg_helper` — mask post-processing used by `seg`
  (hole filling, dilation, sub-pixel boundary offset) and image loading.
- `zebrafish_measure.length` — measurement geometry: fish body length via
  `tube_length_border2border`, eye metrics (`compute_eye_metrics`,
  `compute_eye_diameters`), and generic tube-shaped structure metrics
  (`compute_tube_metrics`, used for e.g. the swim bladder). Also includes the
  curvature classifier (`FishClassifier`, `load_model`,
  `classification_curvature`) and the standalone edge-curvature analysis
  (`compute_curvature_profile`).
- `zebrafish_measure.scalebar` — auto-detection and manual calibration of the
  image scale bar (`detect_scalebar`, `calibrate_from_endpoints`).

All public functions are re-exported from the top-level package, e.g.:

```python
from zebrafish_measure import segmentation_pipeline, tube_length_border2border, detect_scalebar
```

## Scope

This library intentionally excludes anything Gradio/UI-specific — that stays
in `Zebrafish_webapp`'s `app.py`, which depends on this package instead of
containing the logic directly.

## What's new in 0.2.0

- The package imports without `torch`, `timm`, `segmentation-models-pytorch`,
  `huggingface_hub`, `matplotlib` or `torchvision` installed; the heavy
  modules are imported lazily on first use.
- `segmentation_pipeline` has new optional keyword parameters
  `body_model_path`, `local_only`, `device`, `swimbladder_biggest_only` and
  `swimbladder_boundary_offset_px`.
- Masks are returned in the fixed order eyes, edema, swim bladder, each
  independent of `include_eyes`.
- `load_model(model_path=None, device=None)` loads a local checkpoint safely
  when a path is given, and downloads from Hugging Face when `None` (as before).
- `select_torch_device` (CUDA probe with CPU fallback) is now public.
- `classification_curvature` uses the model's device and returns a CPU tensor.
- Removed the module-level globals `seg.device` and `length.device`; use
  `select_torch_device` or the `device` parameters instead.
- `seg._MODEL_TYPES` is now filled lazily on first use; an unknown
  `model_type` raises `AttributeError` (from the `segmentation_models_pytorch`
  lookup).
- `include_eyes=False` combined with `include_edema`/`include_swimbladder` now
  returns those masks (0.1.x silently returned only the first 3 items).

Defaults keep the 0.1.x behaviour.

## Local-only use

For applications that manage their own model downloads (no Hugging Face
access):

```python
from zebrafish_measure import segmentation_pipeline, load_model

result = segmentation_pipeline(
    file_list=["/path/to/fish.png"],
    body_model_path="/path/to/best_model_body.pth",
    eye_model_path="/path/to/best_model_eye.pth",
    include_eyes=True,
    local_only=True,
    device="cpu",
)
model = load_model("/path/to/best_model_class.pth", device="cpu")
```

## Installing without pulling dependencies

```bash
pip install zebrafish-measure --no-deps
```

Use this in environments that already provide the dependencies. The package
declares `opencv-python`, so environments using `opencv-python-headless`
(such as 3D Slicer) should install with `--no-deps` to avoid a second OpenCV
build. The following must then be present: numpy, opencv, pillow, scipy,
scikit-image, matplotlib, torch, torchvision, timm,
segmentation-models-pytorch, and huggingface_hub (only for download mode).
