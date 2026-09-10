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
