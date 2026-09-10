"""
zebrafish_measure: segmentation, measurement, and scale-bar calibration
functions for zebrafish larvae images.

Shared by the Zebrafish_webapp Gradio app and the SlicerZebrafishEmbryoAnalyzer
3D Slicer extension.
"""

from .length import (
    compute_eye_metrics,
    compute_eye_diameters,
    compute_tube_metrics,
    tube_length_border2border,
    normalize_images,
    apply_mask,
    classification_curvature,
    FishClassifier,
    load_model,
    plot_edges_with_curvature,
    compute_curvature_profile,
    compute_curvature,
    preprocess_masked_image,
)
from .seg import segmentation_pipeline
from .seg_helper import (
    load_images_from_path,
    segment_fish,
    fill_holes,
    grow_mask,
    adjust_mask_boundary,
)
from .scalebar import (
    detect_scalebar,
    draw_scalebar_endpoints,
    calibrate_from_endpoints,
)

__all__ = [
    "compute_eye_metrics",
    "compute_eye_diameters",
    "compute_tube_metrics",
    "tube_length_border2border",
    "normalize_images",
    "apply_mask",
    "classification_curvature",
    "FishClassifier",
    "load_model",
    "plot_edges_with_curvature",
    "compute_curvature_profile",
    "compute_curvature",
    "preprocess_masked_image",
    "segmentation_pipeline",
    "load_images_from_path",
    "segment_fish",
    "fill_holes",
    "grow_mask",
    "adjust_mask_boundary",
    "detect_scalebar",
    "draw_scalebar_endpoints",
    "calibrate_from_endpoints",
]

__version__ = "0.1.0"
