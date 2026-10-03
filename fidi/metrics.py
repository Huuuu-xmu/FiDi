"""Pixel-label segmentation metrics and image-level summaries.

Each image contributes equally. A class with no reference and no prediction is
omitted for that image; a one-sided empty mask scores zero for F1/IoU/Dice.
Surface distances require two nonempty boundaries and are otherwise undefined.
"""

from __future__ import annotations

from statistics import mean

import numpy as np


BACKGROUND = 0
CLOSED_GLOTTIS = 1
OPEN_GLOTTIS = 4
GLOTTIS = (1, 4)
VOCAL_FOLDS = (2, 3, 5, 6)
FOREGROUND = tuple(range(1, 7))
CONVENTIONS = {
    "macro_classes": list(FOREGROUND),
    "background_in_macro": False,
    "f1_evaluation_unit": "pixel_label_not_independent_image_phase",
    "group_f1": "per_image_micro_over_separate_class_labels",
    "within_model_aggregation": "mean_of_defined_image_scores",
    "both_empty": "undefined_excluded",
    "one_empty_overlap": 0,
    "one_empty_surface_distance": "undefined_excluded",
    "surface_distance_unit": "pixels",
}


def _binary_score(pred: np.ndarray, truth: np.ndarray) -> tuple[float | None, float | None]:
    predicted = int(np.count_nonzero(pred))
    actual = int(np.count_nonzero(truth))
    if predicted + actual == 0:
        return None, None
    intersection = int(np.count_nonzero(pred & truth))
    union = predicted + actual - intersection
    return 2 * intersection / (predicted + actual), intersection / union


def _group_micro_f1(pred: np.ndarray, truth: np.ndarray, labels: tuple[int, ...]) -> float | None:
    """Pool one-vs-rest counts, retaining errors between labels in the group."""
    predicted = sum(int(np.count_nonzero(pred == label)) for label in labels)
    actual = sum(int(np.count_nonzero(truth == label)) for label in labels)
    if predicted + actual == 0:
        return None
    true_positive = sum(int(np.count_nonzero((pred == label) & (truth == label))) for label in labels)
    return 2 * true_positive / (predicted + actual)


def _surface_distances(pred: np.ndarray, truth: np.ndarray) -> tuple[float | None, float | None]:
    if not pred.any() or not truth.any():
        return None, None
    from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

    structure = generate_binary_structure(2, 1)
    pred_surface = pred ^ binary_erosion(pred, structure=structure, border_value=0)
    truth_surface = truth ^ binary_erosion(truth, structure=structure, border_value=0)
    truth_dist = distance_transform_edt(~truth_surface)[pred_surface]
    pred_dist = distance_transform_edt(~pred_surface)[truth_surface]
    # Table S4 defines ASSD over all boundary points and HD95 as the larger
    # directed 95th percentile, rather than a percentile of pooled distances.
    assd = float((truth_dist.sum() + pred_dist.sum()) / (len(truth_dist) + len(pred_dist)))
    hd95 = float(max(np.percentile(truth_dist, 95), np.percentile(pred_dist, 95)))
    return assd, hd95


def image_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float | None]:
    if pred.ndim != 2 or pred.shape != truth.shape:
        raise ValueError("Prediction and reference must be same-size 2D arrays")
    if not np.isin(pred, range(7)).all() or not np.isin(truth, range(7)).all():
        raise ValueError("Mask labels must be integers 0..6")

    f1s = []
    ious = []
    for cls in FOREGROUND:
        f1, iou = _binary_score(pred == cls, truth == cls)
        if f1 is not None:
            f1s.append(f1)
            ious.append(iou)
    f1_g = _group_micro_f1(pred, truth, GLOTTIS)
    f1_v = _group_micro_f1(pred, truth, VOCAL_FOLDS)

    # Closed-phase measures are defined only when class 1 is present in the
    # reference. An absent prediction then has Dice 0 and undefined distances.
    closed_truth = truth == CLOSED_GLOTTIS
    if closed_truth.any():
        closed_pred = pred == CLOSED_GLOTTIS
        dice_c, _ = _binary_score(closed_pred, closed_truth)
        assd_c, hd95_c = _surface_distances(closed_pred, closed_truth)
    else:
        dice_c = assd_c = hd95_c = None
    return {
        "pixel-F1-G": f1_g,
        "pixel-F1-V": f1_v,
        "pixel-mF1": mean(f1s) if f1s else None,
        "mIoU": mean(ious) if ious else None,
        "Dice-C": dice_c,
        "ASSD-C": assd_c,
        "HD95-C": hd95_c,
    }


def aggregate_images(images: list[dict[str, float | None]]) -> dict[str, dict[str, float | int | None]]:
    if not images:
        raise ValueError("No images to evaluate")
    out = {}
    for key in images[0]:
        values = [item[key] for item in images if item[key] is not None]
        out[key] = {
            "mean": float(mean(values)) if values else None,
            "n_defined": len(values),
            "n_undefined": len(images) - len(values),
        }
    return out
