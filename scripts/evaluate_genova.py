"""Calculate IoU and Dice from six-class Genova masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True, help="Directory of *_gt.png class-index masks")
    parser.add_argument("--prediction", type=Path, required=True, help="Directory of matching *_pred.png masks")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    references = sorted(args.truth.glob("*_gt.png"))
    if not references:
        parser.error("No *_gt.png masks found")
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in range(1, 7)}
    for reference in references:
        prediction = args.prediction / reference.name.replace("_gt.png", "_pred.png")
        if not prediction.is_file():
            parser.error(f"Missing prediction: {prediction}")
        truth = np.asarray(Image.open(reference))
        pred = np.asarray(Image.open(prediction))
        if truth.shape != pred.shape or truth.ndim != 2:
            parser.error(f"Mask shape mismatch or non-2D mask: {reference.name}")
        if not np.isin(truth, range(7)).all() or not np.isin(pred, range(7)).all():
            parser.error(f"Expected label values 0..6: {reference.name}")
        for label in counts:
            ref_class = truth == label
            pred_class = pred == label
            counts[label]["tp"] += int(np.count_nonzero(ref_class & pred_class))
            counts[label]["fp"] += int(np.count_nonzero(~ref_class & pred_class))
            counts[label]["fn"] += int(np.count_nonzero(ref_class & ~pred_class))

    classes = {}
    for label, value in counts.items():
        tp, fp, fn = value["tp"], value["fp"], value["fn"]
        if tp + fp + fn == 0:
            parser.error(f"Class {label} is absent from references and predictions")
        classes[str(label)] = {
            "IoU": tp / (tp + fp + fn),
            "Dice": 2 * tp / (2 * tp + fp + fn),
        }
    report = {
        "status": "six_class_evaluation_completed",
        "n_images": len(references),
        "pixel_aggregation": "pooled_over_images_per_class",
        "macro_aggregation": "unweighted_mean_over_classes_1_through_6",
        "class_scores": classes,
        "mean_IoU": float(np.mean([item["IoU"] for item in classes.values()])),
        "mean_Dice": float(np.mean([item["Dice"] for item in classes.values()])),
    }
    body = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body + "\n", encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
