"""Evaluate paired class-index PNG masks with proposed image-level metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fidi.data import load_mask
from fidi.metrics import CONVENTIONS, aggregate_images, image_metrics
from fidi.provenance import execution_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True, help="Directory of *_gt.png masks")
    parser.add_argument("--prediction", type=Path, required=True, help="Directory of matching *_pred.png masks")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--crop-reference", action="store_true", help="Apply training resize and center crop to reference masks")
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--resize-width", type=int, default=600)
    args = parser.parse_args()

    references = sorted(args.truth.glob("*_gt.png"))
    if not references:
        raise SystemExit("No *_gt.png references found")
    if args.input_size < 32 or args.input_size % 32 or args.resize_width < args.input_size:
        parser.error("input-size must be a positive multiple of 32 and resize-width >= input-size")
    results = []
    for reference in references:
        prediction = args.prediction / reference.name.replace("_gt.png", "_pred.png")
        if not prediction.is_file():
            raise SystemExit(f"Missing prediction: {prediction}")
        truth = load_mask(
            reference, input_size=args.input_size, resize_width=args.resize_width
        ) if args.crop_reference else np.asarray(Image.open(reference))
        pred = np.asarray(Image.open(prediction))
        results.append(image_metrics(pred, truth))
    report = {
        "status": "reconstructed_metric_definition_not_verified_against_manuscript",
        "execution": execution_metadata(),
        "n_images": len(results),
        "image_ids": [path.name for path in references],
        "reference_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in references],
        "crop_reference": args.crop_reference,
        "reference_preprocessing": {
            "input_size": args.input_size,
            "resize_width": args.resize_width,
        } if args.crop_reference else None,
        "metric_conventions": CONVENTIONS,
        "metrics": aggregate_images(results),
        "per_image": [
            {"image_id": path.name, "metrics": values}
            for path, values in zip(references, results)
        ],
    }
    body = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body + "\n", encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
