"""Summarize five model reports while keeping the models as distinct runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs=5, type=Path, help="Five evaluate_masks.py JSON reports")
    parser.add_argument("--output", type=Path, help="Optional JSON summary path")
    args = parser.parse_args()
    if len({path.resolve() for path in args.reports}) != 5:
        raise SystemExit("Provide five distinct report files")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    if any(report.get("crop_reference") != reports[0].get("crop_reference") for report in reports[1:]):
        raise SystemExit("Five reports must use the same reference preprocessing")
    if any(report.get("reference_preprocessing") != reports[0].get("reference_preprocessing") for report in reports[1:]):
        raise SystemExit("Five reports must use the same reference resize and crop")
    if any(report.get("metric_conventions") != reports[0].get("metric_conventions") for report in reports[1:]):
        raise SystemExit("Five reports must use identical metric conventions")
    image_lists = [report.get("image_ids") for report in reports]
    if not image_lists[0] or any(ids != image_lists[0] for ids in image_lists[1:]):
        raise SystemExit("Five reports must evaluate the same image IDs in the same order")
    hashes = [report.get("reference_sha256") for report in reports]
    if not hashes[0] or any(item != hashes[0] for item in hashes[1:]):
        raise SystemExit("Five reports must use identical reference mask files")
    if any(report.get("image_sha256") != reports[0].get("image_sha256") for report in reports[1:]):
        raise SystemExit("Five reports must use identical source images where hashes are available")
    if any(report.get("model_config") != reports[0].get("model_config") for report in reports[1:]):
        raise SystemExit("Five reports must use the same model and training configuration")
    names = reports[0]["metrics"].keys()
    if any(report["metrics"].keys() != names for report in reports[1:]):
        raise SystemExit("Five reports must contain the same metrics")
    summary = {}
    for name in names:
        values = [report["metrics"][name]["mean"] for report in reports]
        defined = sum(value is not None for value in values)
        summary[name] = {
            "mean_of_models": mean(values) if defined == 5 else None,
            "sd_of_models": stdev(values) if defined == 5 else None,
            "n_models_defined": defined,
            "n_models_undefined": 5 - defined,
            "model_values": values,
        }
    body = json.dumps({
        "status": "reconstructed_summary_not_patient_level_uncertainty",
        "n_models": 5,
        "n_images_per_model": len(image_lists[0]),
        "metrics": summary,
    }, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body + "\n", encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
