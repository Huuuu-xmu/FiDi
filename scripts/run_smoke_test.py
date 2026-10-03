"""Check the workflow using synthetic images only; this is not a benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fidi.provenance import execution_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "smoke_cpu")
    parser.add_argument("--all-folds", action="store_true", help="Exercise all five folds and their summary")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = str(ROOT / ".local" / "matplotlib")
    started = time.monotonic()

    def run(label, command):
        print(f"Running {label}...", flush=True)
        log_path = output / f"{label}.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [sys.executable, *command], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, timeout=300,
            )
        if completed.returncode:
            print(log_path.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
            raise SystemExit(f"{label} failed; inspect {log_path}")
        print(f"Completed {label}.", flush=True)

    run("prepare_model", ["scripts/prepare_smp.py"])
    run("generate_data", ["examples/generate_smoke_data.py"])
    folds = range(1, 6) if args.all_folds else (1,)
    for fold in folds:
        run(f"train_fold_{fold}", [
            "scripts/train_fold.py", "--manifest", "examples/smoke_data/manifest.csv",
            "--fold", str(fold), "--output", str(output / f"fold_{fold}"),
            "--epochs", "1", "--batch-size", "1", "--input-size", "64",
            "--no-pretrained", "--device", "cpu",
        ])
    run("predict", [
        "scripts/predict_masks.py", "--images", "examples/synthetic",
        "--checkpoint", str(output / "fold_1" / "best_state_dict.pth"),
        "--output", str(output / "predictions"), "--input-size", "64",
        "--device", "cpu",
    ])
    run("evaluate", [
        "scripts/evaluate_masks.py", "--truth", "examples/synthetic",
        "--prediction", str(output / "predictions"), "--crop-reference",
        "--input-size", "64",
        "--output", str(output / "mask_report.json"),
    ])
    if args.all_folds:
        run("aggregate", [
            "scripts/aggregate_models.py",
            *[str(output / f"fold_{fold}" / "test_report.json") for fold in folds],
            "--output", str(output / "five_model_summary.json"),
        ])
    result = {
        "status": "synthetic_execution_check_passed",
        "clinical_performance_verified": False,
        "folds_checked": list(folds),
        "data": "synthetic repeated image pairs",
        "input_size": 64,
        "epochs_per_fold": 1,
        "device": "cpu",
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "execution": execution_metadata(),
    }
    (output / "execution_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Synthetic workflow passed. Logs and reports: {output}")


if __name__ == "__main__":
    main()
