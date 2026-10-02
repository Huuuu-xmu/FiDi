"""Create a tiny synthetic manifest for checking the training command."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "smoke_data")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(1, 8):
        phase = "closed" if index % 2 else "open"
        stem = f"smoke_{index:02d}"
        rgb = f"{stem}_rgb.png"
        gt = f"{stem}_gt.png"
        shutil.copyfile(HERE / "synthetic" / f"synthetic_{phase}_rgb.png", args.output / rgb)
        shutil.copyfile(HERE / "synthetic" / f"synthetic_{phase}_gt.png", args.output / gt)
        rows.append([
            stem, rgb, gt, f"synthetic_p{index:02d}", f"synthetic_r{index:02d}",
            "test" if index > 5 else "development", "" if index > 5 else str(index),
        ])
    manifest = args.output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image_id", "image_path", "mask_path", "participant_id", "recording_id", "split", "fold"))
        writer.writerows(rows)
    print(f"Generated synthetic smoke-test manifest: {manifest}")


if __name__ == "__main__":
    main()
