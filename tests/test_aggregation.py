"""Check five-model arithmetic and preserve missing metric values."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_models.py"


class AggregationCases(unittest.TestCase):
    def make_reports(self, root):
        paths = []
        for index, value in enumerate((0, 0.25, 0.5, 0.75, 1)):
            report = {
                "image_ids": ["synthetic_1"],
                "reference_sha256": ["same_reference"],
                "metrics": {
                    "mIoU": {"mean": value},
                    "ASSD-C": {"mean": None if index == 4 else value},
                },
            }
            path = root / f"model_{index}.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            paths.append(path)
        return paths

    def run_summary(self, paths):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *map(str, paths)],
            capture_output=True, text=True,
        )

    def test_five_model_mean_and_missing_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_summary(self.make_reports(Path(temporary)))
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)["metrics"]
            self.assertEqual(summary["mIoU"]["mean_of_models"], 0.5)
            self.assertAlmostEqual(summary["mIoU"]["sd_of_models"], (0.625 / 4) ** 0.5)
            self.assertIsNone(summary["ASSD-C"]["mean_of_models"])
            self.assertEqual(summary["ASSD-C"]["n_models_defined"], 4)

    def test_repeated_report_and_reference_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_reports(Path(temporary))
            self.assertNotEqual(self.run_summary([paths[0]] * 5).returncode, 0)
            changed = json.loads(paths[-1].read_text())
            changed["reference_sha256"] = ["different_reference"]
            paths[-1].write_text(json.dumps(changed))
            self.assertNotEqual(self.run_summary(paths).returncode, 0)


if __name__ == "__main__":
    unittest.main()
