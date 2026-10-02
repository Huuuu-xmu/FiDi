"""Check that a participant cannot leak across partitions."""

import csv
import tempfile
import unittest
from pathlib import Path

from fidi.manifest import fold_partition, read_manifest


class ManifestCases(unittest.TestCase):
    def test_fold_assignment_and_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("a.png", "b.png", "c.png", "d.png", "e.png"):
                (root / name).write_bytes(b"placeholder")
            manifest = root / "manifest.csv"
            rows = [
                ["a", "a.png", "a.png", "p1", "r1", "development", "1"],
                ["b", "b.png", "b.png", "p2", "r2", "development", "2"],
                ["c", "c.png", "c.png", "p3", "r3", "test", ""],
                ["d", "d.png", "d.png", "p1", "r1", "development", "2"],
            ]
            with manifest.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["image_id", "image_path", "mask_path", "participant_id", "recording_id", "split", "fold"])
                writer.writerows(rows)
            parsed = read_manifest(manifest)
            train, val, test = fold_partition(parsed, 1)
            self.assertEqual((len(train), len(val), len(test)), (2, 1, 1))
            with self.assertRaisesRegex(ValueError, "multiple development folds"):
                read_manifest(manifest, group_folds_by_participant=True)
            rows.append(["e", "e.png", "e.png", "p1", "r4", "test", ""])
            with manifest.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["image_id", "image_path", "mask_path", "participant_id", "recording_id", "split", "fold"])
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "development and test"):
                read_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
