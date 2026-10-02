"""Small hand-checkable cases for the proposed metrics."""

import unittest

import numpy as np

from fidi.metrics import aggregate_images, image_metrics


class MetricCases(unittest.TestCase):
    def test_identical_closed_mask_and_empty_open_mask(self):
        truth = np.zeros((5, 5), dtype=np.uint8)
        truth[2, 2] = 1
        closed = image_metrics(truth, truth)
        empty = image_metrics(np.zeros_like(truth), np.zeros_like(truth))
        self.assertEqual(closed["Dice-C"], 1.0)
        self.assertEqual(closed["ASSD-C"], 0.0)
        self.assertEqual(closed["HD95-C"], 0.0)
        self.assertIsNone(empty["Dice-C"])
        self.assertIsNone(empty["mIoU"])
        self.assertEqual(aggregate_images([closed, empty])["Dice-C"]["n_defined"], 1)

    def test_single_pixel_shift_and_missing_prediction(self):
        truth = np.zeros((5, 5), dtype=np.uint8)
        pred = np.zeros_like(truth)
        truth[2, 2] = 1
        pred[2, 3] = 1
        shifted = image_metrics(pred, truth)
        self.assertEqual(shifted["Dice-C"], 0.0)
        self.assertEqual(shifted["ASSD-C"], 1.0)
        self.assertEqual(shifted["HD95-C"], 1.0)
        missing = image_metrics(np.zeros_like(truth), truth)
        self.assertEqual(missing["Dice-C"], 0.0)
        self.assertIsNone(missing["ASSD-C"])

    def test_phase_label_confusion_is_not_counted_as_correct_glottis(self):
        truth = np.zeros((3, 3), dtype=np.uint8)
        pred = np.zeros_like(truth)
        truth[1, 1] = 1
        pred[1, 1] = 4
        self.assertEqual(image_metrics(pred, truth)["F1-G"], 0.0)

    def test_asymmetric_surface_matches_table_s4_formula(self):
        truth = np.zeros((5, 5), dtype=np.uint8)
        pred = np.zeros_like(truth)
        truth[2, 1] = 1
        pred[2, 1:4] = 1
        result = image_metrics(pred, truth)
        self.assertAlmostEqual(result["ASSD-C"], 0.75)
        self.assertAlmostEqual(result["HD95-C"], 1.9)


if __name__ == "__main__":
    unittest.main()
