"""Check native-resolution crop geometry and reject implicit resize/padding."""

import unittest
import numpy as np
from PIL import Image
from fidi.data import _crop


class CropCases(unittest.TestCase):
    def test_crop_preserves_native_pixel_coordinates(self):
        source = np.arange(8 * 10, dtype=np.uint8).reshape(8, 10)
        cropped = np.asarray(_crop(Image.fromarray(source), input_size=4))
        np.testing.assert_array_equal(cropped, source[2:6, 3:7])

    def test_undersized_image_is_rejected_without_padding(self):
        with self.assertRaises(ValueError):
            _crop(Image.new("RGB", (4, 3)), input_size=4)


if __name__ == "__main__":
    unittest.main()
