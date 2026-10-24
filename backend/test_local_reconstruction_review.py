from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from review_local_reconstruction import review


class LocalReviewTests(unittest.TestCase):
    def test_missing_eval_is_not_presented_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                review(Path(directory), Path(directory))

    def test_identical_render_is_json_safe_not_infinity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images').mkdir()
            (root / 'exports/eval_1').mkdir(parents=True)
            image = np.full((24, 32, 3), 120, np.uint8)
            cv2.imwrite(str(root / 'images/00001.png'), image)
            cv2.imwrite(str(root / 'exports/eval_1/00001.png'), image)
            result = review(root, root / 'exports')
            self.assertEqual(result['held_out_views'], 1)
            self.assertIsNone(result['measurements'][0]['psnr_db'])
            self.assertTrue(result['measurements'][0]['identical_pixels'])


if __name__ == '__main__':
    unittest.main()
