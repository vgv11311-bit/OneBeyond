import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from reconstruction.multiview_evidence import triangulate_pair, analyze_views


class MultiViewEvidenceTests(unittest.TestCase):
    def setUp(self):
        random = np.random.default_rng(27)
        self.points = np.column_stack((random.uniform(-1, 1, 100),
                                       random.uniform(-.6, .6, 100),
                                       random.uniform(2.5, 5, 100)))
        self.k = np.array([[650., 0, 320.], [0, 650., 240.], [0, 0, 1.]])
        self.primary = self.points[:, :2] / self.points[:, 2, None] * 650 + [320, 240]
        moved = self.points + [-.5, 0, 0]
        self.secondary = moved[:, :2] / moved[:, 2, None] * 650 + [320, 240]
        self.map = np.zeros((480, 640, 3), dtype=np.float32)
        self.valid = np.zeros((480, 640), dtype=bool)
        for pixel, point in zip(self.primary, self.points):
            x, y = np.rint(pixel).astype(int)
            self.map[y, x] = point
            self.valid[y, x] = True

    def test_metric_tracks_are_recovered_from_second_view(self):
        result = triangulate_pair(self.primary, self.secondary, self.k, self.k,
                                  (480, 640), self.map, self.valid)
        self.assertEqual(result['status'], 'metric_consistent')
        self.assertGreaterEqual(result['metric_tracks'], 75)
        self.assertLess(result['median_reprojection_error_px'], .5)
        self.assertLess(result['median_point_disagreement_m'], .1)
        self.assertEqual(result['relative_camera_pose']['translation_scale'],
                         'monocular_depth_aligned_estimate')
        self.assertEqual(len(result['relative_camera_pose']['translation_primary_to_secondary_m']),3)

    def test_identical_view_is_not_claimed_as_three_dimensional_evidence(self):
        result = triangulate_pair(self.primary, self.primary, self.k, self.k,
                                  (480, 640), self.map, self.valid)
        self.assertNotEqual(result['status'], 'metric_consistent')

    def test_single_and_unreadable_capture_sets_are_nonmetric(self):
        with tempfile.TemporaryDirectory() as directory:
            primary=Path(directory)/'first.png'
            cv2.imwrite(str(primary),np.zeros((128,128),np.uint8))
            single=analyze_views([primary])
            self.assertEqual(single['status'],'single_photo')
            self.assertFalse(single['geometry_verified'])
            missing=analyze_views([Path(directory)/'missing.png',primary])
            self.assertEqual(missing['status'],'primary_unreadable')
            self.assertEqual(missing['metric_tracks'],0)


if __name__ == '__main__':
    unittest.main()
