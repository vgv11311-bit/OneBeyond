import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from reconstruction.point_evidence import filter_points_by_quantile


class PointEvidenceTests(unittest.TestCase):
    def test_lateral_outlier_with_same_depth_is_removed(self):
        cloud = np.column_stack((np.linspace(-1, 1, 100), np.zeros(100), np.ones(100)))
        cloud = np.vstack((cloud, [500, 0, 1]))
        original = cloud.copy()
        result = filter_points_by_quantile(cloud)
        self.assertLessEqual(result[:, 0].max(), 1)
        np.testing.assert_array_equal(cloud, original)

    def test_planar_and_identical_clouds_survive(self):
        cloud = np.ones((100, 3), dtype=np.float32)
        np.testing.assert_array_equal(filter_points_by_quantile(cloud), cloud)

    def test_empty_and_nonfinite(self):
        self.assertEqual(filter_points_by_quantile(np.empty((0, 3))).shape, (0, 3))
        cloud = np.array([[1, 2, 3], [np.nan, 0, 0], [0, np.inf, 0]])
        np.testing.assert_array_equal(filter_points_by_quantile(cloud), [[1, 2, 3]])

    def test_small_sample_and_aggressive_trim_fall_back(self):
        cloud = np.arange(120).reshape(40, 3)
        np.testing.assert_array_equal(filter_points_by_quantile(cloud, min_points=40), cloud)
        np.testing.assert_array_equal(filter_points_by_quantile(cloud, q=.49), cloud)

    def test_invalid_inputs(self):
        for q in (-1, .5, np.nan):
            with self.assertRaises(ValueError):
                filter_points_by_quantile(np.ones((40, 3)), q=q)
        with self.assertRaises(ValueError):
            filter_points_by_quantile(np.ones((10, 2)))

    def test_zero_quantile_preserves_finite_input(self):
        cloud = np.arange(120).reshape(40, 3)
        np.testing.assert_array_equal(filter_points_by_quantile(cloud, q=0), cloud)

    def test_refiner_integration_is_opt_in_and_reports_retention(self):
        from reconstruction.scene_analysis import SceneAnalysis, SceneObject
        from reconstruction.moge_geometry import MoGeGeometryRefiner
        yy, xx = np.mgrid[:40, :40]
        points = np.stack(((xx-20)/10, (yy-20)/10, 3+yy/100), axis=-1).astype(np.float32)
        points[15, 15, 0] = 500
        obj = SceneObject('cabinet', .9, (5, 5, 35, 35), [100]*3,
                          [0, .5, -3], [1, .5, 1], 3, 'test', .8)
        scene = SceneAnalysis([obj], 'test', (5, 6, 3), 20)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'geometry.npz'
            np.savez(path, points=points, mask=np.ones((40, 40), bool),
                     intrinsics=np.array([[1,0,.5],[0,1,.5],[0,0,1]]), source_shape=[40,40])
            for enabled in ('0', '1'):
                with patch.dict('os.environ', {'ONE_BEYOND_POINT_FILTER': enabled}):
                    result, details = MoGeGeometryRefiner.refine(scene, path)
                evidence = details['objects'][0]['point_filter']
                self.assertEqual(evidence['enabled'], enabled == '1')
                if enabled == '1':
                    self.assertLess(evidence['retained_points'], evidence['input_points'])
                else:
                    self.assertEqual(evidence['retained_points'], evidence['input_points'])
                self.assertTrue(np.isfinite(result.objects[0].size).all())
                self.assertEqual(result.objects[0].label, 'cabinet')


if __name__ == '__main__':
    unittest.main()
