import unittest
from contextlib import ExitStack
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image

from reconstruction.depth_track_refinement import refine_depth
from reconstruction.photographic_walk import enhance_walk


class DepthTrackRefinementTests(unittest.TestCase):
    def setUp(self):
        self.depth = np.linspace(1., 5., 400).reshape(20, 20)
        self.mask = np.ones_like(self.depth, bool)
        self.ids = np.arange(400)
        self.e = self.depth.ravel()

    def test_consistent_depth_is_unchanged(self):
        result, report = refine_depth(self.depth, self.mask, self.e, self.e, self.ids)
        np.testing.assert_array_equal(result, self.depth)
        self.assertFalse(report['selected'])

    def test_bias_corrected_without_changing_mask_or_invalid_pixels(self):
        target = 1 / (.8 / self.e + .02)
        mask = self.mask.copy(); mask[0, 0] = False
        original = self.depth.copy(); original[0, 0] = np.nan
        result, report = refine_depth(original, mask, self.e, target, self.ids)
        self.assertTrue(report['selected'])
        np.testing.assert_allclose(result[mask], target.reshape(20, 20)[mask], rtol=1e-4)
        self.assertTrue(np.isnan(result[0, 0]))
        self.assertFalse(report['quality_accepted'])
        self.assertTrue(mask[1:, :].all())

    def test_held_out_ids_are_not_fitted(self):
        target = 1 / (.8 / self.e + .02)
        target[self.ids % 5 == 0] = self.e[self.ids % 5 == 0]
        result, report = refine_depth(self.depth, self.mask, self.e, target, self.ids)
        self.assertFalse(report['selected'])
        np.testing.assert_array_equal(result, self.depth)

    def test_extrapolation_pole_is_rejected(self):
        e = np.linspace(1., 2., 400)
        target = 1 / (1 / e - .3)
        result, report = refine_depth(self.depth, self.mask, e, target, self.ids)
        self.assertFalse(report['selected'])
        np.testing.assert_array_equal(result, self.depth)

    def test_insufficient_or_flat_tracks_preserve_original(self):
        for e, r, ids in [(self.e[:20], self.e[:20], self.ids[:20]),
                          (np.ones(400), np.ones(400) * 2, self.ids)]:
            result, report = refine_depth(self.depth, self.mask, e, r, ids)
            self.assertFalse(report['selected'])
            np.testing.assert_array_equal(result, self.depth)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            refine_depth(self.depth, self.mask, self.e, self.e, np.zeros(400, int))
        with self.assertRaises(ValueError):
            refine_depth(self.depth, self.mask, self.e, self.e * np.nan, self.ids)


class DepthRefinementIntegrationTests(unittest.TestCase):
    def run_walk(self, fail_candidate_export=False, enabled=True):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root=Path(directory);images=root/'poses/brush_dataset/images';images.mkdir(parents=True)
            points=np.zeros((24,32,3));points[:,:,2]=3
            geometry=root/'cached-depth.npz';np.savez(geometry,points=points,mask=np.ones((24,32),bool))
            cameras=[]
            for index in range(3):
                name=f'{index:05d}.jpg';Image.new('RGB',(64,48),'tan').save(images/name)
                cameras.append(dict(name=name,position=[0,0,0],rotation=np.eye(3).tolist(),
                                    intrinsics=[64,48,40,40,32,24]))
            config=dict(cameras=cameras)
            stack.enter_context(patch('serve_splat_review.config',return_value=dict(cameras=cameras)))
            stack.enter_context(patch('reconstruction.moge_geometry.MoGeGeometryRefiner.infer',return_value=(geometry,{})))
            stack.enter_context(patch('reconstruction.photographic_finish.depth_scale',return_value=(1.,{})))
            def refined(observations,dataset):
                return [(camera,np.full_like(depth,2.),mask,photo) for camera,depth,mask,photo in observations], dict(applied=True)
            refine_call=stack.enter_context(patch('reconstruction.depth_track_refinement.refine_observations',side_effect=refined))
            def exported(path,*args,**kwargs):
                if fail_candidate_export and path.name=='surface-51.glb':raise ValueError('fixture export failure')
                return dict(vertices=768,triangles=1426)
            stack.enter_context(patch('reconstruction.photographic_walk.export_surface',side_effect=exported))
            def navigation(samples,cameras):
                return dict(bounds=dict(min=[-5,-2,-5],max=[5,2,1]),
                            fixture_median_z=float(np.median(np.concatenate(samples)[:,2])))
            stack.enter_context(patch('reconstruction.photographic_walk.navigation_map',side_effect=navigation))
            fusion=stack.enter_context(patch('reconstruction.photographic_fusion.export_fused',return_value=dict(triangles=100)))
            if enabled is None:
                with patch.dict('os.environ',{},clear=True):result=enhance_walk(root,config)
            else:result=enhance_walk(root,config,refine_tracks=enabled)
            if not enabled:refine_call.assert_not_called()
            actual=fusion.call_args.args[1]
            return result,[depth.copy() for camera,depth,mask,photo in actual]

    def test_corrected_visible_surfaces_and_collision_samples_use_same_depth(self):
        result,depths=self.run_walk()
        self.assertTrue(result['walk_support']['depth_track_refinement']['applied'])
        self.assertEqual([item['asset'] for item in result['surfaces']],
                         ['surface-50.glb','surface-51.glb','surface-52.glb'])
        self.assertTrue(all(np.all(depth==2) for depth in depths))
        # Navigation is generated from the corrected extent, not the stale mesh.
        self.assertAlmostEqual(result['navigation']['fixture_median_z'],-2.)

    def test_partial_optional_export_failure_keeps_original_geometry_and_collisions(self):
        result,depths=self.run_walk(fail_candidate_export=True)
        self.assertFalse(result['walk_support']['depth_track_refinement']['applied'])
        self.assertEqual([item['asset'] for item in result['surfaces']],
                         ['surface-00.glb','surface-01.glb','surface-02.glb'])
        self.assertTrue(all(np.all(depth==3) for depth in depths))
        self.assertAlmostEqual(result['navigation']['fixture_median_z'],-3.)

    def test_unaccepted_calibration_is_disabled_by_default(self):
        result,depths=self.run_walk(enabled=None)
        self.assertFalse(result['walk_support']['depth_track_refinement']['applied'])
        self.assertTrue(all(np.all(depth==3) for depth in depths))


if __name__ == '__main__':
    unittest.main()
