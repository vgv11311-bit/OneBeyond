import json
import gzip
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import app as server
from reconstruction.base_engine import ReconstructionMetrics, ReconstructionResult
from reconstruction.engine_registry import EngineRegistry
from reconstruction.photographic_engine import PhotographicEngine
from reconstruction.photographic_finish import finish_capture, depth_scale
from reconstruction.gaussian_geometry import read_ply, write_ply
from reconstruction.gaussian_cache import training_key
from test_gaussian_geometry import fixture
from gaussian_job import build as build_training


class PhotographicEngineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        override = patch.object(server, 'SPACES_DIR', self.root)
        override.start()
        self.addCleanup(override.stop)
        self.client = server.app.test_client()

    def state(self, **values):
        state = dict(job_id='photo-room', status='uploaded', logs=[], file_paths=['a', 'b', 'c'], input_type='photos')
        state.update(values)
        server.save_job_state('photo-room', state)

    def test_default_engine_is_provider_free_and_advertised(self):
        self.assertIsInstance(EngineRegistry.get_engine(), PhotographicEngine)
        defaults = [engine for engine in self.client.get('/api/engines').json['engines'] if engine['is_default']]
        self.assertEqual([engine['id'] for engine in defaults], ['photographic'])

    def test_pure_capture_does_not_invoke_either_provider_or_a_semantic_companion(self):
        self.state(glb_url='old-model.glb')
        result = {'status': 'ready', 'vertices': 120, 'registered_views': 3, 'sparse_points': 45}
        with patch('reconstruction.photographic_engine.build_companion', return_value=result) as build, \
             patch('reconstruction.groq_scene_engine.GroqSceneEngine.reconstruct', side_effect=AssertionError('Groq called')), \
             patch('reconstruction.gemini_scene_engine.GeminiSceneEngine.reconstruct', side_effect=AssertionError('Gemini called')), \
             patch.object(server, 'build_companion', side_effect=AssertionError('Semantic companion called')):
            server.run_reconstruction_worker('photo-room', 'photographic', {})
        final = server.load_job_state('photo-room')
        self.assertEqual(final['status'], 'completed')
        self.assertEqual(final['gaussian']['status'], 'ready')
        self.assertNotIn('glb_url', final)
        self.assertEqual(final['metrics']['extra_details']['provider_calls'], 0)
        self.assertEqual(final['metrics']['total_frames_processed'], 3)
        self.assertTrue(build.call_args.kwargs['standalone'])
        self.assertIsNone(self.client.get('/api/spaces').json['spaces'][0]['glb_url'])

    def test_failed_training_is_failed_job_without_provider_fallback(self):
        self.state(gaussian={'status': 'ready'})
        with patch('reconstruction.photographic_engine.build_companion', return_value={'status':'unavailable','reason':'Insufficient coverage'}):
            server.run_reconstruction_worker('photo-room', 'photographic', {})
        state = server.load_job_state('photo-room')
        self.assertEqual(state['status'], 'failed')
        self.assertNotIn('gaussian', state)
        self.assertEqual(state['error'], 'Insufficient coverage')

    def test_single_photo_is_rejected_before_a_worker_or_api_call(self):
        self.state(file_paths=['a'])
        with patch.object(server.threading.Thread, 'start') as launch:
            response = self.client.post('/api/reconstruct', json={'job_id':'photo-room'})
        self.assertEqual(response.status_code, 400)
        launch.assert_not_called()
        self.assertEqual(server.load_job_state('photo-room')['status'], 'uploaded')

    def test_training_modes_cannot_reuse_a_different_view_split(self):
        photo = self.root / 'photo.jpg'
        photo.write_bytes(b'photo')
        self.assertNotEqual(training_key([photo], {}, 'companion'), training_key([photo], {}, 'standalone'))

    def test_repeated_identical_uploads_do_not_start_training(self):
        self.state(input_fingerprints=['same', 'same', 'different'])
        with patch.object(server.threading.Thread, 'start') as launch:
            response = self.client.post('/api/reconstruct', json={'job_id':'photo-room'})
        self.assertEqual(response.status_code, 400)
        launch.assert_not_called()

    def test_finishing_preserves_raw_and_recovered_camera_geometry(self):
        values, cameras, header = fixture()
        cameras[0]['name'] = 'first.jpg'
        # A second observed camera verifies that normalization is one rigid
        # transformation of the complete capture, rather than per-view warping.
        cameras.append(dict(cameras[0], name='second.jpg', position=[.4, 0, 0]))
        (self.root / 'config.json').write_text(json.dumps(dict(cameras=cameras)))
        (self.root / 'manifest.json').write_text(json.dumps(dict(status='ready', vertices=len(values))))
        (self.root/'poses').mkdir()
        (self.root/'poses/video_selection.json').write_text(json.dumps(dict(candidate_count=2,selected_count=1,frames=[dict(status='selected',frame='private-frame-path'),dict(status='duplicate')])))
        write_ply(self.root / 'scene.ply', header, values)
        before = (self.root / 'scene.ply').read_bytes()
        with patch('reconstruction.moge_geometry.MoGeGeometryRefiner.is_available', return_value=False):
            result = finish_capture(self.root)
        self.assertEqual((self.root / 'raw-scene.ply').read_bytes(), before)
        _, output = read_ply(self.root / 'scene.ply')
        np.testing.assert_array_equal(output['f_dc_0'], values['f_dc_0'])
        config = json.loads((self.root / 'config.json').read_text())
        np.testing.assert_allclose(config['cameras'][0]['position'], [0, 0, 0])
        self.assertAlmostEqual(np.linalg.norm(np.subtract(config['cameras'][1]['position'], config['cameras'][0]['position'])), .4)
        self.assertNotIn('solid', config)
        self.assertNotIn('surface', config)
        self.assertIn('enclosure',config)
        self.assertTrue(np.isfinite(config['enclosure']['bounds']['min']).all())
        self.assertTrue(np.all(np.asarray(config['enclosure']['bounds']['max'])>config['enclosure']['bounds']['min']))
        self.assertEqual(result['provider_calls'], 0)
        self.assertEqual(result['frame_selection']['selected_frames'],1)
        self.assertEqual(config['capture_statistics'],result['frame_selection'])
        self.assertNotIn('private-frame-path',json.dumps(result['frame_selection']))
        self.assertEqual(gzip.decompress((self.root / 'scene.ply.gz').read_bytes()), (self.root / 'scene.ply').read_bytes())
        with self.assertRaisesRegex(ValueError, 'raw capture'):
            finish_capture(self.root)

    def test_lossless_delivery_respects_compression_preferences_and_range_reads(self):
        revision = 'c' * 32
        prefix = f'/api/spaces/photo-room/gaussian/{revision}'
        self.state(status='completed', gaussian={'status':'ready','ply_url':prefix+'/scene.ply'})
        target = self.root / 'photo-room/gaussian' / revision
        target.mkdir(parents=True)
        payload = b'ply test payload' * 100
        (target / 'scene.ply').write_bytes(payload)
        (target / 'scene.ply.gz').write_bytes(gzip.compress(payload))
        with self.client.get(prefix+'/scene.ply', headers={'Accept-Encoding':'gzip'}) as response:
            self.assertEqual(response.headers['Content-Encoding'], 'gzip')
            self.assertEqual(gzip.decompress(response.data), payload)
            self.assertIn('Accept-Encoding', response.headers['Vary'])
        with self.client.get(prefix+'/scene.ply', headers={'Accept-Encoding':'gzip;q=0'}) as response:
            self.assertNotIn('Content-Encoding', response.headers)
            self.assertEqual(response.data, payload)
        with self.client.get(prefix+'/scene.ply', headers={'Accept-Encoding':'gzip', 'Range':'bytes=0-2'}) as response:
            self.assertEqual(response.status_code, 206)
            self.assertNotIn('Content-Encoding', response.headers)
            self.assertEqual(response.data, b'ply')

    def test_depth_alignment_requires_consistent_multiview_tracks(self):
        sparse = self.root / 'sparse'
        sparse.mkdir()
        tracks = ' '.join(f'{i} 1 {i+1}' for i in range(30))
        (sparse / 'images.txt').write_text('1 1 0 0 0 0 0 0 1 first.jpg\n' + tracks + '\n')
        (sparse / 'points3D.txt').write_text(''.join(f'{i+1} 0 0 2 0 0 0 0\n' for i in range(30)))
        points = np.zeros((4, 32, 3)); points[:, :, 2] = 6
        geometry = self.root / 'depth.npz'
        np.savez(geometry, points=points, mask=np.ones((4, 32), bool))
        scale, report = depth_scale(self.root, geometry, {'name':'first.jpg','intrinsics':[32,4,20,20,16,2]})
        self.assertAlmostEqual(scale, 1/3)
        self.assertEqual(report['correspondences'], 30)
        np.savez(geometry, points=points, mask=np.zeros((4, 32), bool))
        with self.assertRaisesRegex(ValueError, 'Too few'):
            depth_scale(self.root, geometry, {'name':'first.jpg','intrinsics':[32,4,20,20,16,2]})

    def test_slow_fine_pass_continues_from_last_complete_checkpoint_once(self):
        target = self.root / 'attempt'
        (target / 'inputs').mkdir(parents=True)
        for index in range(3):
            (target / f'inputs/{index}.jpg').write_bytes(b'photo')
        values, cameras, header = fixture()
        calls = []
        def prepare(source, poses, **options):
            (poses / 'brush_dataset').mkdir(parents=True)
            return dict(registered_images=3, points3d=45)
        def native(dataset, output, **options):
            calls.append(options)
            exports = output / 'exports'
            exports.mkdir(parents=True)
            step = 1200 if len(calls) == 1 else (2000 if len(calls) == 2 else 2400)
            ply = exports / f'export_{step}.ply'
            write_ply(ply, header, values)
            if len(calls) == 2:
                # A newer but truncated export must not win checkpoint selection.
                broken = exports / 'export_2200.ply'
                broken.write_bytes(b'ply\nformat binary_little_endian 1.0\nend_header\n')
                return dict(status='training_incomplete', error='training exceeded 900s',
                    checkpoints=[dict(path=str(ply)), dict(path=str(broken))])
            return dict(status='training_finished')
        with patch('gaussian_job.reconstruct', side_effect=prepare), patch('gaussian_job.train', side_effect=native), \
             patch('gaussian_job.config', return_value=dict(cameras=cameras)):
            build_training(target, standalone=True)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[2]['start_step'], 2000)
        self.assertEqual(calls[2]['timeout'], 600)
        self.assertFalse(calls[2]['evaluate'])
        self.assertEqual(json.loads((target/'manifest.json').read_text())['status'], 'ready')


if __name__ == '__main__':
    unittest.main()
