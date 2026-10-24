import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

import app as server
from reconstruction.base_engine import ReconstructionMetrics, ReconstructionResult
from reconstruction.gaussian_pipeline import build_companion, requested


class GaussianIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        override = patch.object(server, 'SPACES_DIR', self.root)
        override.start()
        self.addCleanup(override.stop)
        server.JOBS.clear()
        self.client = server.app.test_client()

    def state(self, **values):
        state = dict(job_id='test-room', status='uploaded', logs=[], file_paths=['a','b','c'], input_type='photos')
        state.update(values)
        server.save_job_state('test-room', state)
        return state

    def result(self):
        target = self.root / 'test-room/model.glb'
        target.write_bytes(b'glb-fixture')
        return ReconstructionResult(True, 'test-room', str(target), ReconstructionMetrics())

    def test_groq_defaults_on_other_engines_and_explicit_optout_unchanged(self):
        self.assertTrue(requested('groq_scene', {}))
        self.assertFalse(requested('groq_scene', {'photographic': False}))
        self.assertFalse(requested('gemini_scene', {}))

    def test_single_photo_skips_without_launching_tools(self):
        self.assertEqual(build_companion('test-room', ['a'], self.root, lambda *args: None)['status'], 'skipped')

    def test_missing_tools_is_not_job_failure(self):
        with patch('local_reconstruct.tool', side_effect=RuntimeError('missing')):
            self.assertEqual(build_companion('test-room', ['a','b','c'], self.root, lambda *args: None)['status'], 'unavailable')

    def test_companion_publishes_only_after_complete_child_manifest(self):
        sources = []
        for index in range(3):
            source = self.root / f'{index}.jpg'
            source.write_bytes(b'private image fixture')
            sources.append(str(source))
        vendor = self.root / 'vendor'
        vendor.mkdir()
        (vendor / 'spark.module.js').touch()
        def child(root, name, command, timeout):
            self.assertEqual(timeout, 1200)
            self.assertEqual(len(list((root / 'inputs').iterdir())), 3)
            (root / 'manifest.json').write_text(json.dumps({'status':'ready','vertices':50000}))
            (root / 'scene.ply').write_bytes(b'ply fixture')
            (root / 'config.json').write_text('{"cameras":[]}')
        with patch('local_reconstruct.tool', return_value='tool'), patch('local_reconstruct.run_stage', side_effect=child), patch('reconstruction.gaussian_pipeline.renderer_root', return_value=vendor):
            result = build_companion('test-room', sources, self.root / 'output', lambda *args: None)
        self.assertEqual(result['status'], 'ready')
        self.assertIn('/api/spaces/test-room/gaussian/', result['viewer_url'])
        self.assertNotIn(str(self.root), json.dumps(result))

    def test_cancelled_attempt_cannot_publish_gaussian(self):
        self.state(status='processing', attempt_id='attempt', server_session=server.SERVER_SESSION)
        def companion(*args):
            server.cancellation_path('test-room','attempt').touch()
            return {'status':'ready'}
        engine = SimpleNamespace(reconstruct=lambda **kwargs: self.result())
        with patch.object(server.EngineRegistry, 'get_engine', return_value=engine), patch.object(server, 'build_companion', side_effect=companion):
            server.run_reconstruction_worker('test-room', 'groq_scene', {})
        self.assertEqual(server.load_job_state('test-room')['status'], 'cancelled')

    def test_solid_completion_is_published_with_raw_export_preserved(self):
        sources=[]
        for index in range(3):
            path=self.root/f'photo{index}.jpg';path.write_bytes(b'photo');sources.append(str(path))
        vendor=self.root/'vendor';vendor.mkdir();(vendor/'spark.module.js').touch()
        def child(root,*args):
            (root/'scene.ply').write_bytes(b'original gaussian')
            (root/'config.json').write_text('{}')
            (root/'manifest.json').write_text('{"status":"ready","vertices":100}')
        def completion(source,target,*args):
            target.mkdir();(target/'scene.ply').write_bytes(b'cleaned gaussian')
            (target/'solid.glb').write_bytes(b'solid underlay')
            (target/'config.json').write_text('{"solid":{"enabled":true}}')
            (target/'manifest.json').write_text('{"status":"ready","vertices":80}')
        result=SimpleNamespace(metrics=ReconstructionMetrics(),preview_image_path='primary.jpg')
        with patch('local_reconstruct.tool',return_value='tool'),patch('local_reconstruct.run_stage',side_effect=child),patch('reconstruction.gaussian_pipeline.renderer_root',return_value=vendor),patch('reconstruction.gaussian_geometry.enhance',side_effect=completion):
            descriptor=build_companion('test-room',sources,self.root/'output',lambda *args:None,result)
        self.assertEqual(descriptor['vertices'],80)
        self.assertTrue(descriptor['solid_completion'])
        raw=list((self.root/'output').rglob('raw-scene.ply'))
        self.assertEqual(len(raw),1)
        self.assertEqual(raw[0].read_bytes(),b'original gaussian')

    def test_worker_publishes_successful_companion_and_preserves_glb(self):
        self.state()
        engine = SimpleNamespace(reconstruct=lambda **kwargs: self.result())
        with patch.object(server.EngineRegistry, 'get_engine', return_value=engine), patch.object(server, 'build_companion', return_value={'status':'ready','viewer_url':'fixture'}) as build:
            server.run_reconstruction_worker('test-room', 'groq_scene', {})
        final = server.load_job_state('test-room')
        self.assertEqual(final['status'], 'completed')
        self.assertEqual(final['gaussian']['status'], 'ready')
        self.assertIn('/model.glb', final['glb_url'])
        build.assert_called_once()

    def test_optout_clears_stale_gaussian_on_retry(self):
        self.state(gaussian={'status':'ready'})
        engine = SimpleNamespace(reconstruct=lambda **kwargs: self.result())
        with patch.object(server.EngineRegistry, 'get_engine', return_value=engine), patch.object(server, 'build_companion') as build:
            server.run_reconstruction_worker('test-room', 'groq_scene', {'photographic':False})
        self.assertNotIn('gaussian', server.load_job_state('test-room'))
        build.assert_not_called()

    def test_unavailable_companion_still_completes(self):
        self.state()
        engine = SimpleNamespace(reconstruct=lambda **kwargs: self.result())
        with patch.object(server.EngineRegistry, 'get_engine', return_value=engine), patch.object(server, 'build_companion', return_value={'status':'unavailable'}):
            server.run_reconstruction_worker('test-room', 'groq_scene', {})
        self.assertEqual(server.load_job_state('test-room')['status'], 'completed')

    def test_routes_only_expose_current_complete_attempt_and_fixed_artifacts(self):
        revision = 'a' * 32
        prefix = f'/api/spaces/test-room/gaussian/{revision}'
        self.state(status='completed', gaussian={'status':'ready','ply_url':prefix+'/scene.ply'})
        target = self.root / 'test-room/gaussian' / revision
        target.mkdir(parents=True)
        (target / 'scene.ply').write_bytes(b'ply fixture')
        (target / 'surface.glb').write_bytes(b'calibrated visible surface')
        (target / 'surface-03.glb').write_bytes(b'calibrated additional view')
        with self.client.get(prefix+'/scene.ply') as response:
            self.assertEqual(response.status_code, 200)
        html = self.client.get(prefix+'/viewer').text
        self.assertIn(prefix+'/config.json', html)
        self.assertIn('/api/gaussian-vendor/three/', html)
        with self.client.get(prefix+'/surface.glb') as response:
            self.assertEqual(response.data,b'calibrated visible surface')
        with self.client.get(prefix+'/surface-03.glb') as response:
            self.assertEqual(response.data,b'calibrated additional view')
        with self.client.get(prefix+'/navigation.js') as response:
            self.assertEqual(response.status_code,200)
            self.assertIn(b'createNavigation',response.data)
        for path in [prefix+'/inputs.json',prefix+'/depth-00.glb',prefix+'/surface-003.glb',prefix.replace(revision,'b'*32)+'/scene.ply']:
            self.assertEqual(self.client.get(path).status_code, 404)
        self.state(status='processing', server_session=server.SERVER_SESSION, gaussian={'status':'ready','ply_url':prefix+'/scene.ply'})
        self.assertEqual(self.client.get(prefix+'/scene.ply').status_code, 404)

    def test_vendor_traversal_and_unknown_library(self):
        for path in ['/api/gaussian-vendor/other/file.js','/api/gaussian-vendor/three/../../package.json','/api/gaussian-vendor/three/package.json']:
            self.assertEqual(self.client.get(path).status_code, 404)

    def test_gallery_retains_gaussian_metadata(self):
        self.state(status='completed', gaussian={'status':'ready'})
        self.assertEqual(self.client.get('/api/spaces').json['spaces'][0]['gaussian']['status'], 'ready')

    def test_invalid_boolean_rejected(self):
        self.state()
        response = self.client.post('/api/reconstruct', json={'job_id':'test-room','engine_id':'groq_scene','options':{'photographic':'yes'}})
        self.assertEqual(response.status_code,400)

    def test_numpy_camera_quaternion_matches_scipy_reference(self):
        from scipy.spatial.transform import Rotation
        from local_comparison_camera import image_records
        quaternion = np.array([.5,.1,.3,.7])
        path = self.root / 'images.txt'
        path.write_text('1 '+ ' '.join(map(str, quaternion))+' 1 2 3 1 camera.jpg\n0 0 -1\n')
        actual = image_records(path)['camera.jpg']
        expected = Rotation.from_quat(quaternion[[1,2,3,0]]).as_matrix()
        np.testing.assert_allclose(actual['rotation'], expected, atol=1e-12)
        np.testing.assert_allclose(actual['center'], -expected.T @ np.array([1,2,3]), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
