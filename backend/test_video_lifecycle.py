from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import app as server

class VideoLifecycleTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        override=patch('app.SPACES_DIR',Path(temp.name));override.start();self.addCleanup(override.stop)
        cache=patch.dict(server.JOBS,{},clear=True);cache.start();self.addCleanup(cache.stop)
        self.client=server.app.test_client()

    def state(self,**values):
        state=dict(job_id='video',input_type='video',file_paths=['original-walk.mp4'],status='failed',logs=[],engine_id='photographic',reconstruction_options={})
        state.update(values);server.save_job_state('video',state)

    def test_retry_retains_original_video_and_only_reserves_one_worker(self):
        self.state()
        with patch('app.threading.Thread') as worker:
            response=self.client.post('/api/reconstruct',json=dict(job_id='video'))
            self.assertEqual(response.status_code,202)
            self.assertEqual(self.client.post('/api/reconstruct',json=dict(job_id='video')).status_code,200)
            worker.return_value.start.assert_called_once()
        state=server.load_job_state('video');self.assertEqual(state['input_type'],'video')
        self.assertEqual(state['file_paths'],['original-walk.mp4'])

    def test_video_coverage_failure_has_no_semantic_room_or_provider_fallback(self):
        self.state(status='processing',server_session=server.SERVER_SESSION)
        with patch('reconstruction.photographic_engine.build_companion',return_value=dict(status='unavailable',reason='Camera coverage was insufficient: registered 3/10 selected views.')) as build,patch('reconstruction.groq_scene_engine.GroqSceneEngine.reconstruct',side_effect=AssertionError('Unexpected provider')):
            server.run_reconstruction_worker('video','photographic',{})
        state=server.load_job_state('video');self.assertEqual(state['status'],'failed')
        self.assertNotIn('glb_url',state);self.assertNotIn('gaussian',state)
        self.assertEqual(build.call_args.kwargs['input_type'],'video')
        self.assertEqual(self.client.get('/api/spaces').json['spaces'],[])

    def test_cancelled_video_cannot_publish_a_late_room(self):
        self.state(status='processing',server_session=server.SERVER_SESSION,attempt_id='attempt',attempt_count=1)
        self.assertEqual(self.client.post('/api/reconstruct/cancel/video',json=dict(attempt_id='attempt')).json['status'],'cancelled')
        server.save_job_state('video',dict(server.load_job_state('video'),status='completed',gaussian=dict(status='ready')))
        state=server.load_job_state('video');self.assertEqual(state['status'],'cancelled')
        self.assertEqual(state['file_paths'],['original-walk.mp4'])
        self.assertEqual(self.client.get('/api/spaces').json['spaces'],[])

if __name__=='__main__':unittest.main()
