import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import app as server
from reconstruction.resplat_policy import validate_video_duration,timing_state
from reconstruction.resplat_transport import direct_arguments


class BoundedResplatTests(unittest.TestCase):
    def test_video_envelope_accepts_90_but_not_100(self):
        for seconds in (1,62.65,90):validate_video_duration(seconds)
        for seconds in (0,90.01,100,180,math.nan,math.inf,True,None):
            with self.subTest(seconds=seconds),self.assertRaises(ValueError):validate_video_duration(seconds)

    def test_soft_target_is_not_a_completion_claim(self):
        timing=timing_state(100,281)
        self.assertTrue(timing['soft_target_exceeded'])
        self.assertEqual(timing['remaining_seconds'],119)
        self.assertFalse(timing['successful_room_guaranteed'])
        self.assertFalse(timing['forced_abort_at_target'])
        self.assertTrue(timing_state(100,400)['deadline_exceeded'])

    def test_overlong_saved_video_does_not_start_or_probe_gpu(self):
        with tempfile.TemporaryDirectory() as directory,patch('app.SPACES_DIR',Path(directory)),patch.dict(server.JOBS,{},clear=True):
            server.save_job_state('long-video',dict(job_id='long-video',status='uploaded',input_type='video',
                file_paths=['original.mov'],video_checks={'duration_seconds':100},logs=[]))
            with patch('reconstruction.resplat_engine.availability') as probe,patch('app.threading.Thread') as worker:
                result=server.app.test_client().post('/api/reconstruct',json={'job_id':'long-video','engine_id':'resplat_experimental'})
            self.assertEqual(result.status_code,400);probe.assert_not_called();worker.assert_not_called()
            self.assertEqual(server.load_job_state('long-video')['file_paths'],['original.mov'])

    def test_tunnel_commands_pin_host_key_and_use_only_loopback(self):
        cfg={'host_key':'SHA256:test','user':'tester'}
        paths=tuple(Path(p) for p in ('plink','pscp','private key.ppk'))
        for kind,a,b in [('ssh','echo ready',None),('upload',Path('image data.tgz'),'/safe/input.tgz'),('download','/safe/output.tgz',Path('local.tgz'))]:
            args=direct_arguments(cfg,paths,12345,kind,a,b)
            self.assertIn('-batch',args);self.assertIn('SHA256:test',args);self.assertIn('12345',args)
            self.assertTrue(any('tester@127.0.0.1' in s for s in args))
            self.assertNotIn('-no-antispoof',args)
