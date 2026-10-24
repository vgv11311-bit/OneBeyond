import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from reconstruction.base_engine import ReconstructionResult, ReconstructionMetrics
import app as server


class ApiSafetyTests(unittest.TestCase):
    def test_provider_outage_does_not_double_analysis_budget(self):
        with tempfile.TemporaryDirectory() as directory, patch('app.SPACES_DIR',Path(directory)), patch.dict(server.JOBS,{},clear=True):
            server.save_job_state('outage-job',{'status':'processing','logs':[],'input_type':'photos',
                'file_paths':['saved.jpg'],'reconstruction_options':{'knownRoomWidthM':5}})
            outage=ReconstructionResult(False,'outage-job','',ReconstructionMetrics(),
                error_message='HTTP 503',error_code='analysis_service_unavailable',retry_after_seconds=30)
            with patch('app.EngineRegistry.get_engine') as get_engine, patch('app.time.sleep') as sleep:
                get_engine.return_value.reconstruct.return_value=outage
                server.run_reconstruction_worker('outage-job','gemini_scene',{'knownRoomWidthM':5})
                self.assertEqual(get_engine.return_value.reconstruct.call_count,1)
                sleep.assert_not_called()
                self.assertEqual(get_engine.return_value.reconstruct.call_args.kwargs['options'],{'knownRoomWidthM':5})
            final=server.load_job_state('outage-job')
            self.assertEqual(final['status'],'failed')
            self.assertEqual(final['error_code'],'analysis_service_unavailable')
            self.assertIn('saved',final['error'])

    def test_invalid_photo_does_not_enter_provider_recovery(self):
        with tempfile.TemporaryDirectory() as directory, patch('app.SPACES_DIR',Path(directory)), patch.dict(server.JOBS,{},clear=True):
            server.save_job_state('bad-job',{'status':'processing','logs':[],'input_type':'photos','file_paths':['saved.jpg']})
            with patch('app.EngineRegistry.get_engine') as get_engine, patch('app.time.sleep') as sleep:
                get_engine.return_value.reconstruct.return_value=ReconstructionResult(False,'bad-job','',ReconstructionMetrics(),error_message='Invalid photo')
                server.run_reconstruction_worker('bad-job','gemini_scene',{})
                get_engine.return_value.reconstruct.assert_called_once(); sleep.assert_not_called()

    def test_retry_preserves_upload_and_measurements_and_clears_failure(self):
        with tempfile.TemporaryDirectory() as directory, patch('app.SPACES_DIR',Path(directory)), patch.dict(server.JOBS,{},clear=True):
            server.save_job_state('retry-job',{'status':'failed','logs':[],'input_type':'photos',
                'file_paths':['saved.jpg'], 'progress':45, 'error':'HTTP 503',
                'engine_id':'gemini_scene','reconstruction_options':{'knownRoomWidthM':5}})
            with patch('app.threading.Thread') as thread:
                result=server.app.test_client().post('/api/reconstruct',json={'job_id':'retry-job'})
                self.assertEqual(result.status_code,202)
                state=server.load_job_state('retry-job')
                self.assertEqual(state['file_paths'],['saved.jpg'])
                self.assertEqual(state['reconstruction_options'],{'knownRoomWidthM':5})
                self.assertEqual(state['progress'],0)
                self.assertIsNone(state['error'])
                self.assertEqual(state['attempt_count'],1)
                thread.return_value.start.assert_called_once()

    def test_invalid_inputs_do_not_launch_workers(self):
        client=server.app.test_client()
        with patch('app.threading.Thread') as thread:
            for body in [[1],{'job_id':'../outside'}, {'job_id':'abc','options':[]},
                         {'job_id':'abc','engine_id':'missing'}]:
                self.assertEqual(client.post('/api/reconstruct',json=body).status_code,400)
            thread.assert_not_called()
        self.assertIsNone(server.load_job_state('../outside'))

    def test_double_start_is_reserved_before_worker_runs(self):
        with tempfile.TemporaryDirectory() as directory, patch('app.SPACES_DIR',Path(directory)), patch.dict(server.JOBS,{},clear=True):
            server.save_job_state('test-job',{'status':'uploaded','logs':[],'input_type':'photos','file_paths':['a','b','c']})
            with patch('app.threading.Thread') as thread:
                client=server.app.test_client()
                self.assertEqual(client.post('/api/reconstruct',json={'job_id':'test-job'}).status_code,202)
                self.assertEqual(client.post('/api/reconstruct',json={'job_id':'test-job'}).status_code,200)
                thread.return_value.start.assert_called_once()

    def test_state_snapshots_are_independent_and_file_is_complete(self):
        with tempfile.TemporaryDirectory() as directory, patch('app.SPACES_DIR',Path(directory)), patch.dict(server.JOBS,{},clear=True):
            state={'status':'uploaded','logs':['original']}
            server.save_job_state('test-job',state)
            state['logs'].append('uncommitted')
            loaded=server.load_job_state('test-job'); loaded['logs'].append('also uncommitted')
            self.assertEqual(server.load_job_state('test-job')['logs'],['original'])
            self.assertTrue((Path(directory)/'test-job'/'metadata.json').is_file())
            self.assertFalse((Path(directory)/'test-job'/'metadata.json.tmp').exists())


if __name__=='__main__': unittest.main()
