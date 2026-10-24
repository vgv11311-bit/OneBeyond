import json
import subprocess
import tempfile
import sys
import time
from contextlib import nullcontext
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import app as server


class SupervisionTests(unittest.TestCase):
    def test_competing_process_writes_cannot_regress_terminal_state(self):
        with tempfile.TemporaryDirectory() as folder,patch('app.SPACES_DIR',Path(folder)):
            state={'status':'processing','attempt_id':'a','attempt_count':1,'server_session':server.SERVER_SESSION}
            server.save_job_state('race',state)
            script="import sys,json; from pathlib import Path; import app; app.SPACES_DIR=Path(sys.argv[1]); s=json.loads(sys.argv[2]);\nfor i in range(20): app.save_job_state('race',{**s,'status':'processing','progress':i});\napp.save_job_state('race',{**s,'status':'completed','progress':100})"
            children=[subprocess.Popen([sys.executable,'-c',script,folder,json.dumps(state)],cwd=Path(server.__file__).parent) for _ in range(3)]
            try:
                for child in children:self.assertEqual(child.wait(timeout=60),0)
            finally:
                for child in children:
                    if child.poll() is None:child.kill()
                    child.wait(timeout=10)
            self.assertEqual(json.loads((Path(folder)/'race'/'metadata.json').read_text())['status'],'completed')
            self.assertFalse(list(Path(folder).rglob('*.tmp')))

    def test_cancel_wins_completion_race_and_excludes_recent(self):
        with tempfile.TemporaryDirectory() as folder,patch('app.SPACES_DIR',Path(folder)),patch.dict(server.JOBS,{},clear=True):
            state={'status':'completed','attempt_id':'a','attempt_count':1,'logs':[]}
            server.save_job_state('race',state)
            response=server.app.test_client().post('/api/reconstruct/cancel/race',json={'attempt_id':'a'})
            self.assertEqual(response.json['status'],'cancelled')
            server.save_job_state('race',state)
            self.assertEqual(server.load_job_state('race')['status'],'cancelled')
            self.assertEqual(server.app.test_client().get('/api/spaces').json['spaces'],[])

    def test_stale_worker_cannot_overwrite_newer_attempt(self):
        with tempfile.TemporaryDirectory() as folder,patch('app.SPACES_DIR',Path(folder)):
            server.save_job_state('retry',{'status':'processing','attempt_id':'new','attempt_count':2,'server_session':server.SERVER_SESSION})
            server.save_job_state('retry',{'status':'completed','attempt_id':'old','attempt_count':1})
            self.assertEqual(server.load_job_state('retry')['attempt_id'],'new')

    def test_cancel_terminates_real_isolated_process(self):
        with tempfile.TemporaryDirectory() as directory, patch('app.SPACES_DIR', Path(directory)), patch.dict(server.JOBS, {}, clear=True):
            server.save_job_state('live',{'status':'processing','attempt_id':'active','server_session':server.SERVER_SESSION,'logs':[]})
            child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'])
            try:
                with patch.dict(server.WORKERS,{'live':child},clear=True):
                    response=server.app.test_client().post('/api/reconstruct/cancel/live',json={'attempt_id':'active'})
                self.assertEqual(response.json['status'],'cancelled')
                self.assertIsNotNone(child.poll())
            finally:
                if child.poll() is None:
                    child.kill(); child.wait(timeout=5)

    def test_metadata_retries_windows_reader_lock(self):
        with tempfile.TemporaryDirectory() as directory, patch('app.SPACES_DIR', Path(directory)), patch.dict(server.JOBS, {}, clear=True):
            replace=Path.replace
            calls=[]
            def locked_once(path, target):
                calls.append(path)
                if len(calls)==1: raise PermissionError('Windows reader lock')
                return replace(path,target)
            with patch.object(Path,'replace',locked_once), patch('app.time.sleep'):
                server.save_job_state('done', {'status':'completed'})
            self.assertEqual(server.load_job_state('done')['status'],'completed')
            self.assertEqual(len(calls),2)
            self.assertFalse(list(Path(directory).rglob('*.tmp')))

    def test_cancel_prevents_late_child_result_and_stale_attempt_cancel(self):
        with tempfile.TemporaryDirectory() as directory, patch('app.SPACES_DIR', Path(directory)), patch.dict(server.JOBS, {}, clear=True):
            state={'status':'processing','attempt_id':'new','server_session':server.SERVER_SESSION,'logs':[]}
            server.save_job_state('live',state)
            client=server.app.test_client()
            self.assertEqual(client.post('/api/reconstruct/cancel/live',json={'attempt_id':'old'}).json['status'],'stale_cancellation_ignored')
            self.assertEqual(client.post('/api/reconstruct/cancel/live',json={'attempt_id':'new'}).json['status'],'cancelled')
            server.save_job_state('live',{**state,'status':'completed'})
            self.assertEqual(server.load_job_state('live')['status'],'cancelled')

    def test_completed_artifact_survives_nonzero_native_shutdown(self):
        with tempfile.TemporaryDirectory() as directory, patch('app.SPACES_DIR', Path(directory)), patch.dict(server.JOBS, {}, clear=True):
            server.save_job_state('done',{'status':'processing','logs':[],'server_session':server.SERVER_SESSION})
            child=MagicMock(returncode=1)
            def finish(timeout):
                server.save_job_state('done',{'status':'completed','progress':100,'logs':[]})
                (Path(directory)/'done'/'model.glb').write_bytes(b'glTF')
            child.wait.side_effect=finish
            with patch('app.subprocess.Popen',return_value=child):
                server.supervise_reconstruction('done','groq_scene',{})
            self.assertEqual(server.load_job_state('done')['status'],'completed')

    def test_restart_recovers_orphan_without_losing_input(self):
        with tempfile.TemporaryDirectory() as directory, patch('app.SPACES_DIR', Path(directory)), patch.dict(server.JOBS, {}, clear=True):
            server.save_job_state('orphan', {'status': 'processing', 'logs': [], 'file_paths': ['original.png'],
                                            'reconstruction_options': {'knownRoomWidthM': 5}, 'server_session': 'old'})
            server.JOBS.clear()
            recovered = server.load_job_state('orphan')
            self.assertEqual(recovered['status'], 'failed')
            self.assertEqual(recovered['file_paths'], ['original.png'])
            self.assertEqual(recovered['reconstruction_options']['knownRoomWidthM'], 5)

    def test_timeout_stops_child_and_releases_gate_for_next_job(self):
        with tempfile.TemporaryDirectory() as directory, patch('app.SPACES_DIR', Path(directory)), patch.dict(server.JOBS, {}, clear=True):
            for job in ('first', 'next'):
                server.save_job_state(job, {'status': 'processing', 'logs': [], 'server_session': server.SERVER_SESSION})
            child = MagicMock()
            child.poll.return_value = None
            with patch('app.process_lock',side_effect=lambda *a,**k:nullcontext()), patch('app.subprocess.Popen', return_value=child), patch('app.stop_worker') as stop, patch('app.time.monotonic', side_effect=[0,0,601]):
                server.supervise_reconstruction('first', 'gemini_scene', {})
                stop.assert_called_once_with(child)
            self.assertEqual(server.load_job_state('first')['status'], 'failed')
            self.assertIn('worker time limit', server.load_job_state('first')['error'])
            with patch('app.subprocess.Popen', side_effect=OSError('startup failed')):
                server.supervise_reconstruction('next', 'gemini_scene', {})
            self.assertEqual(server.load_job_state('next')['status'], 'failed')

    def test_status_reads_worker_updates_not_cached_queue_state(self):
        with tempfile.TemporaryDirectory() as directory, patch('app.SPACES_DIR', Path(directory)), patch.dict(server.JOBS, {}, clear=True):
            server.save_job_state('live', {'status': 'processing', 'logs': [], 'server_session': server.SERVER_SESSION})
            path = Path(directory) / 'live' / 'metadata.json'
            # Simulate an independent worker's atomic metadata replacement.
            path.write_text(json.dumps({'status': 'completed', 'progress': 100}), encoding='utf-8')
            result = server.app.test_client().get('/api/reconstruct/status/live')
            self.assertEqual(result.json['progress'], 100)


if __name__ == '__main__':
    unittest.main()
