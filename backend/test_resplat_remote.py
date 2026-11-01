"""Synthetic dispatch safety checks; these do not certify hall quality."""
import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
import io
import tarfile
import subprocess
import hashlib
import os
from unittest.mock import patch

from reconstruction.resplat_remote import load_lease, dispatch, extract_result, unavailable_session_reason, connect_running_worker,sync_worker_code,run_cli


class RemoteResplatTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cli = self.root/'gcloud.cmd'
        self.cli.touch()
        self.path = self.root/'lease.json'
        self.lease = dict(project='test-project', zone='asia-south1-a', vm='test-vm', user='test',
                          root='/home/test/resplat', python='/home/test/resplat/env/bin/python',
                          worker='/home/test/resplat/resplat-infer-job.py',
                          host_key='SHA256:'+'A'*43, gcloud=str(self.cli), runtime_verified=True,
                          expires_at=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=20)).isoformat())

    def write(self):
        self.path.write_text(json.dumps(self.lease))

    def test_lease_requires_live_explicit_approval_and_runtime(self):
        self.write()
        self.assertEqual(load_lease(self.path)['vm'], 'test-vm')
        self.lease['runtime_verified'] = False
        self.write()
        self.assertIsNone(load_lease(self.path))
        self.lease['runtime_verified'] = True
        self.lease['expires_at'] = '2020-01-01T00:00:00Z'
        self.write()
        self.assertIsNone(load_lease(self.path))

    def test_rejects_shell_injection_and_path_escape(self):
        for field, value in [('vm', 'vm;start-other'), ('root', '/home/test/../other'),
                             ('worker', '/tmp/script.py;touch-bad'), ('host_key', 'anything')]:
            with self.subTest(field=field):
                original = self.lease[field]
                self.lease[field] = value
                self.write()
                self.assertIsNone(load_lease(self.path))
                self.lease[field] = original

    def test_expired_and_unverified_connections_have_specific_reasons(self):
        self.lease['expires_at']='2020-01-01T00:00:00Z'
        self.write()
        self.assertEqual(unavailable_session_reason(self.path),'expired')
        self.lease['expires_at']=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=20)).isoformat()
        self.lease['runtime_verified']=False
        self.write()
        self.assertEqual(unavailable_session_reason(self.path),'unverified')

    def manual_connection(self):
        self.lease.update(connection_mode='manual_vm_start',max_session_minutes=45,
                          expires_at='2020-01-01T00:00:00Z',runtime_verified=False)
        self.write()

    def test_worker_sync_only_updates_known_scripts_on_configured_vm(self):
        scripts=Path(__file__).resolve().parents[1]/'scripts'
        files={self.lease['worker']:scripts/'resplat-infer-job.py',
               self.lease['root']+'/resplat-video-low-memory-oct26.py':scripts/'resplat-low-memory.py',
               self.lease['root']+'/resplat_context_selection.py':scripts/'resplat_context_selection.py',
               self.lease['root']+'/resplat_contract.py':Path(__file__).resolve().parent/'reconstruction/resplat_contract.py'}
        expected={remote:hashlib.sha256(local.read_bytes()).hexdigest() for remote,local in files.items()}
        replies=[subprocess.CompletedProcess([],0,'{}','')]+[subprocess.CompletedProcess([],0,'','') for _ in range(4)]+[subprocess.CompletedProcess([],0,json.dumps(expected),'')]
        with patch('reconstruction.resplat_remote.run_cli',side_effect=replies) as commands:
            self.assertEqual(sync_worker_code(self.lease),expected)
        self.assertEqual(sum(call.args[0][2]=='scp' for call in commands.call_args_list),4)
        self.assertFalse(any('start' in call.args[0] or 'create' in call.args[0] for call in commands.call_args_list))
        def direct(kind,source,destination=None):
            return ['shared-'+kind,str(source)]+([destination] if destination else [])
        with patch('reconstruction.resplat_remote.run_cli',side_effect=list(replies)) as commands:
            self.assertEqual(sync_worker_code(self.lease,transport=direct),expected)
        self.assertEqual(sum(call.args[0][0]=='shared-upload' for call in commands.call_args_list),4)
        self.assertTrue(all(call.args[0][0].startswith('shared-') for call in commands.call_args_list))

    def test_manual_restart_reconnects_only_existing_running_vm(self):
        self.manual_connection()
        boot=dt.datetime.now(dt.timezone.utc)-dt.timedelta(minutes=3)
        replies=[subprocess.CompletedProcess([],0,json.dumps(dict(status='RUNNING',lastStartTimestamp=boot.isoformat())),''),
                 subprocess.CompletedProcess([],0,'ONE_BEYOND_CUDA_READY\n','')]
        with patch('reconstruction.resplat_remote.run_cli',side_effect=replies) as commands:
            result=connect_running_worker(self.path)
        self.assertTrue(result['can_run'])
        saved=load_lease(self.path)
        self.assertEqual(dt.datetime.fromisoformat(saved['expires_at']),boot+dt.timedelta(minutes=45))
        self.assertEqual(commands.call_args_list[0].args[0][1:4],['compute','instances','describe'])
        self.assertEqual(commands.call_args_list[1].args[0][1:3],['compute','ssh'])
        self.assertFalse(any('start' in call.args[0] or 'create' in call.args[0] for call in commands.call_args_list))

    def test_running_but_unreachable_cuda_is_not_reported_stopped(self):
        self.manual_connection();boot=dt.datetime.now(dt.timezone.utc)-dt.timedelta(minutes=3)
        replies=[subprocess.CompletedProcess([],0,json.dumps(dict(status='RUNNING',lastStartTimestamp=boot.isoformat())),''),
                 subprocess.CompletedProcess([],1,'','SSH not ready')]
        with patch('reconstruction.resplat_remote.run_cli',side_effect=replies):result=connect_running_worker(self.path)
        self.assertEqual(result['availability_code'],'resplat_remote_booting')
        self.assertEqual(unavailable_session_reason(self.path),'booting')
        self.assertEqual(json.loads(self.path.read_text())['last_vm_status'],'RUNNING')

    def test_reconnect_never_waits_for_worker_code_deployment(self):
        self.manual_connection();self.lease['sync_worker_code']=True;self.write()
        boot=dt.datetime.now(dt.timezone.utc)-dt.timedelta(minutes=3)
        replies=[subprocess.CompletedProcess([],0,json.dumps(dict(status='RUNNING',lastStartTimestamp=boot.isoformat())),''),
                 subprocess.CompletedProcess([],0,'ONE_BEYOND_CUDA_READY\n','')]
        with patch('reconstruction.resplat_remote.run_cli',side_effect=replies), \
             patch('reconstruction.resplat_remote.sync_worker_code') as sync:
            result=connect_running_worker(self.path)
        self.assertTrue(result['can_run']);sync.assert_not_called()

    @unittest.skipUnless(os.name=='nt','Windows process-tree regression')
    def test_cli_timeout_kills_descendants_without_waiting_on_inherited_pipes(self):
        with patch('reconstruction.resplat_remote.subprocess.Popen') as launch, \
             patch('reconstruction.resplat_remote.subprocess.run') as terminate:
            child=launch.return_value;child.pid=43210
            child.wait.side_effect=[subprocess.TimeoutExpired(['gcloud'],.1),0]
            with self.assertRaises(subprocess.TimeoutExpired):run_cli(['gcloud'],timeout=.1)
        self.assertIn('/T',terminate.call_args.args[0]);self.assertIn('/F',terminate.call_args.args[0])
        self.assertNotEqual(launch.call_args.kwargs['stdout'],subprocess.PIPE)
        child.communicate.assert_not_called()

    def test_stopped_vm_disables_old_lease_without_starting_it(self):
        self.manual_connection()
        with patch('reconstruction.resplat_remote.run_cli',return_value=subprocess.CompletedProcess([],0,'{"status":"TERMINATED"}','')) as commands:
            result=connect_running_worker(self.path)
        self.assertFalse(result['can_run']);self.assertEqual(result['availability_code'],'resplat_remote_offline')
        self.assertEqual(commands.call_count,1)
        self.assertIsNone(load_lease(self.path))
        self.assertEqual(unavailable_session_reason(self.path),'offline')

    def test_rechecking_cannot_extend_same_boot_budget(self):
        self.manual_connection()
        boot=dt.datetime.now(dt.timezone.utc)-dt.timedelta(minutes=46)
        with patch('reconstruction.resplat_remote.run_cli',return_value=subprocess.CompletedProcess([],0,
                   json.dumps(dict(status='RUNNING',lastStartTimestamp=boot.isoformat())),'')) as commands:
            result=connect_running_worker(self.path)
        self.assertEqual(result['availability_code'],'resplat_remote_session_limit')
        self.assertEqual(commands.call_count,1)
        self.assertIsNone(load_lease(self.path))

    def test_verified_lease_never_outlives_cloud_automatic_stop(self):
        self.manual_connection()
        boot=dt.datetime.now(dt.timezone.utc)
        state=dict(status='RUNNING',lastStartTimestamp=boot.isoformat(),scheduling={'maxRunDuration':{'seconds':'1800'}})
        replies=[subprocess.CompletedProcess([],0,json.dumps(state),''),
                 subprocess.CompletedProcess([],0,'ONE_BEYOND_CUDA_READY\n','')]
        with patch('reconstruction.resplat_remote.run_cli',side_effect=replies):result=connect_running_worker(self.path)
        self.assertTrue(result['can_run'])
        self.assertEqual(dt.datetime.fromisoformat(result['expires_at']),boot+dt.timedelta(minutes=30))

    def test_regular_private_eval_lease_cannot_be_renewed_automatically(self):
        self.write()
        with patch('reconstruction.resplat_remote.run_cli') as commands:
            result=connect_running_worker(self.path)
        self.assertFalse(result['can_run']);commands.assert_not_called()

    def test_expired_session_launches_no_process(self):
        with patch('reconstruction.resplat_remote.load_lease', return_value=None), \
             patch('reconstruction.resplat_remote.subprocess.Popen') as child:
            with self.assertRaisesRegex(RuntimeError, 'expired'):
                dispatch(self.root, self.root)
        child.assert_not_called()

    def test_discovery_only_reads_private_lease_without_cloud_calls(self):
        self.write()
        with patch.dict('os.environ', {'ONE_BEYOND_RESPLAT_REMOTE_LEASE': str(self.path)}), \
             patch('reconstruction.resplat_remote.subprocess.Popen', side_effect=AssertionError('No cloud discovery call')):
            self.assertIsNotNone(load_lease())

    def archive(self,names):
        path=self.root/'result.tgz'
        with tarfile.open(path,'w:gz') as archive:
            for name in names:
                info=tarfile.TarInfo(name);info.size=3;archive.addfile(info,io.BytesIO(b'abc'))
        return path

    def test_exact_native_stage_evidence_is_retained(self):
        archive=self.archive(['gaussians.ply','report.json','camera-pose-manifest.json','input/00000.png','rendered/00000.png'])
        extract_result(archive,self.root/'result')
        self.assertEqual((self.root/'result/rendered/00000.png').read_bytes(),b'abc')

    def test_rejects_traversal_duplicates_and_unexpected_stage_files(self):
        for bad in ['../escape','input/../../escape','rendered/00000.exe','input/0.png','gaussians.ply']:
            with self.subTest(bad=bad):
                archive=self.archive(['gaussians.ply','report.json',bad])
                with self.assertRaises(ValueError):extract_result(archive,self.root/'rejected')
                self.assertFalse((self.root/'rejected').exists())


if __name__ == '__main__':
    unittest.main()
