"""Synthetic dispatch safety checks; these do not certify hall quality."""
import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
import io
import tarfile
from unittest.mock import patch

from reconstruction.resplat_remote import load_lease, dispatch, extract_result


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
