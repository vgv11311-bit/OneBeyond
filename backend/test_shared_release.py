"""Offline adversarial release checks; never extract or execute packaged code."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

VERIFY = Path(__file__).resolve().parents[1] / 'scripts/verify-shared-release.py'


class SharedReleaseTest(unittest.TestCase):
    def verify(self, payloads, *, manifest=None, duplicate=False):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'candidate.tar.gz'
            if manifest is None:
                manifest = {'release': 'test-only', 'files': {
                    name: hashlib.sha256(data).hexdigest() for name, data in payloads}}
            with tarfile.open(archive, 'w:gz') as tar:
                entries = payloads + [('release.json', json.dumps(manifest).encode())]
                if duplicate:
                    entries.append(entries[0])
                for name, data in entries:
                    item = tarfile.TarInfo(name)
                    item.size = len(data)
                    tar.addfile(item, io.BytesIO(data))
            result = subprocess.run([sys.executable, str(VERIFY), str(archive)],
                                    capture_output=True, text=True, timeout=15)
            self.assertFalse((Path(directory) / 'backend').exists())
            return result

    def test_valid_release_is_inspected_not_executed(self):
        result = self.verify([('backend/example.py', b'raise RuntimeError("must not execute")\n')])
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report['extracted'])
        self.assertFalse(report['runtime_test'])

    def test_traversal_rejected(self):
        self.assertNotEqual(self.verify([('../outside.py', b'pass')]).returncode, 0)

    def test_absolute_path_rejected(self):
        self.assertNotEqual(self.verify([('/outside.py', b'pass')]).returncode, 0)

    def test_backslash_path_rejected(self):
        self.assertNotEqual(self.verify([('backend\\outside.py', b'pass')]).returncode, 0)

    def test_windows_drive_paths_rejected(self):
        for name in ['C:/outside.py', 'C:outside.py', 'backend/a.py:stream']:
            with self.subTest(name=name):
                self.assertNotEqual(self.verify([(name, b'pass')]).returncode, 0)

    def test_private_paths_rejected(self):
        for name in ['backend/.env', 'backend/storage/photo.jpg', 'models/model.bin',
                     'backend/credentials.json', '.git/config']:
            with self.subTest(name=name):
                self.assertNotEqual(self.verify([(name, b'private fixture')]).returncode, 0)

    def test_duplicate_member_rejected(self):
        self.assertNotEqual(self.verify([('backend/a.py', b'pass')], duplicate=True).returncode, 0)

    def test_checksum_mismatch_rejected(self):
        manifest = {'release': 'test', 'files': {'backend/a.py': '0' * 64}}
        self.assertNotEqual(self.verify([('backend/a.py', b'pass')], manifest=manifest).returncode, 0)

    def test_unlisted_member_rejected(self):
        self.assertNotEqual(self.verify([('backend/a.py', b'pass')],
                                       manifest={'release': 'test', 'files': {}}).returncode, 0)

    def test_invalid_python_rejected(self):
        self.assertNotEqual(self.verify([('backend/a.py', b'def broken(')]).returncode, 0)


if __name__ == '__main__':
    unittest.main()
