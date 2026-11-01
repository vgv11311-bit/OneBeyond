import io
import tarfile
import tempfile
from pathlib import Path
import unittest

from reconstruction.resplat_camera_remote import unpack_cameras,prepared_identity


class CameraTransportTests(unittest.TestCase):
    def archive(self,root,names):
        file=root/'result.tgz'
        with tarfile.open(file,'w:gz') as bundle:
            for name in names:
                member=tarfile.TarInfo(name);member.size=1;bundle.addfile(member,io.BytesIO(b'x'))
        return file

    def test_only_exact_camera_evidence_can_be_extracted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);allowed=['camera-report.json','sparse/cameras.txt','sparse/images.txt','sparse/points3D.txt']
            archive=self.archive(root,allowed);unpack_cameras(archive,root/'valid')
            self.assertTrue((root/'valid/sparse/images.txt').is_file())
            for bad in ['../outside.txt','scene.ply','camera-report.json']:
                archive=self.archive(root,[*allowed,bad])
                with self.assertRaises(ValueError):unpack_cameras(archive,root/'invalid')
            self.assertFalse((root/'invalid').exists())

    def test_handoff_cannot_reuse_another_session_or_path(self):
        lease=dict(root='/safe',vm_boot_id='boot',expires_at='expiry')
        prepared=dict(remote='/safe/jobs/'+'a'*32,vm_boot_id='boot',expires_at='expiry')
        self.assertTrue(prepared_identity(prepared,lease))
        for change in [dict(vm_boot_id='old'),dict(expires_at='old'),dict(remote='/safe/jobs/../other')]:
            self.assertFalse(prepared_identity({**prepared,**change},lease))
