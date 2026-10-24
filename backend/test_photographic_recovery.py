import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reconstruction.gaussian_pipeline import reuse_registered_poses
from reconstruction.gaussian_geometry import write_ply
from test_gaussian_geometry import fixture
from gaussian_job import build


class PhotographicRecoveryTests(unittest.TestCase):
    def test_retry_reuses_only_byte_identical_inputs_and_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old=root/'gaussian/old';new=root/'gaussian/new'
            for target in [old,new]:
                (target/'inputs').mkdir(parents=True)
                (target/'inputs/000.jpg').write_bytes(b'original')
            (old/'poses/brush_dataset/sparse').mkdir(parents=True)
            (old/'poses/brush_dataset/images').mkdir()
            (old/'poses/brush_dataset/sparse/images.txt').write_text('camera evidence')
            (old/'poses/local_report.json').write_text(json.dumps(dict(status='poses_ready',profile='quality')))
            exports=old/'preview/exports';exports.mkdir(parents=True)
            values,_,header=fixture();write_ply(exports/'export_1000.ply',header,values)
            (exports/'export_1100.ply').write_bytes(b'truncated')
            self.assertTrue(reuse_registered_poses(root,new))
            self.assertEqual(json.loads((new/'resume.json').read_text()),dict(step=1000,phase='preview'))
            self.assertEqual((new/'resume.ply').read_bytes(),(exports/'export_1000.ply').read_bytes())
            other=root/'gaussian/other';(other/'inputs').mkdir(parents=True)
            (other/'inputs/000.jpg').write_bytes(b'changed')
            self.assertFalse(reuse_registered_poses(root,other))
            self.assertFalse((other/'poses').exists())
            completed=root/'gaussian/completed-preview';(completed/'inputs').mkdir(parents=True)
            (completed/'inputs/000.jpg').write_bytes(b'original')
            write_ply(exports/'export_1200.ply',header,values)
            self.assertTrue(reuse_registered_poses(root,completed))
            self.assertEqual(json.loads((completed/'resume.json').read_text()),dict(step=1200,phase='fine'))

    def test_preview_timeout_continues_once_before_fine_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory);(target/'inputs').mkdir()
            for i in range(3):(target/f'inputs/{i}.jpg').write_bytes(b'photo')
            ply=target/'checkpoint/export_1000.ply';ply.parent.mkdir()
            values,_,header=fixture();write_ply(ply,header,values)
            reports=[dict(status='training_incomplete',error='training exceeded 600s',checkpoints=[dict(path=str(ply))]),
                     dict(status='training_finished'),dict(status='training_finished')]
            with patch('gaussian_job.reconstruct',return_value={}),patch('gaussian_job.train',side_effect=reports) as train,patch('gaussian_job.export_capture') as publish:
                build(target,standalone=True)
            self.assertEqual(train.call_count,3)
            self.assertEqual(train.call_args_list[1].kwargs['start_step'],1000)
            self.assertEqual(train.call_args_list[1].kwargs['warm_start'],ply)
            self.assertEqual(train.call_args_list[2].kwargs['warm_start'],target/'preview-recovery/exports/export_1200.ply')
            publish.assert_called_once()

if __name__=='__main__':unittest.main()
