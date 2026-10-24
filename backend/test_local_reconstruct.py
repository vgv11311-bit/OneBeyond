import tempfile
import unittest
import sys
import json
from pathlib import Path

import cv2
import numpy as np
from unittest.mock import patch

from local_reconstruct import prepare_images, read_model_counts, reconstruct, validate_registration, run_stage, select_video_frames


class LocalReconstructionTests(unittest.TestCase):
    def test_fixed_zoom_video_shares_intrinsics_without_faking_photo_exif(self):
        for video in [True,False]:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);source=root/'source'
                if video:source.write_bytes(b'video-fixture')
                else:source.mkdir()
                target=root/'prepared'
                def prepare(_,output,**kwargs):
                    (output/'inputs.json').write_text(json.dumps([dict(camera_signature=None)]*3));return 3
                commands=[]
                def stop_before_native(_,name,command,*args):
                    commands.append(command);raise RuntimeError('intentional-test-stop')
                with patch('local_reconstruct.prepare_images',side_effect=prepare),patch('local_reconstruct.run_stage',side_effect=stop_before_native),patch('local_reconstruct.tool',return_value='colmap'):
                    with self.assertRaisesRegex(RuntimeError,'intentional-test-stop'):reconstruct(source,target)
                command=commands[0];value=command[command.index('--ImageReader.single_camera')+1]
                self.assertEqual(value,'1' if video else '0')
                report=json.loads((target/'local_report.json').read_text())
                self.assertFalse(report['shared_intrinsics_from_exif'])
                self.assertEqual(report['shared_intrinsics_from_video'],video)
    def test_video_selection_reports_duplicates_and_low_detail_without_faking_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            files=[]
            for index in range(6):
                path=Path(directory)/f'{index}.png'
                image=np.full((64,80),100 if index<4 else 160,np.uint8)
                cv2.imwrite(str(path),image);files.append(path)
            files.append(Path(directory)/'unreadable.png')
            selected,report=select_video_frames(files,4,return_report=True)
            self.assertEqual(len(selected),2)
            self.assertEqual(report['candidate_count'],7)
            self.assertEqual(report['selected_count'],2)
            self.assertTrue(any(item['status']=='duplicate' for item in report['frames']))
            self.assertTrue(any(item['status']=='unreadable' for item in report['frames']))
            self.assertTrue(all(item.get('low_detail_warning',True) for item in report['frames']))
        with self.assertRaises(ValueError):select_video_frames([],0)
    def test_timeout_is_bounded_and_persisted_as_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                run_stage(Path(directory), 'timeout', [sys.executable, '-c', 'import time; time.sleep(30)'], .1)
            report = json.loads(Path(directory, 'stages.jsonl').read_text())
            self.assertTrue(report['timed_out'])
            self.assertLess(report['seconds'], 20)

    def test_colmap_two_line_records_with_empty_observations_and_spaces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images.txt').write_text('# header\n1 1 0 0 0 0 0 0 1 image one.jpg\n\n2 1 0 0 0 1 0 0 2 two.jpg\n1 2 3\n')
            (root / 'points3D.txt').write_text('# header\n1 0 0 0 255 0 0 0\n')
            counts = read_model_counts(root)
            self.assertEqual(counts['registered_images'], 2)
            self.assertEqual(counts['image_names'][0], 'image one.jpg')
            self.assertEqual(counts['points3d'], 1)

    def test_no_fake_pose_fallback_for_insufficient_registration(self):
        for registered, total, points in [(2, 5, 200), (3, 10, 200), (5, 5, 0)]:
            with self.assertRaises(RuntimeError):
                validate_registration({'registered_images': registered, 'points3d': points}, total)
        validate_registration({'registered_images': 3, 'points3d': 100}, 5)

    def test_existing_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                reconstruct(Path(directory), Path(directory))

    def test_duplicate_images_do_not_count_as_multiview_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            output = root / 'run'
            output.mkdir()
            for i in range(3):
                cv2.imwrite(str(source / f'{i}.png'), np.zeros((100, 100, 3), dtype=np.uint8))
            with self.assertRaises(ValueError):
                prepare_images(source, output)
            self.assertEqual(len(list((output / 'images').glob('*.jpg'))), 1)

    def test_video_frame_selection_keeps_sharp_views_across_the_route(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            rng = np.random.default_rng(4)
            for index in range(6):
                path = root / f'{index}.png'
                pixels = rng.integers(0, 255, (240, 320), dtype=np.uint8)
                if index % 2 == 0:
                    pixels = cv2.GaussianBlur(pixels, (21, 21), 5)
                cv2.imwrite(str(path), pixels)
                files.append(path)
            self.assertEqual(select_video_frames(files, 3), files[1::2])


if __name__ == '__main__':
    unittest.main()
