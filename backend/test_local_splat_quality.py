from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from local_reconstruct import prepare_images, read_photo
from local_splat_training import vertex_count, train, native_timing_failure
from review_local_reconstruction import review


class SplatQualityTests(unittest.TestCase):
    def test_only_observed_native_timing_signatures_are_retriable(self):
        self.assertTrue(native_timing_failure('overflow in iter::sum over durations\nOrdering is bigger than operations'))
        self.assertTrue(native_timing_failure('cannot convert float seconds to Duration'))
        self.assertFalse(native_timing_failure('out of memory'))
        self.assertFalse(native_timing_failure('Ordering is bigger than operations'))
    def test_preserve_optical_metadata_without_location_or_unique_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'photo.jpg'
            exif = Image.Exif()
            exif[271], exif[272] = 'Maker', 'Device'
            exif[34853] = {1: 'N', 2: (1., 2., 3.)}
            exif[34665] = {37386: 4., 41989: 26, 42036: 'Wide', 42033: 'private-serial'}
            Image.new('RGB', (100, 80), 'gray').save(source, exif=exif)
            pixels, encoded, signature = read_photo(source)
            recovered = Image.Exif()
            recovered.load(encoded)
            self.assertEqual(pixels.shape[:2], (80, 100))
            self.assertNotIn(34853, recovered)
            self.assertNotIn(42033, recovered.get_ifd(34665))
            self.assertEqual(recovered.get_ifd(34665)[41989], 26)
            self.assertIsNotNone(signature)

    def test_missing_metadata_does_not_infer_shared_camera_from_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'photo.jpg'
            Image.new('RGB', (100, 80)).save(source)
            self.assertIsNone(read_photo(source)[2])

    def test_digital_zoom_separates_shared_intrinsics_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            signatures = []
            for zoom in [1, 2]:
                path = Path(directory) / f'{zoom}.jpg'
                exif = Image.Exif()
                exif[271], exif[272] = 'Maker', 'Device'
                exif[34665] = {37386: 4., 41989: 26, 42036: 'Wide', 41988: zoom}
                Image.new('RGB', (100, 80)).save(path, exif=exif)
                signatures.append(read_photo(path)[2])
            self.assertNotEqual(*signatures)

    def test_orientation_is_applied_before_feature_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'photo.jpg'
            exif = Image.Exif()
            exif[274] = 6
            Image.new('RGB', (100, 80)).save(source, exif=exif)
            self.assertEqual(read_photo(source)[0].shape[:2], (100, 80))

    def test_empty_exif_is_not_reported_as_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'photo.png'
            Image.new('RGB', (100, 80)).save(source)
            self.assertEqual(read_photo(source)[1], b'')

    def test_checkpoint_count_is_not_held_out_view_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images').mkdir()
            Image.new('RGB', (100, 80), 'gray').save(root / 'images/a.png')
            for step in [100, 200]:
                out = root / f'exports/eval_{step}'
                out.mkdir(parents=True)
                Image.new('RGB', (100, 80), 'gray').save(out / 'a.png')
            result = review(root, root / 'exports')
            self.assertEqual(result['held_out_views'], 1)
            self.assertEqual(result['evaluation_renders'], 2)

    def test_ply_header_count_and_cap_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ply = root / 'checkpoint.ply'
            ply.write_bytes(b'ply\nformat binary_little_endian 1.0\nelement vertex 30000\nend_header\n')
            self.assertEqual(vertex_count(ply), 30000)
            with self.assertRaises(ValueError):
                train(root, root, warm_start=ply)

    def test_known_native_timing_panic_retries_once_without_changing_view_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / 'source'
            (dataset / 'images').mkdir(parents=True)
            (dataset / 'sparse').mkdir()
            (dataset / 'images/a.jpg').write_bytes(b'fixture')
            (dataset / 'sparse/images.txt').write_text(''.join(f'{i} 1 0 0 0 0 0 0 1 {i}.jpg\n\n' for i in range(1,4)))
            (dataset / 'sparse/points3D.txt').write_text('1 0 0 2 0 0 0 0\n')
            calls = []
            def native(output, stage, command, timeout):
                calls.append((stage, command, timeout))
                if len(calls) == 1:
                    (output / 'training.log').write_text('overflow in iter::sum over durations')
                    raise RuntimeError('native timing failure')
                (output / 'exports/export_1200.ply').write_bytes(b'ply\nelement vertex 30\nend_header\n')
            with patch('local_splat_training.tool', return_value='brush'), patch('local_splat_training.run_stage', side_effect=native):
                report = train(dataset, root / 'trained', evaluate=False)
            self.assertEqual(report['status'], 'training_finished')
            self.assertEqual(report['native_runtime_retry'], 1)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][1], calls[1][1])
            self.assertNotIn('--eval-split-every', calls[1][1])
            self.assertLessEqual(calls[1][2], calls[0][2])


if __name__ == '__main__':
    unittest.main()
