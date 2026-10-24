import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import app as server
from reconstruction.capture_intake import inspect_photo


def image_bytes(width=900, height=720, format='JPEG'):
    stream = io.BytesIO()
    Image.new('RGB', (width, height), '#8799ab').save(stream, format=format)
    return stream.getvalue()


class CaptureIntakeTests(unittest.TestCase):
    def test_dark_and_clipped_photos_are_warned_not_rejected(self):
        for shade, warning in ((12, 'Underexposed'), (250, 'Overexposed')):
            stream = io.BytesIO()
            Image.new('RGB', (900, 720), (shade, shade, shade)).save(stream, format='JPEG')
            check = inspect_photo(io.BytesIO(stream.getvalue()), '.jpg')
            self.assertTrue(any(warning in item for item in check['warnings']))
            self.assertIn('detail_score', check)

    def test_decode_restores_stream_and_checks_format(self):
        content = image_bytes()
        stream = io.BytesIO(content)
        self.assertEqual(inspect_photo(stream, '.jpg')['width'], 900)
        self.assertEqual(stream.read(), content)
        with self.assertRaisesRegex(ValueError, 'format'):
            inspect_photo(io.BytesIO(content), '.png')
        with self.assertRaisesRegex(ValueError, 'unreadable'):
            inspect_photo(io.BytesIO(content[:-70]), '.jpg')

    def test_upload_rejects_corrupt_view_before_job_creation(self):
        with tempfile.TemporaryDirectory() as directory, patch('app.UPLOADS_DIR', Path(directory)):
            response = server.app.test_client().post('/api/upload', data={
                'files': [(io.BytesIO(image_bytes()), 'first.jpg'), (io.BytesIO(b'not an image'), 'second.jpg')]
            })
            self.assertEqual(response.status_code, 400)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_upload_preserves_two_ordered_views_and_checks(self):
        with tempfile.TemporaryDirectory() as uploads, tempfile.TemporaryDirectory() as spaces, \
             patch('app.UPLOADS_DIR', Path(uploads)), patch('app.SPACES_DIR', Path(spaces)), \
             patch.dict(server.JOBS, {}, clear=True):
            response = server.app.test_client().post('/api/upload', data={
                'files': [(io.BytesIO(image_bytes()), 'overview.jpg'),
                          (io.BytesIO(image_bytes(700, 550, 'PNG')), 'detail.png')]
            })
            self.assertEqual(response.status_code, 201)
            result = response.json
            self.assertEqual(result['files_count'], 2)
            self.assertEqual(result['photo_checks'][1]['format'], 'PNG')
            self.assertTrue(result['photo_checks'][1]['warnings'])
            self.assertTrue(any('Low resolution' in warning for warning in result['capture_warnings']))
            state = server.load_job_state(result['job_id'])
            self.assertEqual([Path(path).name for path in state['file_paths']],
                             ['00_overview.jpg', '01_detail.png'])

    def test_duplicate_views_are_reported_without_discarding_sources(self):
        with tempfile.TemporaryDirectory() as uploads, tempfile.TemporaryDirectory() as spaces, \
             patch('app.UPLOADS_DIR', Path(uploads)), patch('app.SPACES_DIR', Path(spaces)), \
             patch.dict(server.JOBS, {}, clear=True):
            photo = image_bytes()
            response = server.app.test_client().post('/api/upload', data={
                'files': [(io.BytesIO(photo), 'a.jpg'), (io.BytesIO(photo), 'b.jpg')]
            })
            self.assertEqual(response.status_code, 201)
            self.assertTrue(any('Identical photos' in warning
                                for warning in response.json['capture_warnings']))

    def test_accepts_more_than_twelve_photos_for_photographic_coverage(self):
        with tempfile.TemporaryDirectory() as uploads, tempfile.TemporaryDirectory() as spaces, \
             patch('app.UPLOADS_DIR', Path(uploads)), patch('app.SPACES_DIR', Path(spaces)):
            response = server.app.test_client().post('/api/upload', data={'files': [
                (io.BytesIO(image_bytes()), f'{index}.jpg') for index in range(20)]})
            self.assertEqual(response.status_code, 201)
            self.assertEqual(response.json['files_count'], 20)

    def test_video_intake_is_separate_from_photo_intake_and_restores_original(self):
        with tempfile.TemporaryDirectory() as uploads, tempfile.TemporaryDirectory() as spaces, \
             patch('app.UPLOADS_DIR', Path(uploads)), patch('app.SPACES_DIR', Path(spaces)), \
             patch('app.inspect_video', return_value={'duration_seconds':30, 'warnings':[]}) as probe:
            content = b'video fixture'
            response = server.app.test_client().post('/api/upload', data={'files': [(io.BytesIO(content), 'walk.mp4')]})
            self.assertEqual(response.status_code, 201)
            self.assertEqual(response.json['input_type'], 'video')
            state = server.load_job_state(response.json['job_id'])
            self.assertEqual(Path(state['file_paths'][0]).read_bytes(), content)
            self.assertEqual(probe.call_count, 1)
            mixed = server.app.test_client().post('/api/upload', data={'files': [
                (io.BytesIO(content), 'walk.mp4'), (io.BytesIO(image_bytes()), 'photo.jpg')]})
            self.assertEqual(mixed.status_code, 400)

    def test_unreadable_video_is_rejected_before_a_job_is_created(self):
        with tempfile.TemporaryDirectory() as uploads, patch('app.UPLOADS_DIR', Path(uploads)), \
             patch('app.inspect_video', side_effect=ValueError('Walkthrough video is unreadable')):
            response = server.app.test_client().post('/api/upload', data={'files': [(io.BytesIO(b'invalid'), 'walk.mp4')]})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(list(Path(uploads).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
