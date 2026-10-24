import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as app_module


class SavedSpacesListingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.spaces_dir = Path(self.temp.name)
        self.patch = patch.object(app_module, 'SPACES_DIR', self.spaces_dir)
        self.patch.start()
        self.client = app_module.app.test_client()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def write_space(self, job_id, metadata):
        target = self.spaces_dir / job_id
        target.mkdir()
        (target / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')

    def test_genuine_empty_collection(self):
        response = self.client.get('/api/spaces')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {'spaces': [], 'count': 0})
        self.assertEqual(response.headers['Cache-Control'], 'no-store')

    def test_completed_and_saved_review_entries(self):
        self.write_space('hall-audit', {'job_id': 'hall-audit', 'name': 'Hall — audited wall coverage (quality still limited)', 'status': 'completed'})
        self.write_space('hall-resplat', {'job_id': 'hall-resplat', 'name': 'Hall — ReSplat experimental (review, distortion remains)', 'status': 'review_required', 'engine_id': 'resplat_experimental', 'gaussian': {'saved_to_recents': True, 'requires_review': True, 'quality_accepted': False}})
        response = self.client.get('/api/spaces')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['count'], 2)
        self.assertEqual({space['job_id'] for space in response.json['spaces']}, {'hall-audit', 'hall-resplat'})

    def test_corrupt_record_is_not_reported_as_successful_empty(self):
        target = self.spaces_dir / 'broken'
        target.mkdir()
        (target / 'metadata.json').write_text('{broken json', encoding='utf-8')
        response = self.client.get('/api/spaces')
        self.assertEqual(response.status_code, 503)
        self.assertIn('error', response.json)


if __name__ == '__main__':
    unittest.main()
