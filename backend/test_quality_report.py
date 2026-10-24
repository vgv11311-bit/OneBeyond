import unittest

from reconstruction.quality_report import assess_job


class QualityReportTests(unittest.TestCase):
    def test_any_space_uses_evidence_not_expected_furniture(self):
        job = {'job_id': 'example', 'status': 'completed', 'files_count': 3,
               'space_type': 'kitchen', 'capture_warnings': [],
               'metrics': {'total_frames_processed': 3, 'extra_details': {
                   'multi_view_support': {'geometry_verified': True, 'metric_tracks': 60,
                                          'photo_coverage_fraction': .5},
                   'scale_calibration': {'applied': True},
                   'lighting': {'wall_appearance_status': 'photo_evidence',
                                'multiview_object_evidence': [
                                    {'index': 0, 'label': 'sink', 'status': 'cross_view_observed'}]},
                   'self_correction': {'status': 'accepted'}}}}
        report = assess_job(job)
        self.assertEqual(report['space_type'], 'kitchen')
        self.assertEqual(report['findings'], [])
        self.assertFalse(report['visual_fidelity_verified'])

    def test_sparse_object_support_is_not_called_absent(self):
        job = {'status': 'completed', 'files_count': 2, 'capture_warnings': [],
               'metrics': {'total_frames_processed': 2, 'extra_details': {
                   'multi_view_support': {'geometry_verified': True,
                                          'photo_coverage_fraction': .25},
                   'scale_calibration': {'applied': False},
                   'lighting': {'wall_appearance_status': 'photo_evidence',
                       'multiview_object_evidence': [
                           {'index': 1, 'label': 'sofa', 'status': 'not_independently_established'}]},
                   'self_correction': {'status': 'accepted'}}}}
        report = assess_job(job)
        finding = next(item for item in report['findings'] if item['code']=='object_cross_view_sparse')
        self.assertIn('does not prove', finding['finding'])
        self.assertTrue(any(item['code']=='scale_estimated' for item in report['findings']))

    def test_legacy_job_is_unassessed_not_failed(self):
        report = assess_job({'status': 'completed', 'files_count': 3,
                             'metrics': {'total_frames_processed': 3, 'extra_details': {
                                 'multi_view_support': {'matched_views': 2, 'verified_matches': 40}}}})
        self.assertTrue(any(item['code']=='multiview_not_assessed' for item in report['findings']))
        self.assertFalse(any(item['code']=='multiview_unverified' for item in report['findings']))


if __name__ == '__main__':
    unittest.main()
