from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import json
from pathlib import Path
import tempfile
import unittest

from reconstruction.base_engine import ReconstructionResult, ReconstructionMetrics
from reconstruction.run_timing import measure, measured_call, profile_pipeline


class TimingTests(unittest.TestCase):
    def test_real_pipeline_invalid_input_retains_failed_timing(self):
        from reconstruction.feature_sfm_engine import FeatureSfMEngine
        with tempfile.TemporaryDirectory() as directory:
            result = FeatureSfMEngine().reconstruct('invalid', 'photos', [], Path(directory))
            self.assertFalse(result.success)
            self.assertIn('run_timing', result.metrics.extra_details)
            self.assertFalse(json.loads(Path(directory, 'run_timing.json').read_text())['success'])

    def test_success_and_thread_spans_have_no_source_or_credentials(self):
        class Engine:
            @profile_pipeline
            def reconstruct(self, job_id, input_type, paths, output):
                with measure('analysis_total'):
                    with measure('quota_wait'):
                        pass
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        pool.submit(copy_context().run, measured_call, 'local_depth', lambda: None).result()
                return ReconstructionResult(True, job_id, '', ReconstructionMetrics())
        with tempfile.TemporaryDirectory() as directory:
            result = Engine().reconstruct('job', 'photos', ['private-photo-secret'], directory)
            report = json.loads(Path(directory, 'run_timing.json').read_text())
            self.assertTrue(report['success'])
            self.assertTrue(report['within_target'])
            self.assertIn('quota_wait', report['measured_span_seconds'])
            self.assertIn('local_depth', result.metrics.extra_details['run_timing']['measured_span_seconds'])
            self.assertNotIn('private-photo-secret', json.dumps(report))

    def test_failed_attempt_saved_without_turning_failure_into_success(self):
        class Engine:
            @profile_pipeline
            def reconstruct(self, job_id, input_type, paths, output):
                with measure('provider_request'):
                    raise ValueError('unavailable')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                Engine().reconstruct('job', 'photos', [], directory)
            report = json.loads(Path(directory, 'run_timing.json').read_text())
            self.assertFalse(report['success'])
            self.assertFalse(report['spans'][0]['success'])

    def test_tracker_is_reset_between_attempts(self):
        class Engine:
            @profile_pipeline
            def reconstruct(self, job_id, input_type, paths, output):
                return ReconstructionResult(False, job_id, '', ReconstructionMetrics())
        with tempfile.TemporaryDirectory() as directory:
            measured_call('outside', lambda: None)
            result = Engine().reconstruct('job', 'photos', [], directory)
            self.assertEqual(result.metrics.extra_details['run_timing']['spans'], [])


if __name__ == '__main__':
    unittest.main()
