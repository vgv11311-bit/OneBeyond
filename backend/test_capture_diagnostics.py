import json
from pathlib import Path
import tempfile
import unittest
from reconstruction.capture_diagnostics import frame_selection_summary,camera_coverage_failure

class CaptureDiagnosticTests(unittest.TestCase):
    def test_selection_summary_is_small_and_excludes_private_frame_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'selection.json'
            data=dict(candidate_count=3,selected_count=1,frames=[dict(status='selected',frame='private-source',low_detail_warning=True),dict(status='duplicate'),dict(status='unreadable')])
            path.write_text(json.dumps(data));summary=frame_selection_summary(path)
            self.assertEqual(summary['selected_frames'],1)
            self.assertEqual(summary['discarded_by_reason'],dict(duplicate=1,unreadable=1))
            self.assertNotIn('private-source',json.dumps(summary))
            data['selected_count']=3;path.write_text(json.dumps(data));self.assertIsNone(frame_selection_summary(path))

    def test_coverage_error_contains_counts_not_native_paths_and_does_not_relabel_other_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'report.json'
            data=dict(status='failed',registered_images=3,input_images=10,points3d=386,error='C:/private/tool/path')
            path.write_text(json.dumps(data));message=camera_coverage_failure(path)
            self.assertIn('3/10',message);self.assertNotIn('private',message)
            data['registered_images']=8;path.write_text(json.dumps(data));self.assertIsNone(camera_coverage_failure(path))
            data['registered_images']=-1;path.write_text(json.dumps(data));self.assertIsNone(camera_coverage_failure(path))

if __name__=='__main__':unittest.main()
