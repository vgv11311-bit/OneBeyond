from pathlib import Path
import tempfile
import unittest
from reconstruction.resplat_contract import context_count,validate_dataset


class WorkerContractTests(unittest.TestCase):
    def test_96_registered_views_still_use_only_eight_model_contexts(self):
        self.assertEqual(context_count(96),8)
        self.assertEqual(context_count(120),8)
        with self.assertRaises(ValueError):context_count(121)

    def test_preflight_accepts_matching_96_view_dataset_and_rejects_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'images').mkdir();(root/'sparse').mkdir()
            for i in range(96):(root/'images'/f'{i:05d}.jpg').touch()
            (root/'sparse/images.txt').write_text(''.join(f'{i+1} 1 0 0 0 0 0 0 1 {i:05d}.jpg\n\n' for i in range(96)))
            self.assertEqual(validate_dataset(root),96)
            (root/'images/00095.jpg').unlink()
            with self.assertRaisesRegex(ValueError,'differ'):validate_dataset(root)


class FullInferenceWindowTests(unittest.TestCase):
    def test_video_above_upstream_sixty_frame_default_keeps_every_registered_view(self):
        import argparse
        from reconstruction.resplat_contract import inference_frame_window,validate_inference_window
        parser=argparse.ArgumentParser();parser.add_argument('--start_frame',type=int,default=0);parser.add_argument('--frame_distance',type=int,default=60)
        default=parser.parse_args([]);self.assertEqual(default.frame_distance,60)
        for count in (5,60,96,120):
            actual=parser.parse_args(inference_frame_window(count))
            self.assertEqual(list(range(actual.start_frame,min(count,actual.start_frame+actual.frame_distance))),list(range(count)))
            validate_inference_window(count,count)
        with self.assertRaisesRegex(ValueError,'omitted'):validate_inference_window(96,60)
