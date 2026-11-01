from types import SimpleNamespace
import unittest
from unittest.mock import patch

from colmap_prepare import colmap_cpu_options


class ColmapCompatibilityTests(unittest.TestCase):
    def test_guided_matching_uses_advertised_option(self):
        help_text='--FeatureMatching.use_gpu --FeatureMatching.num_threads --FeatureMatching.guided_matching'
        with patch('colmap_prepare.subprocess.run',return_value=SimpleNamespace(stdout='',stderr=help_text)):
            flags=colmap_cpu_options('colmap','exhaustive_matcher','FeatureMatching','SiftMatching',guided=True)
        self.assertEqual(flags[-1],'--FeatureMatching.guided_matching')

    def test_colmap_cpu_options_selects_legacy_flags(self):
        help_text = ' '.join(('--SiftExtraction.use_gpu', '--SiftExtraction.num_threads',
                              '--SiftExtraction.max_image_size'))
        with patch('colmap_prepare.subprocess.run', return_value=SimpleNamespace(stdout=help_text)):
            self.assertEqual(colmap_cpu_options('colmap', 'feature_extractor', 'FeatureExtraction', 'SiftExtraction'), [
                '--SiftExtraction.use_gpu', '--SiftExtraction.num_threads', '--SiftExtraction.max_image_size'])


    def test_colmap_cpu_options_selects_current_flags(self):
        help_text = '--FeatureMatching.use_gpu --FeatureMatching.num_threads'
        with patch('colmap_prepare.subprocess.run', return_value=SimpleNamespace(stdout=help_text)):
            self.assertEqual(colmap_cpu_options('colmap', 'exhaustive_matcher', 'FeatureMatching', 'SiftMatching'), [
                '--FeatureMatching.use_gpu', '--FeatureMatching.num_threads'])


if __name__ == '__main__':
    unittest.main()
