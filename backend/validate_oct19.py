"""Reproducible bounded regression batch; no fresh provider reconstruction."""
import io
import json
from pathlib import Path
import time
import unittest

PATTERNS = [
    'test_gaussian_cache.py', 'test_photographic_surface.py',
    'test_gaussian_integration.py',
    'test_gaussian_geometry.py',
    'test_local_reconstruct.py', 'test_local_reconstruction_review.py', 'test_local_splat_quality.py', 'test_run_timing.py',
    'test_provider_resilience.py', 'test_provider_quota.py', 'test_groq_detail_fallback.py',
    'test_groq_provider.py', 'test_reconstruction_reliability.py',
    'test_camera_recovery_routes.py', 'test_speed_visual_reliability.py',
    'test_mounted_extent_guard.py', 'test_quality_pipeline.py', 'test_texture_color_space.py',
]


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    results = []
    for pattern in PATTERNS:
        start = time.monotonic()
        suite = unittest.defaultTestLoader.discover(str(root), pattern=pattern)
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
        item = dict(module=pattern, tests=result.testsRun, failures=len(result.failures),
                    errors=len(result.errors), skipped=len(result.skipped),
                    seconds=round(time.monotonic()-start, 3), passed=result.wasSuccessful())
        results.append(item)
        print(json.dumps(item), flush=True)
        if not result.wasSuccessful():
            print(stream.getvalue())
    report = dict(date='2026-10-19', tests=sum(item['tests'] for item in results),
                  passed=all(item['passed'] for item in results), modules=results,
                  notice='Focused regressions with provider mocks plus real local Blender export tests; '
                         'not fresh provider/room accuracy evidence.')
    (root.parent / 'reports/oct19-regressions.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    raise SystemExit(0 if report['passed'] else 1)
