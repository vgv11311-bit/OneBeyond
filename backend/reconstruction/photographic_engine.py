"""Photographic reconstruction without semantic providers or a fabricated GLB."""
import time
from pathlib import Path

from .base_engine import BaseReconstructionEngine, ReconstructionMetrics, ReconstructionResult
from .gaussian_pipeline import build_companion


class PhotographicEngine(BaseReconstructionEngine):
    name = 'Photographic room'
    description = 'Reconstruct overlapping photos or a walkthrough with COLMAP and Brush. No Groq or Gemini calls.'
    version = '1.0.0'

    def reconstruct(self, job_id, input_type, file_paths, output_dir, options=None, progress_callback=None):
        started = time.monotonic()
        metrics = ReconstructionMetrics(engine_name=self.name,
            extra_details={'provider_calls': 0, 'scale': 'unmeasured', 'pipeline': 'colmap_brush'})
        def progress(pct, message):
            if progress_callback:
                progress_callback(max(5, min(98, (pct - 85) * 7)), message, None)
        result = build_companion(job_id, file_paths, Path(output_dir), progress,
                                 standalone=True, input_type=input_type)
        metrics.processing_duration_sec = round(time.monotonic() - started, 3)
        metrics.total_frames_processed = result.get('registered_views', 0)
        metrics.points_3d_triangulated = result.get('sparse_points', 0)
        metrics.extra_details.update(cleanup=result.get('cleanup'), depth_surface=result.get('depth_surface'),
                                     cache_hit=result.get('cache_hit', False),frame_selection=result.get('frame_selection'))
        ready = result.get('status') == 'ready'
        return ReconstructionResult(ready, job_id, '', metrics,
            error_message=None if ready else result.get('reason', 'Photographic reconstruction could not finish.'),
            error_code=None if ready else 'photographic_unavailable',
            preview_image_path=file_paths[0] if input_type == 'photos' and file_paths else None,
            gaussian=result if ready else None)
