"""
Reference / Sample Spatial Reconstruction Engine.
Creates high-fidelity architectural 3D room spaces with realistic materials,
proportions, and spatial features for testing and baseline comparison.
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base_engine import (
    BaseReconstructionEngine,
    ReconstructionResult,
    ReconstructionMetrics,
    ProgressCallback,
)
from .frame_extractor import FrameExtractor
from .asset_library import AssetLibrary


class SampleSpatialEngine(BaseReconstructionEngine):
    """
    Sample Spatial Engine for baseline comparison, rapid benchmarking,
    and fallback testing.
    """

    @property
    def name(self) -> str:
        return "Architectural Spatial Generator (Sample Reference)"

    @property
    def description(self) -> str:
        return "Original reusable GLB assets assembled into a colourful walkable home"

    @property
    def version(self) -> str:
        return "2.0.0"

    def reconstruct(
        self,
        job_id: str,
        input_type: str,
        file_paths: List[str],
        output_dir: Path,
        options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> ReconstructionResult:
        start_time = time.time()
        options = options or {}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = output_dir / "extracted_frames"

        def report(progress: float, stage: str, extra: Optional[Dict[str, Any]] = None):
            if progress_callback:
                progress_callback(progress, stage, extra)

        try:
            report(10.0, "Analyzing room media input...")
            
            # Extract sample frames if input provided
            if file_paths:
                if input_type == "video":
                    frame_paths = FrameExtractor.extract_from_video(file_paths[0], frames_dir, max_frames=10)
                else:
                    frame_paths = FrameExtractor.load_and_preprocess_photos(file_paths, frames_dir, max_photos=10)
            else:
                frame_paths = []

            report(35.0, "Creating reusable original furniture GLB assets...")
            asset_paths = AssetLibrary.ensure_generated()
            report(65.0, f"Arranging {len(asset_paths)} assets into the sample home...")
            time.sleep(0.25)

            report(85.0, "Exporting colourful sample-home GLB scene...")
            output_glb = str(output_dir / "model.glb")
            AssetLibrary.create_sample_home(output_glb)

            report(100.0, "Sample room space generated successfully.")
            duration = time.time() - start_time

            metrics = ReconstructionMetrics(
                total_frames_processed=len(frame_paths),
                keypoints_detected=0,
                points_3d_triangulated=0,
                estimated_room_width=8.0,
                estimated_room_length=10.0,
                estimated_room_height=3.0,
                processing_duration_sec=round(duration, 2),
                engine_name=self.name,
                extra_details={"asset_count": len(asset_paths), "asset_library": "backend/assets/models"},
            )

            return ReconstructionResult(
                success=True,
                job_id=job_id,
                output_glb_path=output_glb,
                metrics=metrics,
                preview_image_path=frame_paths[0] if frame_paths else None,
            )

        except Exception as e:
            return ReconstructionResult(
                success=False,
                job_id=job_id,
                output_glb_path="",
                metrics=ReconstructionMetrics(engine_name=self.name),
                error_message=str(e),
            )
