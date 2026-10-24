"""
Base abstract interface for 3D Reconstruction Engines.
Enables plug-and-play replacement of photogrammetry / SfM / NeRF / Gaussian Splatting backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any, List
from pathlib import Path


@dataclass
class ReconstructionMetrics:
    total_frames_processed: int = 0
    keypoints_detected: int = 0
    points_3d_triangulated: int = 0
    estimated_room_width: float = 0.0
    estimated_room_length: float = 0.0
    estimated_room_height: float = 0.0
    processing_duration_sec: float = 0.0
    engine_name: str = ""
    extra_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconstructionResult:
    success: bool
    job_id: str
    output_glb_path: str
    metrics: ReconstructionMetrics
    error_message: Optional[str] = None
    preview_image_path: Optional[str] = None
    error_code: Optional[str] = None
    retry_after_seconds: int = 0
    gaussian: Optional[Dict[str, Any]] = None


ProgressCallback = Callable[[float, str, Optional[Dict[str, Any]]], None]


class BaseReconstructionEngine(ABC):
    """
    Abstract Base Class for modular 3D Reconstruction Engines.
    Any new 3D engine (e.g., COLMAP, Gaussian Splatting, NeRF, AI Monocular Depth)
    implements this interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the engine."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the reconstruction technique."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Engine version string."""
        pass

    @abstractmethod
    def reconstruct(
        self,
        job_id: str,
        input_type: str,  # 'photos' | 'video'
        file_paths: List[str],
        output_dir: Path,
        options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> ReconstructionResult:
        """
        Execute the 3D reconstruction pipeline.
        
        Args:
            job_id: Unique identifier for the job.
            input_type: 'photos' or 'video'.
            file_paths: List of absolute file paths to input media.
            output_dir: Directory where generated artifacts (GLB, metadata) should be stored.
            options: Optional configuration dictionary.
            progress_callback: Callback function receiving (progress_pct: 0.0-100.0, stage_description: str, telemetry_dict).
            
        Returns:
            ReconstructionResult containing metrics and a GLB path when available,
            or a photographic artifact descriptor for a Gaussian reconstruction.
        """
        pass
