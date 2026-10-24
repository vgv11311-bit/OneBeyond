"""
Engine Registry and Factory for 3D Reconstruction Engines.
Enables dynamic discovery, registration, and swapping of reconstruction engines.
"""

from typing import Dict, Type, List, Optional
from .base_engine import BaseReconstructionEngine
from .feature_sfm_engine import FeatureSfMEngine
from .sample_engine import SampleSpatialEngine
from .gemini_scene_engine import GeminiSceneEngine
from .groq_scene_engine import GroqSceneEngine
from .photographic_engine import PhotographicEngine
from .vggt_engine import VggtEngine, availability as vggt_availability
from .resplat_engine import ResplatEngine, availability as resplat_availability


class EngineRegistry:
    """Registry maintaining available 3D Reconstruction Engines."""

    _engines: Dict[str, Type[BaseReconstructionEngine]] = {}
    _instances: Dict[str, BaseReconstructionEngine] = {}

    @classmethod
    def register(cls, engine_id: str, engine_class: Type[BaseReconstructionEngine]):
        cls._engines[engine_id] = engine_class

    @classmethod
    def get_engine(cls, engine_id: Optional[str] = None) -> BaseReconstructionEngine:
        """
        Retrieves an instance of the requested engine, defaulting to photographic.
        """
        if not engine_id:
            engine_id = "photographic"
        if engine_id not in cls._engines:
            raise ValueError("Unknown reconstruction engine.")

        if engine_id not in cls._instances:
            cls._instances[engine_id] = cls._engines[engine_id]()

        return cls._instances[engine_id]

    @classmethod
    def list_engines(cls) -> List[Dict[str, str]]:
        """Returns metadata for all registered engines."""
        results = []
        for eid, eclass in cls._engines.items():
            instance = cls.get_engine(eid)
            results.append({
                "id": eid,
                "name": instance.name,
                "description": instance.description,
                "version": instance.version,
                "is_default": eid == "photographic",
                **({'experimental':True,**vggt_availability()} if eid=='vggt_experimental' else {}),
                **({'experimental':True,**resplat_availability()} if eid=='resplat_experimental' else {}),
            })
        return results


# Register built-in engines
EngineRegistry.register("photographic", PhotographicEngine)
EngineRegistry.register("vggt_experimental", VggtEngine)
EngineRegistry.register("resplat_experimental", ResplatEngine)
EngineRegistry.register("feature_sfm", FeatureSfMEngine)
EngineRegistry.register("gemini_scene", GeminiSceneEngine)
EngineRegistry.register("groq_scene", GroqSceneEngine)
EngineRegistry.register("sample_architectural", SampleSpatialEngine)
