# Reconstruction package
from .base_engine import BaseReconstructionEngine, ReconstructionResult, ProgressCallback

def __getattr__(name):
    # Budget/locking utilities must not initialize CV engines in every process.
    if name == 'EngineRegistry':
        from .engine_registry import EngineRegistry
        return EngineRegistry
    raise AttributeError(name)

__all__ = ["BaseReconstructionEngine", "ReconstructionResult", "ProgressCallback", "EngineRegistry"]
