"""The supported COLMAP + ReSplat reconstruction entry point."""
from .resplat_engine import ResplatEngine, availability

class EngineRegistry:
    _engines = {'resplat_experimental': ResplatEngine}
    _instances = {}

    @classmethod
    def get_engine(cls, engine_id=None):
        engine_id = engine_id or 'resplat_experimental'
        if engine_id not in cls._engines:
            raise ValueError('Unknown reconstruction engine')
        if engine_id not in cls._instances:
            cls._instances[engine_id] = cls._engines[engine_id]()
        return cls._instances[engine_id]

    @classmethod
    def list_engines(cls):
        instance = cls.get_engine()
        return [dict(id='resplat_experimental',name=instance.name,
            description=instance.description,version=instance.version,
            is_default=True,experimental=True,**availability())]
