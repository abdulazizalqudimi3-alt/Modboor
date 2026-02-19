from typing import Dict
from modelhub.core.base import BaseModelManager
from modelhub.managers.huggingface import HuggingFaceManager
from modelhub.managers.ollama import OllamaManager

class ModelManagerFactory:
    """
    Factory for creating and managing ModelManager instances.
    """
    _managers: Dict[str, BaseModelManager] = {}

    @classmethod
    def get_manager(cls, source: str) -> BaseModelManager:
        source = source.lower()
        if source not in cls._managers:
            if source in ["huggingface", "hf"]:
                cls._managers[source] = HuggingFaceManager()
            elif source == "ollama":
                cls._managers[source] = OllamaManager()
            else:
                raise ValueError(f"Unsupported model source: {source}")
        return cls._managers[source]

    @classmethod
    def get_all_managers(cls) -> Dict[str, BaseModelManager]:
        return {
            "huggingface": cls.get_manager("huggingface"),
            "ollama": cls.get_manager("ollama")
        }
