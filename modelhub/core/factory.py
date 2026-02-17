from typing import Dict
from .base import BaseModelManager
from .huggingface import HuggingFaceManager
from .ollama import OllamaManager

class ModelManagerFactory:
    _managers: Dict[str, BaseModelManager] = {}

    @classmethod
    def get_manager(cls, source: str) -> BaseModelManager:
        source = source.lower()
        if source not in cls._managers:
            if source == "huggingface" or source == "hf":
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
