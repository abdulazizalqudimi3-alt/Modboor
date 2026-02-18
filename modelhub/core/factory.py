from typing import Dict
from .base import BaseModelManager
from .huggingface import HuggingFaceManager
from .ollama import OllamaManager

class ModelManagerFactory:
    """
    Factory for creating and managing ModelManager instances.

    This class ensures that only one instance of each manager is created
    and provides a centralized way to access them.
    """
    _managers: Dict[str, BaseModelManager] = {}

    @classmethod
    def get_manager(cls, source: str) -> BaseModelManager:
        """
        Get or create a manager instance for the given source.

        Args:
            source (str): The model source (e.g., 'huggingface', 'ollama').

        Returns:
            BaseModelManager: The manager instance.

        Raises:
            ValueError: If the source is not supported.
        """
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
        """
        Get instances of all supported managers.

        Returns:
            Dict[str, BaseModelManager]: Dictionary of source names to manager instances.
        """
        return {
            "huggingface": cls.get_manager("huggingface"),
            "ollama": cls.get_manager("ollama")
        }
