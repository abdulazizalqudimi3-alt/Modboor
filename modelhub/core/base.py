from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseModelManager(ABC):
    """
    Abstract base class for all model managers (e.g., Hugging Face, Ollama).
    """

    @abstractmethod
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available/downloaded models.
        """
        pass

    @abstractmethod
    def download_model(self, model_name: str, **kwargs) -> bool:
        """
        Download a specific model.
        """
        pass

    @abstractmethod
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a specific model.
        """
        pass

    @abstractmethod
    def generate(self, model_name: str, prompt: str, **kwargs) -> str:
        """
        Generate a response from the model.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the model manager backend (e.g., Ollama service) is available.
        """
        pass
