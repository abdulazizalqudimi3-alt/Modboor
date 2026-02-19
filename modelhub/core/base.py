from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseModelManager(ABC):
    """
    Abstract base class for all model managers.
    Provides a unified interface for different model sources.
    """

    @abstractmethod
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models from this source."""
        pass

    @abstractmethod
    def download_model(self, model_id: str, **kwargs) -> bool:
        """Download a specific model."""
        pass

    @abstractmethod
    def delete_model(self, model_id: str) -> bool:
        """Delete a specific model from local storage."""
        pass

    @abstractmethod
    def generate(self, model_id: str, prompt: str, **kwargs) -> str:
        """Run inference on the model."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend/source is available."""
        pass

    @abstractmethod
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model."""
        pass

    @abstractmethod
    def load_model(self, model_id: str, **kwargs) -> bool:
        """Load a model into memory."""
        pass

    @abstractmethod
    def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory."""
        pass
