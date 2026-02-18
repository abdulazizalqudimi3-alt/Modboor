import ollama
import os
from typing import List, Dict, Any, Optional
from .base import BaseModelManager
from modelhub.config.settings import settings
from modelhub.config.logging_config import get_logger
from modelhub.utils.system import (
    install_ollama as sys_install_ollama,
    is_ollama_installed as sys_is_ollama_installed,
    start_ollama as sys_start_ollama,
    stop_ollama as sys_stop_ollama
)

logger = get_logger(__name__)

class OllamaManager(BaseModelManager):
    """
    Manager for Ollama models and server.

    This class wraps the official ollama Python library and provides
    system-level controls for installing and running the Ollama server.
    """

    def _setup_client(self):
        """Configure the OLLAMA_HOST environment variable for the client."""
        if settings.OLLAMA_HOST:
            os.environ["OLLAMA_HOST"] = settings.OLLAMA_HOST

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all locally available Ollama models.

        Returns:
            List[Dict[str, Any]]: A list of model metadata.
        """
        try:
            self._setup_client()
            response = ollama.list()
            models = []
            model_list = getattr(response, 'models', [])
            for model in model_list:
                models.append({
                    "name": model.model if hasattr(model, 'model') else model.get('name', ''),
                    "type": "ollama",
                    "size": getattr(model, 'size', 0),
                    "modified_at": str(getattr(model, 'modified_at', '')),
                    "details": getattr(model, 'details', {})
                })
            return models
        except Exception as e:
            logger.error(f"Error listing Ollama models: {e}")
            return []

    def download_model(self, model_name: str, **kwargs) -> bool:
        """Alias for pull_model to satisfy BaseModelManager interface."""
        return self.pull_model(model_name, **kwargs)

    def pull_model(self, model_name: str, **kwargs) -> bool:
        """
        Download/pull a model from the Ollama registry.

        Args:
            model_name (str): The name of the model to pull.
            **kwargs: Ignored for now.

        Returns:
            bool: True if successful.
        """
        try:
            self._setup_client()
            logger.info(f"Pulling Ollama model: {model_name}...")
            ollama.pull(model_name)
            return True
        except Exception as e:
            logger.error(f"Error pulling Ollama model {model_name}: {e}")
            return False

    def delete_model(self, model_name: str) -> bool:
        """Alias for remove_model to satisfy BaseModelManager interface."""
        return self.remove_model(model_name)

    def remove_model(self, model_name: str) -> bool:
        """
        Remove a model from the local Ollama storage.

        Args:
            model_name (str): The model to delete.

        Returns:
            bool: True if deleted.
        """
        try:
            self._setup_client()
            logger.info(f"Removing Ollama model: {model_name}...")
            ollama.delete(model_name)
            return True
        except Exception as e:
            logger.error(f"Error deleting Ollama model {model_name}: {e}")
            return False

    def generate(self, model_name: str, prompt: str, **kwargs) -> str:
        """
        Run inference on an Ollama model.

        Args:
            model_name (str): Name of the model.
            prompt (str): Input text.
            **kwargs: Generation parameters.

        Returns:
            str: Generated response text.
        """
        try:
            self._setup_client()
            response = ollama.generate(model=model_name, prompt=prompt, **kwargs)
            return response.get('response', '')
        except Exception as e:
            logger.error(f"Error during Ollama inference for {model_name}: {e}")
            return f"Error: {str(e)}"

    def is_available(self) -> bool:
        """Check if the Ollama server is running and reachable."""
        try:
            self._setup_client()
            ollama.list()
            return True
        except Exception:
            return False

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get details about a specific Ollama model.

        Args:
            model_name (str): The model name.

        Returns:
            Optional[Dict[str, Any]]: Model details or None.
        """
        try:
            self._setup_client()
            info = ollama.show(model_name)
            return dict(info)
        except Exception as e:
            logger.error(f"Error getting Ollama model info for {model_name}: {e}")
            return None

    def load_model(self, model_name: str, **kwargs) -> bool:
        """Alias for run_model to satisfy interface."""
        return self.run_model(model_name)

    def run_model(self, model_name: str) -> bool:
        """
        Load a model into memory in Ollama and keep it alive.

        Args:
            model_name (str): The model to load.

        Returns:
            bool: True if load signal was successful.
        """
        try:
            self._setup_client()
            logger.info(f"Loading Ollama model {model_name}...")
            ollama.generate(model=model_name, prompt="", keep_alive="5m")
            return True
        except Exception as e:
            logger.error(f"Error running Ollama model {model_name}: {e}")
            return False

    def unload_model(self, model_name: str) -> bool:
        """
        Unload an Ollama model from memory by setting keep_alive to 0.
        """
        try:
            self._setup_client()
            logger.info(f"Unloading Ollama model {model_name}...")
            ollama.generate(model=model_name, prompt="", keep_alive=0)
            return True
        except Exception as e:
            logger.error(f"Error unloading Ollama model {model_name}: {e}")
            return False

    def install_ollama(self) -> (bool, str):
        """Install Ollama on the system."""
        return sys_install_ollama()

    def is_ollama_installed(self) -> bool:
        """Check if Ollama is installed."""
        return sys_is_ollama_installed()

    def start_ollama(self) -> (bool, str):
        """Start the Ollama server process."""
        return sys_start_ollama()

    def stop_ollama(self) -> (bool, str):
        """Stop the Ollama server process."""
        return sys_stop_ollama()
