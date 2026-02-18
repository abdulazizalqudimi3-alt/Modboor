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
    Manager for Ollama models using the official ollama library.
    """

    def _setup_client(self):
        if settings.OLLAMA_HOST:
            os.environ["OLLAMA_HOST"] = settings.OLLAMA_HOST

    def list_models(self) -> List[Dict[str, Any]]:
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
        """Alias for pull_model"""
        return self.pull_model(model_name, **kwargs)

    def pull_model(self, model_name: str, **kwargs) -> bool:
        try:
            self._setup_client()
            logger.info(f"Pulling Ollama model: {model_name}...")
            ollama.pull(model_name)
            return True
        except Exception as e:
            logger.error(f"Error pulling Ollama model {model_name}: {e}")
            return False

    def delete_model(self, model_name: str) -> bool:
        """Alias for remove_model"""
        return self.remove_model(model_name)

    def remove_model(self, model_name: str) -> bool:
        try:
            self._setup_client()
            ollama.delete(model_name)
            return True
        except Exception as e:
            logger.error(f"Error deleting Ollama model {model_name}: {e}")
            return False

    def generate(self, model_name: str, prompt: str, **kwargs) -> str:
        try:
            self._setup_client()
            response = ollama.generate(model=model_name, prompt=prompt, **kwargs)
            return response.get('response', '')
        except Exception as e:
            logger.error(f"Error during Ollama inference: {e}")
            return f"Error: {str(e)}"

    def is_available(self) -> bool:
        try:
            self._setup_client()
            ollama.list()
            return True
        except Exception:
            return False

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        try:
            self._setup_client()
            info = ollama.show(model_name)
            return dict(info)
        except Exception as e:
            logger.error(f"Error getting Ollama model info for {model_name}: {e}")
            return None

    def run_model(self, model_name: str):
        """Basically ensures the model is loaded in Ollama (by doing a dummy generate)"""
        try:
            self._setup_client()
            ollama.generate(model=model_name, prompt="", keep_alive="5m")
            return True
        except Exception as e:
            logger.error(f"Error running Ollama model {model_name}: {e}")
            return False

    def install_ollama(self):
        return sys_install_ollama()

    def is_ollama_installed(self):
        return sys_is_ollama_installed()

    def start_ollama(self):
        return sys_start_ollama()

    def stop_ollama(self):
        return sys_stop_ollama()
