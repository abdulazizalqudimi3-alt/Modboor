import asyncio
from typing import List, Dict, Any, Optional
from .factory import ModelManagerFactory
from modelhub.config.logging_config import get_logger
from modelhub.config.settings import settings
from modelhub.utils.system import (
    install_ollama as sys_install_ollama,
    is_ollama_installed as sys_is_ollama_installed,
    start_ollama as sys_start_ollama,
    stop_ollama as sys_stop_ollama
)

logger = get_logger(__name__)

class ModelService:
    """
    Service Layer that orchestrates model operations across multiple sources.

    This class implements the Singleton pattern and provides a high-level,
    asynchronous API for the FastAPI server and library users.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.managers = ModelManagerFactory.get_all_managers()
        self._initialized = True

    async def list_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all models from all available managers.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Grouped models by source.
        """
        results = {}
        for source, manager in self.managers.items():
            results[source] = manager.list_models()
        return results

    async def find_model_source(self, model_name: str) -> Optional[str]:
        """
        Determine the source of a model by checking all managers.

        Args:
            model_name (str): Name or ID of the model.

        Returns:
            Optional[str]: Source name ('huggingface' or 'ollama') or None.
        """
        for source, manager in self.managers.items():
            models = manager.list_models()
            if any(m['name'] == model_name for m in models):
                return source
        return None

    async def download_model(self, source: str, model_id: str, **kwargs) -> bool:
        """
        Asynchronously download a model.

        Args:
            source (str): 'huggingface' or 'ollama'.
            model_id (str): Model identifier.
            **kwargs: Extra arguments.

        Returns:
            bool: Success status.
        """
        manager = ModelManagerFactory.get_manager(source)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: manager.download_model(model_id, **kwargs))

    async def pull_model(self, source: str, model_name: str, **kwargs) -> bool:
        """Alias for download_model."""
        return await self.download_model(source, model_name, **kwargs)

    async def delete_model(self, source: str, model_id: str) -> bool:
        """
        Delete a model from local storage.

        Args:
            source (str): Source name.
            model_id (str): Model identifier.

        Returns:
            bool: Success status.
        """
        manager = ModelManagerFactory.get_manager(source)
        return manager.delete_model(model_id)

    async def remove_model(self, source: str, model_name: str) -> bool:
        """Alias for delete_model."""
        return await self.delete_model(source, model_name)

    async def generate_response(self, source: str, model_name: str, prompt: str, **kwargs) -> str:
        """
        Run inference on a model.

        Args:
            source (str): Source name.
            model_name (str): Model name.
            prompt (str): Input text.
            **kwargs: Inference params.

        Returns:
            str: Generated text.
        """
        manager = ModelManagerFactory.get_manager(source)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: manager.generate(model_name, prompt, **kwargs))

    async def get_info(self, source: str, model_id: str) -> Optional[Dict[str, Any]]:
        """Fetch model metadata."""
        manager = ModelManagerFactory.get_manager(source)
        return manager.get_model_info(model_id)

    async def load_model(self, source: str, model_id: str, **kwargs) -> bool:
        """
        Load a model into memory.

        Args:
            source (str): Source name.
            model_id (str): Model identifier.
            **kwargs: Load params.

        Returns:
            bool: Success status.
        """
        manager = ModelManagerFactory.get_manager(source)
        if source == "ollama":
            return manager.run_model(model_id)
        elif source == "huggingface":
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: manager.load_model(model_id, **kwargs))
        return False

    async def unload_model(self, source: str, model_id: str) -> bool:
        """Remove a model from memory."""
        manager = ModelManagerFactory.get_manager(source)
        if hasattr(manager, 'unload_model'):
            return manager.unload_model(model_id)
        return True

    async def get_status(self) -> Dict[str, Any]:
        """Get the availability status of all sources."""
        return {
            "sources": {
                source: manager.is_available() for source, manager in self.managers.items()
            }
        }

    def install_ollama(self) -> (bool, str):
        """Install Ollama on the system."""
        return sys_install_ollama()

    def is_ollama_installed(self) -> bool:
        """Check if Ollama is installed."""
        return sys_is_ollama_installed()

    def start_ollama(self) -> (bool, str):
        """Start the Ollama server."""
        return sys_start_ollama()

    def stop_ollama(self) -> (bool, str):
        """Stop the Ollama server."""
        return sys_stop_ollama()

    async def initialize_defaults(self):
        """Pre-configure and pre-download models specified in settings."""
        for model_entry in settings.INITIAL_MODELS:
            try:
                if ":" in model_entry:
                    source, name = model_entry.split(":", 1)
                else:
                    source = settings.DEFAULT_MODEL_SOURCE
                    name = model_entry

                manager = ModelManagerFactory.get_manager(source)
                existing = manager.list_models()
                if not any(m['name'] == name for m in existing):
                    logger.info(f"Auto-downloading initial model: {name} from {source}")
                    await self.download_model(source, name)
            except Exception as e:
                logger.error(f"Failed to initialize model {model_entry}: {e}")
