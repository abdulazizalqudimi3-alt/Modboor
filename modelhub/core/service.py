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
import asyncio

logger = get_logger(__name__)

class ModelService:
    """
    Service Layer that orchestrates model operations across different managers.
    Implements the Singleton pattern.
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
        results = {}
        for source, manager in self.managers.items():
            results[source] = manager.list_models()
        return results

    async def find_model_source(self, model_name: str) -> Optional[str]:
        """Try to find which source a model belongs to."""
        for source, manager in self.managers.items():
            models = manager.list_models()
            if any(m['name'] == model_name for m in models):
                return source
        return None

    async def download_model(self, source: str, model_name: str, **kwargs) -> bool:
        manager = ModelManagerFactory.get_manager(source)
        # Run in thread pool as download is blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: manager.download_model(model_name, **kwargs))

    async def pull_model(self, source: str, model_name: str, **kwargs) -> bool:
        """Alias for download_model"""
        return await self.download_model(source, model_name, **kwargs)

    async def delete_model(self, source: str, model_name: str) -> bool:
        manager = ModelManagerFactory.get_manager(source)
        return manager.delete_model(model_name)

    async def remove_model(self, source: str, model_name: str) -> bool:
        """Alias for delete_model"""
        return await self.delete_model(source, model_name)

    async def generate_response(self, source: str, model_name: str, prompt: str, **kwargs) -> str:
        manager = ModelManagerFactory.get_manager(source)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: manager.generate(model_name, prompt, **kwargs))

    async def get_info(self, source: str, model_name: str) -> Optional[Dict[str, Any]]:
        manager = ModelManagerFactory.get_manager(source)
        return manager.get_model_info(model_name)

    async def load_model(self, source: str, model_name: str, **kwargs) -> bool:
        """Ensures a model is loaded into memory/ready for use."""
        manager = ModelManagerFactory.get_manager(source)
        if source == "ollama":
            return manager.run_model(model_name)
        elif source == "huggingface":
            # For HF, loading happens during first generate or explicitly via pipeline
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: manager.generate(model_name, "", **kwargs))
            return True
        return False

    async def unload_model(self, source: str, model_name: str) -> bool:
        """Unloads model to free resources."""
        manager = ModelManagerFactory.get_manager(source)
        if hasattr(manager, 'unload_model'):
            return manager.unload_model(model_name)
        elif source == "ollama":
            # Ollama unloads after keep_alive timeout, we can force it by setting keep_alive to 0
            # but current library might not have direct 'unload'
            return True
        return False

    async def get_status(self) -> Dict[str, Any]:
        return {
            "sources": {
                source: manager.is_available() for source, manager in self.managers.items()
            }
        }

    # Ollama direct management
    def install_ollama(self):
        return sys_install_ollama()

    def is_ollama_installed(self):
        return sys_is_ollama_installed()

    def start_ollama(self):
        return sys_start_ollama()

    def stop_ollama(self):
        return sys_stop_ollama()

    async def initialize_defaults(self):
        """Pre-downloads initial models if specified in settings."""
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
