import asyncio
from typing import List, Dict, Any, Optional
from modelhub.core.factory import ModelManagerFactory
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
        for source, manager in self.managers.items():
            models = manager.list_models()
            if any(m['name'] == model_name for m in models):
                return source
        return None

    async def download_model(self, source: str, model_id: str, **kwargs) -> bool:
        manager = ModelManagerFactory.get_manager(source)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: manager.download_model(model_id, **kwargs))

    async def delete_model(self, source: str, model_id: str) -> bool:
        manager = ModelManagerFactory.get_manager(source)
        return manager.delete_model(model_id)

    async def generate_response(self, source: str, model_name: str, prompt: str, **kwargs) -> str:
        manager = ModelManagerFactory.get_manager(source)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: manager.generate(model_name, prompt, **kwargs))

    async def get_info(self, source: str, model_id: str) -> Optional[Dict[str, Any]]:
        manager = ModelManagerFactory.get_manager(source)
        return manager.get_model_info(model_id)

    async def load_model(self, source: str, model_id: str, **kwargs) -> bool:
        manager = ModelManagerFactory.get_manager(source)
        if source == "ollama":
            return manager.load_model(model_id)
        elif source == "huggingface":
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: manager.load_model(model_id, **kwargs))
        return False

    async def unload_model(self, source: str, model_id: str) -> bool:
        manager = ModelManagerFactory.get_manager(source)
        return manager.unload_model(model_id)

    async def get_status(self) -> Dict[str, Any]:
        return {
            "sources": {
                source: manager.is_available() for source, manager in self.managers.items()
            }
        }

    def install_ollama(self) -> (bool, str):
        return sys_install_ollama()

    def is_ollama_installed(self) -> bool:
        return sys_is_ollama_installed()

    def start_ollama(self) -> (bool, str):
        return sys_start_ollama()

    def stop_ollama(self) -> (bool, str):
        return sys_stop_ollama()

    async def initialize_defaults(self):
        """Pre-configure and pre-download models specified in settings."""
        if settings.AUTO_INSTALL_OLLAMA and not self.is_ollama_installed():
            logger.info("Auto-installing Ollama...")
            self.install_ollama()

        if self.is_ollama_installed():
            self.start_ollama()

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
