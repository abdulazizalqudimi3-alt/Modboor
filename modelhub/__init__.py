from modelhub.api.controller import ServerController
from modelhub.core.service import ModelService
from modelhub.managers.huggingface import HuggingFaceManager
from modelhub.managers.ollama import OllamaManager
from modelhub.utils.system import install_ollama, is_ollama_installed, start_ollama, stop_ollama

# Singleton service for easy access
service = ModelService()

# Library control functions
start_server = ServerController.start_server
stop_server = ServerController.stop_server
restart_server = ServerController.restart_server

__all__ = [
    "start_server",
    "stop_server",
    "restart_server",
    "ModelService",
    "HuggingFaceManager",
    "OllamaManager",
    "install_ollama",
    "is_ollama_installed",
    "start_ollama",
    "stop_ollama",
    "service"
]
