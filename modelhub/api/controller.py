import uvicorn
import threading
import time
import sys
import os
from modelhub.api.main import app
from modelhub.config.settings import settings
from modelhub.config.logging_config import get_logger, setup_logging
from modelhub.utils.tunnel import start_tunnel, stop_tunnel
from modelhub.utils.system import is_ollama_installed, install_ollama

logger = get_logger(__name__)

class ServerController:
    """
    Controller for managing the FastAPI server lifecycle.
    """

    @classmethod
    def start_server(
        cls,
        host: str = None,
        port: int = None,
        public: bool = False,
        auto_install_ollama: bool = False,
        initial_models: list = None
    ):
        """
        Starts the FastAPI server with given configurations.
        """
        setup_logging()

        # Update settings
        if host: settings.API_HOST = host
        if port: settings.API_PORT = port
        if public: settings.USE_NGROK = public
        if auto_install_ollama: settings.AUTO_INSTALL_OLLAMA = auto_install_ollama
        if initial_models: settings.INITIAL_MODELS = initial_models

        logger.info(f"Initializing {settings.API_TITLE}...")

        if settings.AUTO_INSTALL_OLLAMA and not is_ollama_installed():
            logger.info("Auto-installing Ollama...")
            install_ollama()

        if settings.USE_NGROK:
            start_tunnel(settings.API_PORT)

        logger.info(f"Starting server on {settings.API_HOST}:{settings.API_PORT}")
        uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)

    @classmethod
    def stop_server(cls):
        """
        Stops the server and cleanup.
        """
        logger.info("Stopping server...")
        stop_tunnel()
        # In a real scenario with uvicorn.run, you'd need to handle process exit
        sys.exit(0)

    @classmethod
    def restart_server(cls):
        """
        Restarts the server process.
        """
        logger.info("Restarting server...")
        stop_tunnel()
        os.execv(sys.executable, [sys.executable] + sys.argv)
