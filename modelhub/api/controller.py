import uvicorn
import os
import sys
from modelhub.api.main import app
from modelhub.config.settings import settings
from modelhub.config.logging_config import setup_logging, get_logger
from modelhub.utils.tunnel import start_tunnel, stop_tunnel

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
        setup_logging()

        # Update settings
        if host: settings.API_HOST = host
        if port: settings.API_PORT = port
        if public: settings.USE_NGROK = public
        if auto_install_ollama: settings.AUTO_INSTALL_OLLAMA = auto_install_ollama
        if initial_models: settings.INITIAL_MODELS = initial_models

        if settings.USE_NGROK:
            start_tunnel(settings.API_PORT)

        logger.info(f"Starting server on {settings.API_HOST}:{settings.API_PORT}")
        uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)

    @classmethod
    def stop_server(cls):
        logger.info("Stopping server...")
        stop_tunnel()
        sys.exit(0)

    @classmethod
    def restart_server(cls):
        logger.info("Restarting server...")
        stop_tunnel()
        os.execv(sys.executable, [sys.executable] + sys.argv)
