from pyngrok import ngrok
from typing import Optional
from modelhub.config.settings import settings
from modelhub.config.logging_config import get_logger

logger = get_logger(__name__)

_public_url = None

def start_tunnel(port: int = None, auth_token: str = None) -> Optional[str]:
    """
    Start an ngrok tunnel for the given port.
    """
    global _public_url
    try:
        port = port or settings.API_PORT
        token = auth_token or settings.NGROK_AUTHTOKEN

        if token:
            ngrok.set_auth_token(token)

        logger.info(f"Starting ngrok tunnel on port {port}...")
        _public_url = ngrok.connect(port).public_url
        logger.info(f"Public URL: {_public_url}")
        return _public_url
    except Exception as e:
        logger.error(f"Error starting ngrok tunnel: {e}")
        return None

def stop_tunnel():
    """
    Stop all ngrok tunnels.
    """
    global _public_url
    try:
        logger.info("Stopping ngrok tunnels...")
        ngrok.kill()
        _public_url = None
    except Exception as e:
        logger.error(f"Error stopping ngrok tunnel: {e}")

def get_public_url() -> Optional[str]:
    """
    Get the current public URL.
    """
    return _public_url
