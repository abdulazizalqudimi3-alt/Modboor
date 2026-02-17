import os
from pyngrok import ngrok
from dotenv import load_dotenv

load_dotenv()

def start_tunnel(port: int = 8000, auth_token: str = None):
    """
    Start an ngrok tunnel for the given port.
    """
    try:
        if auth_token:
            ngrok.set_auth_token(auth_token)
        elif os.getenv("NGROK_AUTHTOKEN"):
            ngrok.set_auth_token(os.getenv("NGROK_AUTHTOKEN"))

        public_url = ngrok.connect(port).public_url
        return public_url
    except Exception as e:
        print(f"Error starting ngrok tunnel: {e}")
        return None

def stop_tunnel():
    """
    Stop all ngrok tunnels.
    """
    ngrok.kill()
