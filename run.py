import argparse
from modelhub.api.main import start_server
from modelhub.core.factory import ModelManagerFactory
from modelhub.utils.system import install_ollama, is_ollama_installed
from modelhub.utils.tunnel import start_tunnel
import time

def setup_and_run(
    initial_models: list = None,
    install_ollama_flag: bool = False,
    use_ngrok: bool = False,
    ngrok_token: str = None,
    port: int = 8000
):
    """
    Configure and start the AI Model Hub server.
    """
    if install_ollama_flag:
        if not is_ollama_installed():
            print("Installing Ollama as requested...")
            success, msg = install_ollama()
            print(msg)
        else:
            print("Ollama is already installed.")

    if initial_models:
        print(f"Ensuring initial models are downloaded: {initial_models}")
        for model_info in initial_models:
            source = model_info.get("source")
            name = model_info.get("name")
            if source and name:
                manager = ModelManagerFactory.get_manager(source)
                # Check if already downloaded
                existing = manager.list_models()
                if not any(m['name'] == name for m in existing):
                    print(f"Downloading {name} from {source}...")
                    manager.download_model(name)
                else:
                    print(f"{name} is already available.")

    public_url = None
    if use_ngrok:
        print("Starting ngrok tunnel...")
        public_url = start_tunnel(port, ngrok_token)
        if public_url:
            print(f"Public URL: {public_url}")
        else:
            print("Failed to start ngrok tunnel.")

    print(f"Starting server on port {port}...")
    start_server(port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Model Hub")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--install-ollama", action="store_true", help="Install Ollama on startup")
    parser.add_argument("--ngrok", action="store_true", help="Use ngrok for public access")
    parser.add_argument("--ngrok-token", type=str, help="ngrok auth token")
    parser.add_argument("--models", type=str, help="Initial models to download (format: source:name,source:name)")

    args = parser.parse_args()

    initial_models = []
    if args.models:
        for m in args.models.split(","):
            if ":" in m:
                s, n = m.split(":", 1)
                initial_models.append({"source": s, "name": n})

    setup_and_run(
        initial_models=initial_models,
        install_ollama_flag=args.install_ollama,
        use_ngrok=args.ngrok,
        ngrok_token=args.ngrok_token,
        port=args.port
    )
