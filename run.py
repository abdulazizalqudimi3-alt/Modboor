import argparse
import sys
from modelhub import start_server

def main():
    parser = argparse.ArgumentParser(description="AI Model Hub Production Runner")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("--public", action="store_true", help="Enable public ngrok tunnel")
    parser.add_argument("--auto-install-ollama", action="store_true", help="Auto-install Ollama if missing")
    parser.add_argument("--models", type=str, help="Initial models to pre-download (source:name,source:name)")

    args = parser.parse_args()

    initial_models = []
    if args.models:
        initial_models = args.models.split(",")

    try:
        start_server(
            host=args.host,
            port=args.port,
            public=args.public,
            auto_install_ollama=args.auto_install_ollama,
            initial_models=initial_models
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()
