# AI Model Hub

A powerful and extensible Python library and FastAPI server to manage AI models from Hugging Face and Ollama.

## Features

- **Unified Interface**: Manage models from both Hugging Face and Ollama using a single API.
- **Model Management**: List, download, and delete models.
- **Inference**: Run inference on downloaded models.
- **System Integration**: Install Ollama directly via the API.
- **Public Access**: Integrated `pyngrok` support to expose the server to the internet.
- **Scalable Architecture**: Built with SOLID principles and Clean Code in mind.

## Project Structure

- `modelhub/core/`: Core logic and abstract base classes for model management.
- `modelhub/api/`: FastAPI server implementation and endpoints.
- `modelhub/utils/`: Utility functions for system tasks and tunneling.
- `run.py`: Entry point for starting the server with pre-launch configurations.

## Installation

```bash
pip install fastapi uvicorn ollama huggingface-hub pyngrok python-dotenv pydantic transformers torch
```

## Usage

### Starting the Server

You can start the server using `run.py`. It supports several flags for customization:

```bash
python run.py --port 8000 --install-ollama --models "ollama:llama3,huggingface:gpt2" --ngrok
```

- `--port`: Port to run the server on (default: 8000).
- `--install-ollama`: Automatically install Ollama if not present.
- `--models`: Comma-separated list of models to ensure are downloaded on startup.
- `--ngrok`: Expose the server via an ngrok public URL.
- `--ngrok-token`: Your ngrok authentication token.

### API Endpoints

- `GET /models/list`: Lists all available models from all sources.
- `POST /models/download`: Downloads a model.
  - Body: `{"source": "huggingface", "model_name": "gpt2"}`
- `POST /inference`: Runs a prompt through a model.
  - Body: `{"source": "ollama", "model_name": "llama3", "prompt": "Hello!"}`
- `DELETE /models/delete`: Deletes a model.
  - Query params: `source`, `model_name`
- `POST /system/install-ollama`: Triggers Ollama installation.
- `GET /server/status`: Checks the status of the server and model managers.

## Development

Run tests using pytest:

```bash
pytest tests/
```

## Architecture

The project follows the Strategy and Factory design patterns. `BaseModelManager` defines the interface, and concrete implementations (`HuggingFaceManager`, `OllamaManager`) handle source-specific logic. The `ModelManagerFactory` provides a centralized way to access these managers, ensuring high cohesion and low coupling.
