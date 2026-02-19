# AI Model Hub: Enterprise-Grade Model Management

`modelhub` is a professional Python library and FastAPI server designed for managing, deploying, and orchestrating AI models from **Hugging Face** and **Ollama**.

## 🌟 Key Features

- **Unified Interface**: Single API for both Hugging Face and Ollama.
- **Production-Grade**: Built with Clean Architecture, SOLID principles, and professional logging.
- **Full Model Lifecycle**: Download, Load, Run Inference, Unload, and Delete.
- **Ollama Orchestration**: Automated installation, server management, and model pulling.
- **Public Tunneling**: Built-in support for `pyngrok` for instant public URLs.
- **Docker Ready**: Includes `Dockerfile` and `docker-compose.yml`.

## 🏗 Project Architecture

- **`modelhub/core/`**: Interface definitions and Factory pattern.
- **`modelhub/managers/`**: Specialized managers for HF and Ollama.
- **`modelhub/api/`**: FastAPI app, schemas, and controllers.
- **`modelhub/utils/`**: System utilities and tunneling.
- **`modelhub/config/`**: Centralized configuration and logging.

## 🚀 Quick Start

### Installation

```bash
pip install .
```

### Starting the Server (CLI)

```bash
python run.py --port 8000 --public --auto-install-ollama --models "ollama:llama3,hf:gpt2"
```

## 💻 Library Usage

```python
from modelhub import start_server, ModelService

# Start server
start_server(port=8000, public=True)

# Use Service Layer directly
service = ModelService()
models = await service.list_all_models()
```

## 🌐 API Endpoints

### Model Management
- `GET /models`: List all available models.
- `GET /models/{model_name}`: Get detailed info.
- `POST /models/download`: Download a new model (Background).
- `POST /models/load`: Load model into memory.
- `POST /models/unload`: Unload model from memory.
- `POST /inference`: Run generation.

### Ollama Management
- `GET /ollama/status`: Check status.
- `POST /ollama/install`: Trigger installation.
- `POST /ollama/start`: Start server.
- `POST /ollama/stop`: Stop server.
- `POST /ollama/models/pull`: Pull model to Ollama.

## 🧪 Testing

```bash
pytest tests/
```

## 🛡 License

MIT
