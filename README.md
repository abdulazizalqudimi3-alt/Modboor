# AI Model Hub: Enterprise Model Management & Inference Framework

`modelhub` is a high-performance, production-ready Python library and FastAPI server designed for the seamless management, deployment, and orchestration of AI models from **Hugging Face** and **Ollama**.

---

## 🌟 Key Features

- **Unified Interface**: Single, consistent API for Hugging Face and Ollama.
- **Full Model Lifecycle**: Download, Load, Run Inference, Unload, and Delete.
- **Advanced Ollama Orchestration**: Automated installation and process management.
- **Production-Grade Architecture**: Built with SOLID principles, Factory patterns, and a clean Service layer.
- **Public Accessibility**: Integrated `pyngrok` for instant public URLs.
- **Docker Ready**: Includes `Dockerfile` and `docker-compose.yml` for containerized deployment.
- **Highly Extensible**: Easily add new model sources by implementing a single interface.
- **Comprehensive API**: Full RESTful control over models and system status.

---

## 🏗 Project Architecture

The project follows **Clean Architecture** principles:
- **`modelhub/core/`**: Core business logic and model source abstractions.
- **`modelhub/api/`**: REST API implementation using FastAPI.
- **`modelhub/config/`**: Centralized configuration and professional logging.
- **`modelhub/utils/`**: System-level utilities for platform integration.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/modelhub.git
cd modelhub

# Install as a package
pip install .
```

### Running with Docker

```bash
# Start with Docker Compose
docker-compose up -d
```

### Starting the Server (CLI)

```bash
python run.py --port 8000 --public --auto-install-ollama
```

---

## 💻 Library Usage

```python
from modelhub import start_server, ModelService, HuggingFaceManager, OllamaManager

# Programmatic server startup
start_server(
    port=8080,
    public=True,
    initial_models=["ollama:llama3", "hf:gpt2"]
)

# Using managers directly
hf = HuggingFaceManager()
hf.download_model("gpt2")
print(hf.list_downloaded_models())

ollama = OllamaManager()
ollama.pull_model("mistral")
print(ollama.list_models())
```

---

## 🌐 API Endpoints

### Model Management
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/models` | List all available models. |
| `GET` | `/models/{model_name}` | Get detailed model metadata. |
| `POST` | `/models/download` | Download a new model (Background). |
| `POST` | `/models/load` | Load a model into memory. |
| `POST` | `/models/unload` | Unload a model to free resources. |
| `POST` | `/inference` | Run inference on a model. |
| `DELETE` | `/models/{model_name}` | Delete a local model. |

### Ollama Management
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/ollama/status` | Check Ollama installation & server status. |
| `POST` | `/ollama/install` | Install Ollama on the host system. |
| `POST` | `/ollama/start` | Start the Ollama server process. |
| `POST` | `/ollama/stop` | Stop the Ollama server process. |
| `POST` | `/ollama/models/pull` | Pull a model to Ollama. |

---

## ⚙️ Configuration

You can configure the system using environment variables or a `.env` file:

- `API_PORT`: Server port (default: 8000).
- `HF_CACHE_DIR`: Path to Hugging Face cache.
- `OLLAMA_HOST`: Host URL for Ollama.
- `NGROK_AUTHTOKEN`: ngrok token for public tunneling.
- `INITIAL_MODELS`: Comma-separated list of models to pre-load.

---

## 🧪 Testing

The project includes a comprehensive test suite using `pytest`.

```bash
pytest tests/
```

---

## 🛡 License

MIT License. See [LICENSE](LICENSE) for details.
