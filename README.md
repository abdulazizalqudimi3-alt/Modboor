# AI Model Hub: Enterprise-Grade Model Management

`modelhub` is a professional, production-ready Python library and FastAPI server designed for managing, deploying, and orchestrating AI models from multiple sources, specifically **Hugging Face** and **Ollama**.

## 🚀 Key Features

- **Unified Interface**: Single API for managing both Hugging Face and Ollama models.
- **Production-Grade**: Built with Clean Architecture, SOLID principles, and professional logging.
- **Full Model Lifecycle**: Download, Load, Run Inference, Unload, and Delete models.
- **Ollama Orchestration**: Automated installation, server management (start/stop), and model pulling.
- **Public Tunneling**: Built-in support for `pyngrok` to expose your server publicly.
- **Asynchronous Core**: Fully async service layer for high-performance operations.
- **Auto-Initialization**: Pre-configure models to download and install on startup.
- **Cross-Platform**: Designed for Linux, macOS, and Windows.

## 🛠 Project Structure

- `modelhub/core/`: Business logic, managers, and service layer.
- `modelhub/api/`: FastAPI server, schemas, and controllers.
- `modelhub/config/`: Centralized settings and logging configuration.
- `modelhub/utils/`: System utilities and tunneling support.
- `run.py`: Production entry point.
- `setup.py`: Package installation configuration.

## 📦 Installation

```bash
pip install .
```

Dependencies: `fastapi`, `uvicorn`, `ollama`, `huggingface-hub`, `pyngrok`, `pydantic-settings`, `transformers`, `torch`.

## 💻 Usage as a Library

```python
from modelhub import start_server, ModelService, HuggingFaceManager, OllamaManager

# Start the server programmatically
start_server(
    port=8000,
    public=True,
    auto_install_ollama=True,
    initial_models=["ollama:llama3", "hf:gpt2"]
)

# Use HuggingFaceManager directly
hf = HuggingFaceManager()
hf.download_model("gpt2")
models = hf.list_downloaded_models()

# Use OllamaManager directly
ollama_mgr = OllamaManager()
ollama_mgr.pull_model("llama3")
models = ollama_mgr.list_models()
```

## 🌐 API Reference

### Model Management
- `GET /models`: List all available models.
- `GET /models/{source}/{model_name}`: Get detailed model info.
- `POST /models/download`: Download a new model (Background Task).
- `POST /models/load`: Load a model into memory.
- `POST /models/unload`: Unload model to free resources.
- `POST /inference`: Run AI generation.

### Ollama Management
- `GET /ollama/status`: Check installation and server status.
- `POST /ollama/install`: Install Ollama on the host.
- `POST /ollama/start`: Start the Ollama process.
- `POST /ollama/stop`: Stop the Ollama process.
- `GET /ollama/models`: List Ollama-specific models.

### Server Control
- `GET /server/status`: Overall system health and public URL.
- `POST /server/restart`: Restart the API server.
- `GET /health`: Basic health check.

## ⚙️ Configuration

Use a `.env` file or environment variables:

```env
API_PORT=8000
USE_NGROK=True
NGROK_AUTHTOKEN=your_token
HF_CACHE_DIR=/path/to/custom/cache
AUTO_INSTALL_OLLAMA=True
INITIAL_MODELS=ollama:llama3,huggingface:gpt2
```

## 🧪 Testing

```bash
pytest tests/
```

## 🛡 License

MIT
