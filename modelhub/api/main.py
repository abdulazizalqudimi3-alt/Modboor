from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import time
import asyncio

from modelhub.api.schemas import (
    ModelDownloadRequest, InferenceRequest, InferenceResponse,
    ModelLoadRequest, StatusResponse, MessageResponse, OllamaPullRequest
)
from modelhub.core.service import ModelService
from modelhub.config.settings import settings
from modelhub.config.logging_config import setup_logging, get_logger
from modelhub.utils.tunnel import get_public_url

logger = get_logger(__name__)

# Singleton Service
model_service = ModelService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_logging()
    logger.info("Starting AI Model Hub API...")
    # Initialize defaults in background
    asyncio.create_task(model_service.initialize_defaults())
    yield
    # Shutdown
    logger.info("Shutting down AI Model Hub API...")

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Production-grade AI Model Management Hub",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Management ---

@app.get("/models", response_model=Dict[str, List[Dict[str, Any]]])
async def list_models():
    """List all models from all available sources."""
    return await model_service.list_all_models()

@app.get("/models/{model_name:path}")
async def get_model_info(model_name: str, source: Optional[str] = None):
    """Get detailed information about a specific model."""
    if not source and ":" in model_name:
        parts = model_name.split(":", 1)
        if parts[0].lower() in ["huggingface", "hf", "ollama"]:
            source = parts[0].lower()
            model_name = parts[1]

    if not source:
        source = await model_service.find_model_source(model_name)

    if not source:
        raise HTTPException(status_code=404, detail="Model source not found")

    info = await model_service.get_info(source, model_name)
    if not info:
        raise HTTPException(status_code=404, detail="Model info not found")
    return info

@app.post("/models/download", response_model=MessageResponse)
async def download_model(request: ModelDownloadRequest, background_tasks: BackgroundTasks):
    """Download a model from the specified source."""
    background_tasks.add_task(model_service.download_model, request.source, request.model_name, **request.kwargs)
    return {"message": f"Download of {request.model_name} from {request.source} started in background.", "success": True}

@app.delete("/models/{model_name:path}", response_model=MessageResponse)
async def delete_model(model_name: str, source: Optional[str] = None):
    """Delete a model from local storage."""
    if not source and ":" in model_name:
        parts = model_name.split(":", 1)
        source = parts[0].lower()
        model_name = parts[1]

    if not source:
        source = await model_service.find_model_source(model_name)

    if not source:
        raise HTTPException(status_code=404, detail="Model not found to delete")

    success = await model_service.delete_model(source, model_name)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to delete model")
    return {"message": f"Model {model_name} deleted successfully.", "success": True}

@app.post("/models/load", response_model=MessageResponse)
async def load_model(request: ModelLoadRequest):
    """Load a model into memory."""
    source = request.source
    model_name = request.model_name
    if not source and ":" in model_name:
        parts = model_name.split(":", 1)
        source = parts[0].lower()
        model_name = parts[1]

    if not source:
        source = await model_service.find_model_source(model_name)

    if not source:
        raise HTTPException(status_code=404, detail="Model source not found to load")

    success = await model_service.load_model(source, model_name, **request.kwargs)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to load model")
    return {"message": f"Model {model_name} loaded successfully.", "success": True}

@app.post("/models/unload", response_model=MessageResponse)
async def unload_model(model_name: str, source: Optional[str] = None):
    """Unload a model from memory."""
    if not source and ":" in model_name:
        parts = model_name.split(":", 1)
        source = parts[0].lower()
        model_name = parts[1]

    if not source:
        source = await model_service.find_model_source(model_name)

    if not source:
        raise HTTPException(status_code=404, detail="Model source not found to unload")

    success = await model_service.unload_model(source, model_name)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to unload model")
    return {"message": f"Model {model_name} unloaded successfully.", "success": True}

@app.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """Run a prompt through a model."""
    response = await model_service.generate_response(
        request.source, request.model_name, request.prompt, **request.kwargs
    )
    return {
        "response": response,
        "model_name": request.model_name,
        "source": request.source
    }

# --- Ollama Management ---

@app.get("/ollama/status")
async def ollama_status():
    """Get Ollama installation and server status."""
    return {
        "installed": model_service.is_ollama_installed(),
        "running": ModelManagerFactory.get_manager('ollama').is_available()
    }

@app.post("/ollama/install", response_model=MessageResponse)
def trigger_ollama_install():
    """Install Ollama on the system."""
    success, message = model_service.install_ollama()
    return {"message": message, "success": success}

@app.post("/ollama/start", response_model=MessageResponse)
def trigger_ollama_start():
    """Start the Ollama server."""
    success, message = model_service.start_ollama()
    return {"message": message, "success": success}

@app.post("/ollama/stop", response_model=MessageResponse)
def trigger_ollama_stop():
    """Stop the Ollama server."""
    success, message = model_service.stop_ollama()
    return {"message": message, "success": success}

@app.get("/ollama/models")
async def list_ollama_models():
    """List available models in Ollama."""
    return ModelManagerFactory.get_manager('ollama').list_models()

@app.post("/ollama/models/pull", response_model=MessageResponse)
async def pull_ollama_model(request: OllamaPullRequest, background_tasks: BackgroundTasks):
    """Pull a model to Ollama in background."""
    background_tasks.add_task(ModelManagerFactory.get_manager('ollama').download_model, request.model_name)
    return {"message": f"Pulling {request.model_name} started.", "success": True}

@app.delete("/ollama/models/{model_name}", response_model=MessageResponse)
async def remove_ollama_model(model_name: str):
    """Remove a model from Ollama."""
    success = ModelManagerFactory.get_manager('ollama').delete_model(model_name)
    return {"message": f"Model {model_name} removed.", "success": success}

# --- Server Management ---

@app.get("/server/status", response_model=StatusResponse)
async def server_status():
    """Get general server status."""
    service_status = await model_service.get_status()
    return {
        "ollama_installed": model_service.is_ollama_installed(),
        "sources": service_status["sources"],
        "public_url": get_public_url()
    }

@app.post("/server/refresh", response_model=MessageResponse)
async def server_refresh():
    """Refresh managers and states."""
    return {"message": "Server state refreshed.", "success": True}

@app.post("/server/restart", response_model=MessageResponse)
async def server_restart(background_tasks: BackgroundTasks):
    """Restart the server process."""
    def restart():
        import os
        import sys
        time.sleep(1)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    background_tasks.add_task(restart)
    return {"message": "Server restarting...", "success": True}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}
