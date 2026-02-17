from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from modelhub.core.factory import ModelManagerFactory
from modelhub.utils.system import install_ollama, is_ollama_installed
import uvicorn

app = FastAPI(title="AI Model Hub API")

class DownloadRequest(BaseModel):
    source: str # 'huggingface' or 'ollama'
    model_name: str
    kwargs: Optional[Dict[str, Any]] = {}

class InferenceRequest(BaseModel):
    source: str
    model_name: str
    prompt: str
    kwargs: Optional[Dict[str, Any]] = {}

@app.get("/models/list")
async def list_models():
    """List all models from all managers."""
    results = {}
    managers = ModelManagerFactory.get_all_managers()
    for source, manager in managers.items():
        results[source] = manager.list_models()
    return results

@app.post("/models/download")
async def download_model(request: DownloadRequest, background_tasks: BackgroundTasks):
    """Download a model in the background."""
    try:
        manager = ModelManagerFactory.get_manager(request.source)
        # For simplicity, we'll start it in background
        background_tasks.add_task(manager.download_model, request.model_name, **request.kwargs)
        return {"message": f"Download of {request.model_name} from {request.source} started in background."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/models/delete")
async def delete_model(source: str, model_name: str):
    """Delete a model."""
    try:
        manager = ModelManagerFactory.get_manager(source)
        success = manager.delete_model(model_name)
        if success:
            return {"message": f"Model {model_name} deleted from {source}."}
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found or could not be deleted.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/inference")
def inference(request: InferenceRequest):
    """
    Run inference on a model.
    Note: defined as 'def' instead of 'async def' so FastAPI runs it in a threadpool,
    preventing blocking the main event loop during heavy computation.
    """
    try:
        manager = ModelManagerFactory.get_manager(request.source)
        response = manager.generate(request.model_name, request.prompt, **request.kwargs)
        return {"response": response}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/system/install-ollama")
async def trigger_ollama_install():
    """Trigger Ollama installation on the system."""
    if is_ollama_installed():
        return {"message": "Ollama is already installed."}

    success, message = install_ollama()
    if success:
        return {"message": message}
    else:
        raise HTTPException(status_code=500, detail=message)

@app.get("/server/status")
async def server_status():
    """Get server and managers status."""
    managers = ModelManagerFactory.get_all_managers()
    status = {
        "ollama_installed": is_ollama_installed(),
        "managers": {
            source: manager.is_available() for source, manager in managers.items()
        }
    }
    return status

def start_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)
