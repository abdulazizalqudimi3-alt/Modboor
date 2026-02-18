from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ModelDownloadRequest(BaseModel):
    source: str = Field(..., description="Model source: 'huggingface' or 'ollama'")
    model_name: str = Field(..., description="The name or ID of the model to download")
    kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional arguments for the download")

class InferenceRequest(BaseModel):
    source: str = Field(..., description="Model source: 'huggingface' or 'ollama'")
    model_name: str = Field(..., description="The name of the model to use")
    prompt: str = Field(..., description="The prompt for generation")
    kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Inference parameters")

class InferenceResponse(BaseModel):
    response: str
    model_name: str
    source: str

class ModelLoadRequest(BaseModel):
    source: str
    model_name: str
    kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict)

class StatusResponse(BaseModel):
    ollama_installed: bool
    sources: Dict[str, bool]
    public_url: Optional[str] = None

class MessageResponse(BaseModel):
    message: str
    success: bool = True
