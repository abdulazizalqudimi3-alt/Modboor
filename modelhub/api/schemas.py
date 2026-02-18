from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ModelDownloadRequest(BaseModel):
    """Request schema for downloading a model."""
    source: str = Field(..., description="Model source: 'huggingface' or 'ollama'")
    model_name: str = Field(..., description="The name or ID of the model to download")
    kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional arguments for the download")

class InferenceRequest(BaseModel):
    """Request schema for running inference."""
    source: str = Field(..., description="Model source: 'huggingface' or 'ollama'")
    model_name: str = Field(..., description="The name of the model to use")
    prompt: str = Field(..., description="The prompt for generation")
    kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Inference parameters")

class InferenceResponse(BaseModel):
    """Response schema for inference results."""
    response: str
    model_name: str
    source: str

class ModelLoadRequest(BaseModel):
    """Request schema for loading a model into memory."""
    source: Optional[str] = None
    model_name: str
    kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict)

class StatusResponse(BaseModel):
    """Response schema for server status."""
    ollama_installed: bool
    sources: Dict[str, bool]
    public_url: Optional[str] = None

class MessageResponse(BaseModel):
    """Generic message response schema."""
    message: str
    success: bool = True

class OllamaPullRequest(BaseModel):
    """Request schema specifically for pulling Ollama models."""
    model_name: str
