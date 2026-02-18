from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List
import os
from pathlib import Path

class Settings(BaseSettings):
    # API Settings
    API_TITLE: str = "AI Model Hub Production API"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False

    # Model Settings
    HF_CACHE_DIR: Optional[str] = str(Path.home() / ".cache" / "huggingface" / "hub")
    OLLAMA_HOST: str = "http://localhost:11434"
    DEFAULT_MODEL_SOURCE: str = "huggingface"

    # Auto-Initialization
    AUTO_INSTALL_OLLAMA: bool = False
    INITIAL_MODELS: List[str] = [] # Format: "source:model_name"

    # Ngrok Settings
    NGROK_AUTHTOKEN: Optional[str] = None
    USE_NGROK: bool = False

    # Storage
    DATA_DIR: str = "data"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.makedirs(self.DATA_DIR, exist_ok=True)

settings = Settings()
