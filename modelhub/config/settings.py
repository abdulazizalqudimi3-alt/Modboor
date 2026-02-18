from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    API_TITLE: str = "AI Model Hub API"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Model Settings
    HF_CACHE_DIR: Optional[str] = None
    OLLAMA_HOST: str = "http://localhost:11434"

    # Ngrok Settings
    NGROK_AUTHTOKEN: Optional[str] = None
    USE_NGROK: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
