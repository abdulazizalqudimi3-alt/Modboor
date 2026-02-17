from fastapi.testclient import TestClient
from modelhub.api.main import app
from unittest.mock import patch, MagicMock

client = TestClient(app)

def test_list_models_endpoint():
    with patch("modelhub.core.huggingface.HuggingFaceManager.list_models", return_value=[]), \
         patch("modelhub.core.ollama.OllamaManager.list_models", return_value=[]):
        response = client.get("/models/list")
        assert response.status_code == 200
        assert "huggingface" in response.json()
        assert "ollama" in response.json()

def test_server_status_endpoint():
    with patch("modelhub.api.main.is_ollama_installed", return_value=True):
        response = client.get("/server/status")
        assert response.status_code == 200
        assert response.json()["ollama_installed"] == True
