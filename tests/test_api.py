from fastapi.testclient import TestClient
from modelhub.api.main import app
from unittest.mock import patch, MagicMock
import pytest

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_list_models_endpoint():
    with patch("modelhub.core.service.ModelService.list_all_models", return_value={"hf": [], "ollama": []}):
        response = client.get("/models")
        assert response.status_code == 200
        assert "hf" in response.json()

@patch("modelhub.core.service.ModelService.download_model")
def test_download_model_endpoint(mock_download):
    response = client.post("/models/download", json={
        "source": "huggingface",
        "model_name": "gpt2"
    })
    assert response.status_code == 200
    assert "started" in response.json()["message"]

@patch("modelhub.core.service.ModelService.generate_response")
def test_inference_endpoint(mock_generate):
    mock_generate.return_value = "Hello"
    response = client.post("/inference", json={
        "source": "ollama",
        "model_name": "llama2",
        "prompt": "Hi"
    })
    assert response.status_code == 200
    assert response.json()["response"] == "Hello"

def test_server_status_endpoint():
    with patch("modelhub.api.main.is_ollama_installed", return_value=True), \
         patch("modelhub.core.service.ModelService.get_status", return_value={"sources": {"hf": True}}):
        response = client.get("/server/status")
        assert response.status_code == 200
        assert response.json()["ollama_installed"] == True
