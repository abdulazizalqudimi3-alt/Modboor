from fastapi.testclient import TestClient
from modelhub.api.main import app
from unittest.mock import patch
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
