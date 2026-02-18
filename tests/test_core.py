import pytest
from unittest.mock import MagicMock, patch
from modelhub.core.service import ModelService
from modelhub.core.huggingface import HuggingFaceManager
from modelhub.core.ollama import OllamaManager

@pytest.fixture
def model_service():
    return ModelService()

@patch("modelhub.core.huggingface.scan_cache_dir")
def test_hf_list_models(mock_scan):
    mock_repo = MagicMock()
    mock_repo.repo_id = "test/model"
    mock_repo.repo_type = "model"
    mock_repo.size_on_disk = 100
    mock_repo.nb_files = 1
    mock_repo.last_modified = "today"

    mock_cache = MagicMock()
    mock_cache.repos = [mock_repo]
    mock_scan.return_value = mock_cache

    mgr = HuggingFaceManager()
    models = mgr.list_models()
    assert len(models) == 1
    assert models[0]["name"] == "test/model"

@patch("ollama.list")
def test_ollama_list_models(mock_list):
    mock_model = MagicMock()
    mock_model.model = "llama2"
    mock_model.size = 1000
    mock_model.modified_at = "today"

    mock_response = MagicMock()
    mock_response.models = [mock_model]
    mock_list.return_value = mock_response

    mgr = OllamaManager()
    models = mgr.list_models()
    assert len(models) == 1
    assert models[0]["name"] == "llama2"

@pytest.mark.asyncio
async def test_service_list_models(model_service):
    with patch("modelhub.core.huggingface.HuggingFaceManager.list_models", return_value=[{"name": "hf_model"}]), \
         patch("modelhub.core.ollama.OllamaManager.list_models", return_value=[{"name": "ollama_model"}]):
        results = await model_service.list_all_models()
        assert "huggingface" in results
        assert "ollama" in results
        assert results["huggingface"][0]["name"] == "hf_model"
        assert results["ollama"][0]["name"] == "ollama_model"
