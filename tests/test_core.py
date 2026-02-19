import pytest
from unittest.mock import MagicMock, patch
from modelhub.core.service import ModelService
from modelhub.managers.huggingface import HuggingFaceManager
from modelhub.managers.ollama import OllamaManager

@pytest.fixture
def model_service():
    return ModelService()

@patch("modelhub.managers.huggingface.scan_cache_dir")
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

@pytest.mark.asyncio
async def test_service_list_models(model_service):
    with patch("modelhub.managers.huggingface.HuggingFaceManager.list_models", return_value=[{"name": "hf_model"}]), \
         patch("modelhub.managers.ollama.OllamaManager.list_models", return_value=[{"name": "ollama_model"}]):
        results = await model_service.list_all_models()
        assert "huggingface" in results
        assert "ollama" in results
