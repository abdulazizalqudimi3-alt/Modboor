import pytest
from unittest.mock import MagicMock, patch
from modelhub.core.factory import ModelManagerFactory
from modelhub.core.huggingface import HuggingFaceManager
from modelhub.core.ollama import OllamaManager

def test_factory():
    hf = ModelManagerFactory.get_manager("huggingface")
    assert isinstance(hf, HuggingFaceManager)

    ollama_mgr = ModelManagerFactory.get_manager("ollama")
    assert isinstance(ollama_mgr, OllamaManager)

@patch("modelhub.core.huggingface.scan_cache_dir")
def test_hf_list_models(mock_scan):
    # Mocking scan_cache_dir return
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
