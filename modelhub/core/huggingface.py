import os
from typing import List, Dict, Any, Optional
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download, scan_cache_dir
from .base import BaseModelManager

class HuggingFaceManager(BaseModelManager):
    """
    Manager for Hugging Face models.
    """

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all downloaded HF models from the local cache.
        """
        try:
            cache_info = scan_cache_dir()
            models = []
            for repo in cache_info.repos:
                if repo.repo_type == "model":
                    models.append({
                        "name": repo.repo_id,
                        "type": "huggingface",
                        "size": repo.size_on_disk,
                        "nb_files": repo.nb_files,
                        "last_modified": str(repo.last_modified),
                    })
            return models
        except Exception as e:
            print(f"Error listing HF models: {e}")
            return []

    def download_model(self, model_name: str, **kwargs) -> bool:
        """
        Download a model from Hugging Face Hub.
        """
        try:
            print(f"Downloading HF model: {model_name}...")
            snapshot_download(repo_id=model_name, **kwargs)
            return True
        except Exception as e:
            print(f"Error downloading HF model {model_name}: {e}")
            return False

    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model from the HF local cache.
        Note: This is a bit complex with scan_cache_dir.
        For simplicity, we'll try to find the revision and delete it.
        """
        try:
            cache_info = scan_cache_dir()
            for repo in cache_info.repos:
                if repo.repo_id == model_name:
                    delete_strategy = cache_info.delete_revisions(*[r.commit_hash for r in repo.revisions])
                    delete_strategy.execute()
                    return True
            return False
        except Exception as e:
            print(f"Error deleting HF model {model_name}: {e}")
            return False

    _model_cache: Dict[str, Any] = {}

    def generate(self, model_name: str, prompt: str, **kwargs) -> str:
        """
        Generate response using transformers pipeline.
        Caches the model pipeline for better performance.
        """
        try:
            from transformers import pipeline

            if model_name not in self._model_cache:
                print(f"Loading HF model {model_name} into memory...")
                # Basic text-generation pipeline
                # For real usage, you'd want to manage device (CPU/GPU)
                self._model_cache[model_name] = pipeline("text-generation", model=model_name, **kwargs)

            pipe = self._model_cache[model_name]
            result = pipe(prompt)

            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            return str(result)
        except Exception as e:
            return f"Error during HF inference: {str(e)}"

    def is_available(self) -> bool:
        # Hugging Face Hub is a library, so as long as it's installed and we have internet (for downloads), it's available.
        return True
