import os
import gc
from typing import List, Dict, Any, Optional
from huggingface_hub import snapshot_download, scan_cache_dir, model_info
from modelhub.core.base import BaseModelManager
from modelhub.config.settings import settings
from modelhub.config.logging_config import get_logger

logger = get_logger(__name__)

class HuggingFaceManager(BaseModelManager):
    """
    Manager for Hugging Face models using huggingface_hub and transformers.
    """

    _model_cache: Dict[str, Any] = {}

    def list_models(self) -> List[Dict[str, Any]]:
        """Alias for list_downloaded_models to satisfy interface."""
        return self.list_downloaded_models()

    def list_downloaded_models(self) -> List[Dict[str, Any]]:
        try:
            cache_info = scan_cache_dir(cache_dir=settings.HF_CACHE_DIR)
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
            logger.error(f"Error listing HF models: {e}")
            return []

    def download_model(self, model_id: str, **kwargs) -> bool:
        try:
            logger.info(f"Downloading HF model: {model_id}...")
            if "cache_dir" not in kwargs and settings.HF_CACHE_DIR:
                kwargs["cache_dir"] = settings.HF_CACHE_DIR
            snapshot_download(repo_id=model_id, **kwargs)
            return True
        except Exception as e:
            logger.error(f"Error downloading HF model {model_id}: {e}")
            return False

    def delete_model(self, model_id: str) -> bool:
        try:
            cache_info = scan_cache_dir(cache_dir=settings.HF_CACHE_DIR)
            for repo in cache_info.repos:
                if repo.repo_id == model_id:
                    delete_strategy = cache_info.delete_revisions(*[r.commit_hash for r in repo.revisions])
                    delete_strategy.execute()
                    self.unload_model(model_id)
                    return True
            return False
        except Exception as e:
            logger.error(f"Error deleting HF model {model_id}: {e}")
            return False

    def generate(self, model_id: str, prompt: str, **kwargs) -> str:
        try:
            from transformers import pipeline

            # Extract task if provided, otherwise default to text-generation
            task = kwargs.pop("task", "text-generation")
            cache_key = f"{model_id}:{task}"

            if cache_key not in self._model_cache:
                logger.info(f"Loading HF model {model_id} for task {task} into memory...")
                self._model_cache[cache_key] = pipeline(task, model=model_id, **kwargs)

            pipe = self._model_cache[cache_key]
            result = pipe(prompt)

            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            return str(result)
        except Exception as e:
            logger.error(f"Error during HF inference for {model_id}: {e}")
            return f"Error: {str(e)}"

    def is_available(self) -> bool:
        return True

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        try:
            info = model_info(model_id)
            return {
                "id": info.modelId,
                "author": info.author,
                "last_modified": str(info.lastModified),
                "tags": info.tags,
                "downloads": getattr(info, "downloads", 0),
                "likes": getattr(info, "likes", 0),
            }
        except Exception as e:
            logger.error(f"Error getting HF model info for {model_id}: {e}")
            return None

    def load_model(self, model_id: str, **kwargs) -> bool:
        try:
            from transformers import pipeline
            task = kwargs.get("task", "text-generation")
            cache_key = f"{model_id}:{task}"
            if cache_key not in self._model_cache:
                logger.info(f"Loading HF model {model_id} for task {task}...")
                self._model_cache[cache_key] = pipeline(task, model=model_id, **kwargs)
            return True
        except Exception as e:
            logger.error(f"Error loading HF model {model_id}: {e}")
            return False

    def unload_model(self, model_id: str) -> bool:
        keys_to_del = [k for k in self._model_cache if k.startswith(f"{model_id}:")]
        if keys_to_del:
            for k in keys_to_del:
                del self._model_cache[k]

            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded HF model {model_id} from memory.")
            return True
        return False
