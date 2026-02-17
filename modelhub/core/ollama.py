import ollama
from typing import List, Dict, Any, Optional
from .base import BaseModelManager

class OllamaManager(BaseModelManager):
    """
    Manager for Ollama models.
    """

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all downloaded Ollama models.
        """
        try:
            response = ollama.list()
            models = []
            # In recent ollama versions, it returns an object with a 'models' attribute
            # which is a list of model objects
            model_list = getattr(response, 'models', [])
            for model in model_list:
                models.append({
                    "name": model.get('name') if isinstance(model, dict) else getattr(model, 'model', ''),
                    "type": "ollama",
                    "size": model.get('size') if isinstance(model, dict) else getattr(model, 'size', 0),
                    "modified_at": model.get('modified_at') if isinstance(model, dict) else str(getattr(model, 'modified_at', '')),
                })
            return models
        except Exception as e:
            print(f"Error listing Ollama models: {e}")
            return []

    def download_model(self, model_name: str, **kwargs) -> bool:
        """
        Pull a model from Ollama library.
        """
        try:
            print(f"Pulling Ollama model: {model_name}...")
            ollama.pull(model_name)
            return True
        except Exception as e:
            print(f"Error pulling Ollama model {model_name}: {e}")
            return False

    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model from Ollama.
        """
        try:
            ollama.delete(model_name)
            return True
        except Exception as e:
            print(f"Error deleting Ollama model {model_name}: {e}")
            return False

    def generate(self, model_name: str, prompt: str, **kwargs) -> str:
        """
        Generate response using Ollama.
        """
        try:
            response = ollama.generate(model=model_name, prompt=prompt, **kwargs)
            return response.get('response', '')
        except Exception as e:
            return f"Error during Ollama inference: {str(e)}"

    def is_available(self) -> bool:
        """
        Check if Ollama server is running.
        """
        try:
            ollama.list()
            return True
        except Exception:
            return False
