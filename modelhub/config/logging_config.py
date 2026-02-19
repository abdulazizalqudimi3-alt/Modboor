import logging
import sys
from pathlib import Path

def setup_logging(level=logging.INFO):
    """
    Sets up a professional logging configuration.
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "modelhub.log")
        ]
    )

    # Set levels for noisy libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("ollama").setLevel(logging.WARNING)

def get_logger(name: str):
    return logging.getLogger(name)
