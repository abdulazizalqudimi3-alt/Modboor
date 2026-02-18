import subprocess
import sys
import os
import platform
import time
from modelhub.config.logging_config import get_logger

logger = get_logger(__name__)

def is_ollama_installed() -> bool:
    """Check if Ollama is installed and available in the PATH."""
    try:
        subprocess.run(["ollama", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def install_ollama() -> (bool, str):
    """Attempt to install Ollama based on the current platform."""
    system = platform.system()
    try:
        if system == "Linux":
            logger.info("Installing Ollama for Linux...")
            process = subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, capture_output=True, text=True)
            if process.returncode == 0:
                return True, "Ollama installed successfully."
            return False, f"Ollama installation failed: {process.stderr}"

        elif system == "Darwin": # macOS
            return False, "Ollama for macOS installation via script is not supported. Please download from https://ollama.com/download"

        elif system == "Windows":
            return False, "Ollama for Windows installation via script is not supported. Please download from https://ollama.com/download"

        else:
            return False, f"Unsupported platform: {system}"
    except Exception as e:
        logger.error(f"Error during Ollama installation: {e}")
        return False, str(e)

def start_ollama() -> (bool, str):
    """Start the Ollama server process."""
    if not is_ollama_installed():
        return False, "Ollama is not installed."

    try:
        # Check if already running
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', 11434)) == 0:
                return True, "Ollama is already running."

        logger.info("Starting Ollama server...")
        if platform.system() == "Windows":
            subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            # On Linux/macOS, we can run it in background
            with open(os.devnull, 'w') as devnull:
                subprocess.Popen(["ollama", "serve"], stdout=devnull, stderr=devnull)

        # Wait for it to start
        max_retries = 10
        for i in range(max_retries):
            time.sleep(1)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', 11434)) == 0:
                    return True, "Ollama started successfully."

        return False, "Ollama failed to start within timeout."
    except Exception as e:
        logger.error(f"Error starting Ollama: {e}")
        return False, str(e)

def stop_ollama() -> (bool, str):
    """Stop the Ollama server process."""
    try:
        system = platform.system()
        if system == "Windows":
            subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True)
        else:
            subprocess.run(["pkill", "ollama"], capture_output=True)
        return True, "Ollama stopped successfully."
    except Exception as e:
        logger.error(f"Error stopping Ollama: {e}")
        return False, str(e)
