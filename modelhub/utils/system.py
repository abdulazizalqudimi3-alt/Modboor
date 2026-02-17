import subprocess
import sys
import os

def install_ollama():
    """
    Attempt to install Ollama on the system.
    Focus on Linux as per sandbox environment.
    """
    try:
        if sys.platform.startswith("linux"):
            print("Detected Linux. Installing Ollama...")
            # Using the official install script
            process = subprocess.Popen(
                "curl -fsSL https://ollama.com/install.sh | sh",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                return True, "Ollama installed successfully."
            else:
                return False, f"Ollama installation failed: {stderr}"
        elif sys.platform == "darwin":
            return False, "MacOS installation via script is not directly supported here. Please download from ollama.com"
        elif sys.platform == "win32":
            return False, "Windows installation via script is not directly supported here. Please download from ollama.com"
        else:
            return False, f"Unsupported platform: {sys.platform}"
    except Exception as e:
        return False, f"An error occurred during installation: {str(e)}"

def is_ollama_installed():
    """
    Check if the ollama binary is in the PATH.
    """
    try:
        subprocess.run(["ollama", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False
