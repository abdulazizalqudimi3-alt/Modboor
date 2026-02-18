from setuptools import setup, find_packages

setup(
    name="modelhub",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "ollama",
        "huggingface-hub",
        "pyngrok",
        "python-dotenv",
        "pydantic",
        "pydantic-settings",
        "transformers",
        "torch",
        "httpx"
    ],
    entry_points={
        "console_scripts": [
            "modelhub-run=run:main",
        ],
    },
)
