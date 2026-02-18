# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environmental variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama (optional, usually better to run in a separate container)
# RUN curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/bin/ollama && chmod +x /usr/bin/ollama

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project
COPY . .

# Create data directory
RUN mkdir -p data logs

# Expose port
EXPOSE 8000

# Run the server
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8000"]
