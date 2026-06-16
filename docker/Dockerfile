FROM python:3.12-slim

# Install system dependencies for audio and computer vision
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY . .

# Create data and models directories
RUN mkdir -p /app/data /app/models

# Non-root user
RUN useradd -m -u 1000 agentuser && \
    chown -R agentuser:agentuser /app
USER agentuser

EXPOSE 8009

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8009/health', timeout=5)"

CMD ["python", "-m", "agent.main", "serve", "--host", "0.0.0.0", "--port", "8009"]
