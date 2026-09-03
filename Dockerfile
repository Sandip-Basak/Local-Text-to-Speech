# =============================================================================
# Kokoro-82M Real-Time Streaming TTS Dockerfile
# Optimized for NVIDIA CUDA GPU acceleration with automatic CPU fallback
# =============================================================================

FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    HOST=0.0.0.0 \
    PORT=8001 \
    DEVICE=auto \
    DEFAULT_VOICE=af_heart

# Install system dependencies
# - espeak-ng: Required phonemizer for Kokoro / misaki
# - libsndfile1: Required for soundfile audio encoding
# - libportaudio2: PortAudio library for audio streaming support
# - ffmpeg: Audio processing and transcoding utilities
# - curl: Container health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libsndfile1 \
    libportaudio2 \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 runtime support (automatically falls back to CPU if no GPU present)
# and install required Python packages
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt

# Pre-download and cache Kokoro-82M model weights into image
# This ensures zero cold-start delay and allows the container to run completely offline
RUN python3 -c "from kokoro import KPipeline; KPipeline(lang_code='a', device='cpu')"

# Copy application source code
COPY TTS.py .
COPY api.py .
COPY client_example.py .
COPY README.md .

# Expose HTTP / WebSocket API port
EXPOSE 8001

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Start FastAPI WebSocket streaming server with Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
