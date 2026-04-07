FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Use /data/models/huggingface as the HuggingFace Hub cache so pyannote
# models are persisted in the same mounted volume as Whisper models.
ENV HF_HOME=/data/models/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 12.4 support first (needed by pyannote.audio)
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ ./src/

EXPOSE 5100

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "5100"]
