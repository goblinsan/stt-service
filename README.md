# stt-service

GPU-accelerated speech-to-text service with a web UI, powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Architecture

```
stt-service/
├── apps/
│   ├── stt-api/      ← Python FastAPI backend (faster-whisper on GPU)
│   └── stt-ui/       ← React SPA (Vite)
├── infra/nginx/       ← Reverse proxy config
└── docker-compose.yml ← 3-service stack (API, UI, nginx)
```

## Features

- Drag-and-drop audio file transcription
- GPU-accelerated via NVIDIA CUDA (CTranslate2)
- Multiple Whisper model sizes (tiny → large-v3)
- Language auto-detection or manual selection
- Translate-to-English mode
- Word-level timestamps
- Segment-by-segment timeline view
- JSON export of full results

## Quick start (Docker)

```bash
docker compose up --build
```

Open `http://localhost:5101` — the UI is served through nginx with the API at `/api/`.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_MODEL_SIZE` | `large-v3` | Whisper model size |
| `STT_DEVICE` | `auto` | `cuda` or `cpu` |
| `STT_COMPUTE_TYPE` | `auto` | `float16`, `int8`, etc. |
| `STT_MODEL_DIR` | `./data/models` | Host path for cached models |
| `HOST_PORT` | `5101` | External port |

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check + model status |
| GET | `/api/info` | GPU info, model details |
| POST | `/api/transcribe` | Upload audio, get transcription |

## Control-plane integration

Deploy as a remote workload to a GPU node registered in `gateway-control-plane`.
