# stt-service

GPU-accelerated speech-to-text service with a web UI, powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 backend).

## Architecture

```
stt-service/
├── apps/
│   ├── stt-api/      ← Python FastAPI backend (faster-whisper on GPU)
│   └── stt-ui/       ← React SPA (Vite)
├── infra/nginx/       ← Reverse proxy config
├── scripts/           ← Benchmark and utility scripts
└── docker-compose.yml ← 3-service stack (API, UI, nginx)
```

## Features

- Drag-and-drop audio file transcription
- GPU-accelerated via NVIDIA CUDA (CTranslate2)
- Multiple Whisper model sizes (tiny → large-v3)
- Language auto-detection or manual selection
- Translate-to-English mode
- Word-level timestamps
- Initial prompt field for domain hints / glossary biasing
- Segment-by-segment timeline view
- JSON export of full results
- Speaker diarization via [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- Browser tester shows live model / diarization readiness, GPU status, and speaker summary

## Quick start (Docker)

```bash
docker compose up --build
```

Open `http://localhost:5101` — the UI is served through nginx with the API at `/api/`.

## Environment variables

### Core settings

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_ENGINE` | `faster-whisper` | STT backend engine. `faster-whisper` is the production default and currently the only supported value. |
| `STT_MODEL_SIZE` | `large-v3` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`, …) |
| `STT_DEVICE` | `auto` | `cuda` or `cpu` |
| `STT_COMPUTE_TYPE` | `auto` | `float16`, `int8`, etc. |
| `STT_MODEL_DIR` | `./data/models` | Host path for cached models |
| `HOST_PORT` | `5101` | External port |

### Speaker diarization (pyannote.audio)

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_HF_TOKEN` | *(unset)* | HuggingFace token for gated pyannote model access |
| `STT_PYANNOTE_MODEL` | `pyannote/speaker-diarization-3.1` | Pyannote model ID |
| `STT_WARMUP_PYANNOTE` | `true` | Pre-warm pyannote at startup; set `false` for lazy-load |

### VRAM management

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_DIARIZE_WHISPER_MODEL` | *(unset)* | Whisper model to use for diarized requests (overrides `STT_MODEL_SIZE`). Use `medium` to reduce VRAM when diarization is active. |
| `STT_PYANNOTE_IDLE_TIMEOUT_SEC` | `300` | Unload pyannote after N seconds of inactivity to free VRAM; `0` disables auto-unload. |

## VRAM resource limits and co-tenancy

The production image is pinned to a CUDA 12.4-compatible PyTorch stack. If the
host NVIDIA driver is older than that target, diarization startup will fail even
when `STT_HF_TOKEN` is configured correctly.

Whisper large-v3 + pyannote combined can approach **5 GB VRAM**.  Use the
following guidance when co-locating this service with other GPU workloads:

| Configuration | Approx. VRAM |
|---------------|-------------|
| Whisper `large-v3` only | ~3.1 GB |
| pyannote `speaker-diarization-3.1` only | ~1.5 GB |
| Both loaded simultaneously | ~4.5–5.0 GB |
| Whisper `medium` + pyannote | ~3.0–3.5 GB |
| Whisper `small` + pyannote | ~2.3–2.7 GB |

### Recommended strategies

**Strategy A – Idle unload (default)**
Pyannote is unloaded automatically after `STT_PYANNOTE_IDLE_TIMEOUT_SEC`
(default 5 min) of inactivity.  VRAM is returned to the OS between diarize
workloads, allowing other services to use it.  The pipeline is transparently
reloaded on the next diarize request.

```env
STT_PYANNOTE_IDLE_TIMEOUT_SEC=300   # seconds; 0 = never unload
STT_WARMUP_PYANNOTE=true            # pre-warm at startup; false = lazy-load
```

**Strategy B – Smaller Whisper for diarized requests**
If the combined budget of large-v3 + pyannote exceeds your GPU's VRAM, set
`STT_DIARIZE_WHISPER_MODEL=medium`.  Diarized requests will use the medium
model (~1.5 GB) while non-diarized requests still use large-v3.

```env
STT_MODEL_SIZE=large-v3             # used for non-diarized requests
STT_DIARIZE_WHISPER_MODEL=medium    # used when diarize=true
```

**Strategy C – Lazy-load pyannote**
Disable startup warmup so pyannote is never loaded unless a diarize request
arrives.  Combine with a short idle timeout to free VRAM quickly:

```env
STT_WARMUP_PYANNOTE=false
STT_PYANNOTE_IDLE_TIMEOUT_SEC=120
```

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check + model status |
| GET | `/api/info` | GPU info, model details, diarization config |
| POST | `/api/transcribe` | Upload audio, get transcription |

### Transcription response fields

The `/api/transcribe` response includes timing breakdown fields useful for
benchmarking (issue #30):

| Field | Description |
|-------|-------------|
| `processing_time` | Total wall-clock time (seconds) |
| `whisper_time` | Time spent in Whisper inference only |
| `diarization_time` | Time spent in pyannote diarization (null when `diarize=false`) |

## Benchmarking (issue #30)

Use the included benchmark script to measure latency across file durations:

```bash
# Non-diarized baseline only
python scripts/benchmark.py sample_30s.wav sample_2min.wav sample_10min.wav --url http://localhost:5101

# Compare baseline vs diarized; run each file 3 times for a stable mean
python scripts/benchmark.py --diarize --runs 3 sample_30s.wav sample_2min.wav
```

The script reports per-run breakdown (`whisper_s`, `diarize_s`, realtime
factor) and flags whether diarization overhead stays within the < 50% target.

## Downstream integration

The STT service exposes a stable, versioned API for downstream consumers. See
the [docs/integration/](docs/integration/) directory for detailed contracts and
examples:

| Document | Description |
|----------|-------------|
| [gateway-api-workflow.md](docs/integration/gateway-api-workflow.md) | Request/response contract for a gateway-api workflow step that calls the STT service with `diarize=true`. Includes full example payloads, diarized transcript shape, and speaker summary. |
| [chat-platform-agent.md](docs/integration/chat-platform-agent.md) | API examples, constraints, and follow-up notes for a chat-platform agent tool that transcribes uploaded audio with speaker attribution. |

## Control-plane integration

Deploy as a remote workload to a GPU node registered in `gateway-control-plane`.
See [docs/control-plane-followup.md](docs/control-plane-followup.md) for health
probe timeout expectations, model warmup behaviour, new env vars, and VRAM
resource limits.

## Engine selection

The production engine is **faster-whisper (large-v3)**.  See
[docs/engine-decision.md](docs/engine-decision.md) for benchmark data, VRAM
budget analysis, and the rationale behind this decision.  A rollback plan is
also documented there.
