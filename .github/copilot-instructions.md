# Copilot Instructions

This repository provides GPU-backed speech-to-text and diarization services and is intended to be deployed by `gateway-control-plane` as a `container-service` using the repo's `docker-compose.yml`.

## Keep Contracts Aligned

- Keep `docker-compose.yml`, `apps/stt-api/Dockerfile`, `apps/stt-api/src/`, `apps/stt-ui/`, and `README.md` aligned.
- When you add operator-facing API capabilities, update the local browser UI too.
- Preserve the current public API surface unless there is an explicit migration plan:
  - `GET /api/health`
  - `GET /api/info`
  - `POST /api/transcribe`
  - `POST /api/transcribe-from-url`
- `gateway-tools-platform` depends on `POST /api/transcribe-from-url` for large uploads. Do not remove or silently change that contract.

## Deployment Assumptions

- This service runs on the GPU node, not the gateway host.
- The deployed stack is the repo root `docker-compose.yml`.
- Environment values must remain configurable via compose/env, especially:
  - `STT_MODEL_SIZE`
  - `STT_DEVICE`
  - `STT_COMPUTE_TYPE`
  - `STT_HF_TOKEN`
  - `STT_DIARIZE_WHISPER_MODEL`
  - `STT_PYANNOTE_IDLE_TIMEOUT_SEC`
  - `STT_WARMUP_PYANNOTE`
  - `STT_REMOTE_SOURCE_ALLOWED_HOSTS`
  - `STT_REMOTE_SOURCE_TIMEOUT_SEC`

## GPU / Runtime Constraints

- Do not broaden core GPU runtime dependencies casually.
- Keep PyTorch / torchaudio pinned to a CUDA stack compatible with the documented node driver target.
- Do not replace pinned production dependencies with broad `>=` ranges for torch, torchaudio, pyannote, or huggingface_hub.
- Avoid assuming Ubuntu package names from the host OS; the relevant environment is the container image.

## Audio / Diarization Rules

- Supported uploaded audio types must stay aligned across:
  - API validation in `apps/stt-api/src/main.py`
  - the local UI file picker / copy in `apps/stt-ui/`
  - repo docs
- Keep `.aif` / `.aiff` support intact.
- Do not rely on pyannote's internal file decoder path on this node when an in-memory / preloaded audio path is safer.
- For the default `pyannote/speaker-diarization-3.1` pipeline, remember the linked gated dependency on `pyannote/segmentation-3.0`. Operator docs and error messages should say that explicitly instead of only mentioning `STT_HF_TOKEN`.
- Remote-source ingest should be treated as a server-to-server fetch path. Prefer allowlisting expected object-store hosts via `STT_REMOTE_SOURCE_ALLOWED_HOSTS` instead of fetching arbitrary URLs.

## Validation

Before considering changes complete, prefer verifying:

- Python syntax passes for changed modules
- API request/response models still match the documented routes
- UI changes still expose the backend capability that was added
- compose/env wiring still passes through any new settings
