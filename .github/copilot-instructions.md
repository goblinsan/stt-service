# Copilot Instructions

This repository provides GPU-backed speech-to-text and diarization services and is intended to be deployed by `gateway-control-plane` as a `container-service` using the repo's `docker-compose.yml`.

## Deployment contract

- Keep `docker-compose.yml`, `apps/stt-api/Dockerfile`, `apps/stt-api/src/`, `apps/stt-ui/`, and `README.md` aligned.
- The API service must continue listening on container port `5100`.
- The published UI/API entrypoint must remain configurable through `HOST_PORT` on the nginx service.
- Health checks must continue to use `GET /api/health`.

## Configuration rules

- Do not hardcode private LAN IPs, node IDs, hostnames, usernames, or local volume paths in tracked files.
- Keep local operator values as placeholders in docs.
- If you add or change an STT environment variable, update all of:
  - `apps/stt-api/src/config.py`
  - `docker-compose.yml`
  - `README.md`
  - any control-plane follow-up doc affected by the change

## Feature discipline

- `STT_ENGINE` is configuration, but only documented and supported engines should be accepted at runtime.
- If diarization behavior changes, keep these in sync:
  - API request fields
  - response models
  - UI rendering
  - `docs/control-plane-followup.md`
  - integration docs under `docs/integration/`
- Treat the web tester as part of the product contract, not a demo. If the API gains a new operator-facing control or readiness state, `apps/stt-ui/` should expose it clearly enough that a human can verify the feature from the browser.
- Do not claim gateway-control-plane, gateway-api, or chat-platform changes are complete unless they were actually implemented in those repositories.

## VRAM and co-tenancy

- Treat VRAM budgeting as part of the contract, not a comment.
- If adding a new model path or warmup mode, keep compose wiring and operator docs synchronized.
- Avoid changes that silently force large-v3 + pyannote co-residency without documenting the impact.
- Do not leave GPU-runtime dependencies open-ended across major versions. For `torch`, `torchaudio`, and `pyannote.audio`, prefer explicit pins that match the documented CUDA/driver target instead of `>=` ranges.
- Treat `pyannote.audio` and `huggingface_hub` as a compatibility pair. If one changes, verify Hugging Face auth still works through the full diarization path instead of assuming `token` or `use_auth_token` is stable across both packages.
- Do not rely on `pyannote.audio` transitive runtime dependencies being present by accident. If the diarization runtime imports packages like `matplotlib`, add and pin them explicitly in `apps/stt-api/requirements.txt`.
- Prefer preloading audio into memory for pyannote inference instead of relying on its built-in file decoding path. Decoder/runtime warnings on the GPU node are a real deployment constraint, not harmless noise.
- For the default `pyannote/speaker-diarization-3.1` pipeline, remember the linked gated dependency on `pyannote/segmentation-3.0`. Operator docs and error messages should say that explicitly instead of only mentioning `STT_HF_TOKEN`.
- GPU status/reporting endpoints must degrade safely. Never let optional CUDA metadata calls take down `/api/info` or other readiness surfaces when the driver/runtime stack is imperfect.

## Validation

Before considering a task complete, prefer verifying:

- Python syntax for API, diarization, and benchmark modules
- UI build shape still matches the API responses
- Compose env passthrough matches the documented configuration surface
