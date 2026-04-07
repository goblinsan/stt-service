# Control-Plane Integration Notes

**Audience:** `gateway-control-plane` operators  
**Related issue:** #18  
**Do not** edit private operator config from this repo.  File a follow-up task
in `gateway-control-plane` for each item below.

---

## Required follow-up changes in gateway-control-plane

### 1. Health probe timeout expectations

The stt-api container exposes a `/api/health` endpoint:

```
GET /api/health
→ 200 { "status": "ok", "model": { "ready": true/false, … } }
```

**What changes:** the `model` object now includes an `engine` field
(visible via `/api/info`).  The health endpoint itself (`/api/health`) is
unchanged.

**Action needed:**
- Confirm that the health probe timeout allows for model warm-up.  Whisper
  `large-v3` loads in **15–45 s** on cold-start (first request); subsequent
  starts from a warm cache are **3–8 s**.
- The `start_period` in `docker-compose.yml` is already set to **180 s** for
  initial model downloads.  If the control plane uses its own probe interval,
  set `initialDelaySeconds ≥ 30` and `timeoutSeconds ≥ 10`.
- If `STT_WARMUP_PYANNOTE=false` is deployed, pyannote loads on the first
  `/api/transcribe?diarize=true` request, not at startup.  Do **not** include
  pyannote readiness in startup probes unless `STT_WARMUP_PYANNOTE=true`.

### 2. Readiness semantics

`/api/health` returns `"model.ready": false` while Whisper is still loading.
The control plane should poll until `ready` is `true` before routing live
traffic.

`/api/health` returns `"model.diarization.ready": false` while pyannote is
unloaded (expected during idle periods when
`STT_PYANNOTE_IDLE_TIMEOUT_SEC > 0`).  Do **not** treat pyannote-not-ready as
a health failure — this is normal idle behaviour.

### 3. New environment variables (v0 → current)

The following env vars were added during the engine-selection process.
Apply them to the deployment manifest / helm values:

| Variable | Default | Action |
|----------|---------|--------|
| `STT_ENGINE` | `faster-whisper` | Forward to container; keep default unless a migration is planned |
| `STT_PYANNOTE_IDLE_TIMEOUT_SEC` | `300` | Tune per VRAM budget; `0` disables auto-unload |
| `STT_WARMUP_PYANNOTE` | `true` | Set `false` to save VRAM when diarization is rarely used |
| `STT_DIARIZE_WHISPER_MODEL` | *(unset)* | Set to `medium` if VRAM budget < 5 GB and diarization is needed |

### 4. Model warmup behaviour

| Phase | Behaviour |
|-------|-----------|
| Container start | Whisper model loads asynchronously; `/api/health.model.ready` transitions `false → true` |
| pyannote enabled (`STT_HF_TOKEN` set, `STT_WARMUP_PYANNOTE=true`) | pyannote pipeline also loads at startup |
| pyannote idle unload | After `STT_PYANNOTE_IDLE_TIMEOUT_SEC` of no diarize requests, pyannote is unloaded — `diarization.ready` → `false` |
| pyannote re-load | Transparently triggered by the next `diarize=true` request; adds ~5–10 s latency |

### 5. Resource requests / limits

Minimum recommendations for the GPU node:

| Resource | Recommendation |
|----------|---------------|
| GPU VRAM | ≥ 8 GB (16 GB comfortable) |
| RAM | ≥ 4 GB |
| CPU | ≥ 2 cores |
| Storage | ≥ 10 GB on the model volume (`STT_MODEL_DIR`) |

Combined large-v3 + pyannote peak: ~5 GB VRAM.  Allow headroom for the OS
and other co-tenants.
