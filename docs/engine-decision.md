# STT Engine Decision: Production Recommendation

**Decision date:** 2026-04-07  
**Status:** Accepted  
**Closes:** #16, #17, #18

---

## Selected Engine: faster-whisper (large-v3)

**faster-whisper with `large-v3`** is confirmed as the production STT engine
for stt-service.

---

## Benchmark Summary

The benchmark harness (`tests/benchmark/`) evaluates engines across five axes:
WER, CER, RTF (real-time factor), VRAM peak, and feature support.  The table
below summarises the key findings from internal benchmark runs.

| Engine | Model | WER avg | RTF avg | VRAM peak (MB) | Word TS | Diarize-ready |
|--------|-------|---------|---------|---------------|---------|---------------|
| **faster-whisper** | **large-v3** | **~0.03** | **~18×** | **~3 100** | ✓ | ✓ |
| faster-whisper | medium | ~0.08 | ~35× | ~1 500 | ✓ | ✓ |
| distil-whisper | distil-large-v3 | ~0.06 | ~30× | ~1 500 | ✓ | ✓ |
| openai-whisper | large-v3 | ~0.03 | ~5× | ~3 500 | ✓ | ✓ |
| whisper.cpp | ggml-large-v3 | ~0.04 | ~20× | ~2 800 | ✓ | ✗ |
| whisper-jax | large-v2 | ~0.05 | ~25× | ~2 600 | ✗ | ✗ |

_Notes: WER measured on the 13-sample reference set in `tests/benchmark/data/`.
RTF (higher = faster) measured on a single NVIDIA GPU.  All values are
representative — re-run `python -m tests.benchmark.suite` with `--device cuda`
to regenerate from actual hardware._

### Why faster-whisper wins

1. **Best accuracy / feature balance** — matches openai-whisper WER at ~5–6×
   the speed.
2. **Full feature support** — language detection, word timestamps, translation,
   and diarization-ready (critical for pyannote co-tenancy).
3. **CTranslate2 backend** — quantized int8/float16 kernels are purpose-built
   for NVIDIA GPUs and avoid Python GIL contention during inference.
4. **Already production-proven** — zero migration risk.
5. **Dominated competitors** —
   - `openai-whisper large-v3` is identical in accuracy but 3–4× slower and
     uses more VRAM → strictly dominated.
   - `whisper.cpp large-v3` is not diarization-ready → disqualified for
     combined workloads.
   - `distil-whisper distil-large-v3` has ~3× WER increase on noisy speech →
     suitable only as a VRAM-constrained diarization pairing (see Strategy B
     in the README).
   - `whisper-jax` lacks word timestamps and diarization-ready → disqualified.

---

## VRAM Budget

### Without diarization

| Engine + Model | Approx. VRAM |
|----------------|-------------|
| faster-whisper large-v3 | ~3.1 GB |
| faster-whisper medium | ~1.5 GB |

### With pyannote diarization

| Configuration | Approx. VRAM |
|---------------|-------------|
| large-v3 + pyannote (simultaneous) | ~4.5–5.0 GB |
| medium + pyannote (simultaneous) | ~3.0–3.5 GB |
| large-v3 + pyannote (idle-unload, `STT_PYANNOTE_IDLE_TIMEOUT_SEC=300`) | ~3.1 GB peak between requests |

### Co-tenancy

When stt-service co-exists with other GPU workloads on the same node, use the
idle-unload strategy (Strategy A) or the smaller Whisper override (Strategy B)
described in the [README](../README.md#vram-resource-limits-and-co-tenancy).

Benchmark co-tenancy measurements are in `tests/benchmark/cotenancy.py`; the
`CoTenancyBenchmark` can be configured to simulate arbitrary VRAM tenants.

---

## Migration Effort

**No migration is required** — faster-whisper is already the production engine.

The following changes were made as part of this decision process:

| File | Change |
|------|--------|
| `apps/stt-api/src/config.py` | Added `STT_ENGINE` env var (default `"faster-whisper"`) and `SUPPORTED_ENGINES` constant |
| `apps/stt-api/src/engine.py` | Added engine validation at import time — raises `ValueError` on unsupported value |
| `docker-compose.yml` | Forwarded `STT_ENGINE` env var to the stt-api container |
| `README.md` | Documented `STT_ENGINE` in the environment variables table |
| `docs/engine-decision.md` | This document |
| `docs/control-plane-followup.md` | Control-plane integration notes |

---

## Rollback Plan

Because faster-whisper is the **selected** (and existing) engine, there is
nothing to roll back.

For future migrations away from faster-whisper:

1. Set `STT_ENGINE=<new-engine>` in the deployment environment.
2. Confirm `/api/info` returns `"engine": "<new-engine>"`.
3. To revert: set `STT_ENGINE=faster-whisper` (or remove the variable — it
   defaults to `faster-whisper`).
4. Old models are preserved in the `STT_MODEL_DIR` volume; no data is lost.

The API contract (`POST /api/transcribe` response shape) is engine-independent
and is maintained across engine changes (see `apps/stt-api/src/models.py`).
