# Gateway-API Workflow Step: Diarized Transcription

This document defines the stable request/response contract for a gateway-api
workflow step that calls the STT service with `diarize=true`. It provides
ready-to-use example payloads and describes the output shapes that downstream
workflow steps should expect.

> **Scope**: deliverable lives in `stt-service`. The gateway-api workflow step
> implementation is a follow-up task in `gateway-api`.

---

## Endpoint

```
POST /api/transcribe
Content-Type: multipart/form-data
```

Hosted behind the nginx reverse proxy at `/api/`. When deployed as described
in `docker-compose.yml`, the full URL is:

```
http://<host>:5101/api/transcribe
```

---

## Request

The endpoint accepts `multipart/form-data`. All parameters except `file` are
optional form fields.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | binary | ✓ | — | Audio file. Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.webm`, `.mp4`, `.aac`, `.wma`, `.aif`, `.aiff`. This is the authoritative list; keep `chat-platform-agent.md` in sync when it changes. |
| `diarize` | bool | — | `false` | Set to `true` to enable speaker diarization. Requires `STT_HF_TOKEN` to be configured on the service. |
| `language` | string | — | auto-detect | ISO 639-1 language code (e.g. `en`, `de`, `fr`). Omit for automatic detection. |
| `task` | string | — | `transcribe` | `transcribe` or `translate` (translate always produces English output). |
| `word_timestamps` | bool | — | `false` | Include word-level timestamps in each segment. |
| `initial_prompt` | string | — | — | Optional prompt to condition the Whisper model (e.g. domain vocabulary). |
| `min_speakers` | int | — | — | Lower bound hint for speaker count. Must be ≥ 1 and ≤ `max_speakers`. |
| `max_speakers` | int | — | — | Upper bound hint for speaker count. Must be ≥ 1. |

### Maximum upload size

Default: **500 MB** (configurable via `STT_MAX_UPLOAD_MB`). This is the
authoritative value; keep `chat-platform-agent.md` in sync when it changes.

### Example request (curl)

```bash
curl -X POST http://localhost:5101/api/transcribe \
  -F "file=@interview.wav" \
  -F "diarize=true" \
  -F "language=en" \
  -F "min_speakers=2" \
  -F "max_speakers=4"
```

### Example request (Python)

```python
import requests

with open("interview.wav", "rb") as audio_file:
    response = requests.post(
        "http://localhost:5101/api/transcribe",
        files={"file": ("interview.wav", audio_file, "audio/wav")},
        data={
            "diarize": "true",
            "language": "en",
            "min_speakers": "2",
            "max_speakers": "4",
        },
        timeout=600,
    )
response.raise_for_status()
result = response.json()
```

---

## Response

HTTP `200 OK` with `Content-Type: application/json`.

### Schema (`TranscribeResult`)

```json
{
  "text": "string",
  "language": "string",
  "language_probability": 0.99,
  "duration": 0.0,
  "segments": [ /* see Segment below */ ],
  "processing_time": 0.0,
  "whisper_time": 0.0,
  "diarization_time": 0.0,
  "speakers": [ /* see SpeakerSummary below */ ]
}
```

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `text` | string | no | Full concatenated transcript. |
| `language` | string | no | Detected (or forced) ISO 639-1 language code. |
| `language_probability` | float | no | Model confidence in the detected language (0–1). |
| `duration` | float | no | Audio length in seconds. |
| `segments` | array | no | Time-stamped transcript segments (see below). |
| `processing_time` | float | no | Total wall-clock time in seconds. |
| `whisper_time` | float | yes | Whisper-only inference time. `null` when unavailable. |
| `diarization_time` | float | yes | Pyannote-only diarization time. `null` when `diarize=false`. |
| `speakers` | array | yes | Per-speaker summary. `null` when `diarize=false`. |

### Segment

```json
{
  "id": 0,
  "start": 0.0,
  "end": 2.45,
  "text": "Hello, how are you today?",
  "speaker": "SPEAKER_00",
  "words": null
}
```

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `id` | int | no | Zero-based segment index. |
| `start` | float | no | Segment start time (seconds, 3 decimal places). |
| `end` | float | no | Segment end time (seconds, 3 decimal places). |
| `text` | string | no | Transcript text for the segment. |
| `speaker` | string | yes | Speaker label (e.g. `SPEAKER_00`). `null` when `diarize=false` or coverage gap. |
| `words` | array | yes | Word-level timestamps. Present only when `word_timestamps=true`. |

### WordSegment (when `word_timestamps=true`)

```json
{
  "word": "Hello",
  "start": 0.12,
  "end": 0.54,
  "probability": 0.998
}
```

### SpeakerSummary

```json
{
  "id": "SPEAKER_00",
  "total_duration": 42.75,
  "segment_count": 8
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Speaker label assigned by pyannote (e.g. `SPEAKER_00`, `SPEAKER_01`). |
| `total_duration` | float | Total speaking time for this speaker in seconds (sum of assigned segment durations). |
| `segment_count` | int | Number of transcript segments assigned to this speaker. |

Speaker labels are zero-indexed (`SPEAKER_00`, `SPEAKER_01`, …) in the order
pyannote detects them. The mapping is **not** stable across calls; do not
persist raw speaker IDs as business identifiers.

---

## Full example response

A 3-minute two-speaker interview (`diarize=true`, `language=en`):

```json
{
  "text": "Hello, how are you today? I'm doing great, thanks for asking. Let's get started with the interview. Sounds good.",
  "language": "en",
  "language_probability": 0.999,
  "duration": 180.123,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.45,
      "text": "Hello, how are you today?",
      "speaker": "SPEAKER_00",
      "words": null
    },
    {
      "id": 1,
      "start": 2.8,
      "end": 5.1,
      "text": "I'm doing great, thanks for asking.",
      "speaker": "SPEAKER_01",
      "words": null
    },
    {
      "id": 2,
      "start": 5.6,
      "end": 8.3,
      "text": "Let's get started with the interview.",
      "speaker": "SPEAKER_00",
      "words": null
    },
    {
      "id": 3,
      "start": 8.7,
      "end": 9.9,
      "text": "Sounds good.",
      "speaker": "SPEAKER_01",
      "words": null
    }
  ],
  "processing_time": 22.4,
  "whisper_time": 14.1,
  "diarization_time": 8.3,
  "speakers": [
    {
      "id": "SPEAKER_00",
      "total_duration": 5.2,
      "segment_count": 2
    },
    {
      "id": "SPEAKER_01",
      "total_duration": 4.6,
      "segment_count": 2
    }
  ]
}
```

---

## Error responses

| HTTP status | Condition |
|-------------|-----------|
| `400 Bad Request` | Missing filename, unsupported file extension, or invalid `task` value. |
| `413 Payload Too Large` | Audio file exceeds `STT_MAX_UPLOAD_MB` (default 500 MB). |
| `422 Unprocessable Entity` | `diarize=true` but `STT_HF_TOKEN` is not configured; or invalid `min_speakers`/`max_speakers` values. |
| `500 Internal Server Error` | Transcription or diarization failed unexpectedly. |

### 422 example — diarization not configured

```json
{
  "detail": "Speaker diarization is not configured. Set the STT_HF_TOKEN environment variable to enable it."
}
```

### 422 example — invalid speaker constraints

```json
{
  "detail": "min_speakers (5) must be <= max_speakers (2)"
}
```

---

## Workflow step integration notes

### Health check before invoking

Before submitting audio, the workflow step should verify the service is ready:

```
GET /api/health
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "model": {
    "model_size": "large-v3",
    "device": "cuda",
    "compute_type": "float16",
    "ready": true,
    "diarization": {
      "model": "pyannote/speaker-diarization-3.1",
      "ready": true
    }
  }
}
```

Gate on `model.ready == true`. If `diarize=true` is required, also gate on
`model.diarization.ready == true` (or accept the lazy-load latency on first
request).

### Downstream step contracts

The following fields are stable and safe to pass to downstream workflow steps:

| Downstream need | Field(s) to use |
|-----------------|-----------------|
| Readable transcript | `text` |
| Per-speaker turn list | `segments[].text` + `segments[].speaker` |
| Speaker analytics | `speakers[].id`, `speakers[].total_duration`, `speakers[].segment_count` |
| Timing / SLA checks | `processing_time`, `whisper_time`, `diarization_time` |
| Detected language | `language`, `language_probability` |

### VRAM and latency guidance

- Diarization adds roughly **40–80% overhead** on top of Whisper inference time
  (varies with audio length and GPU model).
- Set `min_speakers` / `max_speakers` when the number of participants is known
  in advance; this reduces pyannote's search space and improves both accuracy
  and speed.
- If the GPU is shared with other services, configure `STT_DIARIZE_WHISPER_MODEL`
  to a smaller model (e.g. `medium`) to keep combined VRAM under budget. See
  `README.md` — *VRAM resource limits and co-tenancy* for details.
- Set a generous HTTP timeout in the workflow step (≥ 600 s for long recordings).
