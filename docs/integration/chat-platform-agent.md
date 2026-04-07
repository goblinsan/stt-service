# Chat-Platform Agent Tool: Diarized Audio Transcription

This document provides the API examples and constraints needed to build a
chat-platform agent tool that transcribes uploaded audio files with speaker
attribution. The actual agent-tool implementation lives in
`gateway-chat-platform`; this file is the stable API reference and integration
guide for that work.

> **Scope**: deliverable lives in `stt-service`. The agent-tool implementation
> is a follow-up task in `gateway-chat-platform`.

---

## Overview

A chat-platform agent tool allows users to upload audio recordings (calls,
meetings, voice notes) and receive a speaker-attributed transcript inline. The
tool posts the audio to the STT service `/api/transcribe` endpoint with
`diarize=true` and converts the structured response into a user-readable
summary.

---

## Constraints and requirements

### Supported audio formats

The STT service accepts the following file types (authoritative list in
[gateway-api-workflow.md](gateway-api-workflow.md#request)). The agent tool
**must** validate the extension before uploading and surface a clear error to
the user for unsupported types:

```
.wav  .mp3  .flac  .ogg  .m4a  .webm  .mp4  .aac  .wma  .aif  .aiff
```

> **Note**: if the service adds or removes supported formats, update
> `gateway-api-workflow.md` first — it is the single source of truth — then
> reflect the change here.

### File size limit

Default maximum: **500 MB** (configurable via `STT_MAX_UPLOAD_MB` on the
service side; see [gateway-api-workflow.md](gateway-api-workflow.md#maximum-upload-size)).
The agent tool should inform the user early if their file exceeds this limit
rather than letting the upload fail at the API boundary.

### Diarization availability

Speaker attribution (`diarize=true`) requires the service to be started with a
valid `STT_HF_TOKEN` (HuggingFace token for `pyannote/speaker-diarization-3.1`).
If the token is absent the service returns HTTP `422` with:

```json
{
  "detail": "Speaker diarization is not configured. Set the STT_HF_TOKEN environment variable to enable it."
}
```

The agent tool should detect this and fall back gracefully to plain
transcription (`diarize=false`), or inform the user that speaker attribution
is unavailable.

### Latency expectations

| Audio length | Typical processing time |
|--------------|------------------------|
| 30 s | 5–15 s |
| 2 min | 20–60 s |
| 10 min | 90–300 s |

Times vary with GPU hardware and the configured Whisper model size. The agent
tool should show a progress indicator and use a request timeout of at least
**600 seconds**.

### Concurrency

The STT service serialises GPU inference with an internal lock, so concurrent
requests from multiple chat sessions queue rather than fail. There is no
built-in rate-limit; the gateway layer should enforce per-user quotas as
appropriate.

---

## API reference

### Endpoint

```
POST /api/transcribe
Content-Type: multipart/form-data
```

### Request fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | binary | — | Audio file (required). |
| `diarize` | bool | `false` | `true` for speaker attribution. |
| `language` | string | auto | ISO 639-1 code. Omit for auto-detect. |
| `task` | string | `transcribe` | `transcribe` or `translate` (→ English). |
| `word_timestamps` | bool | `false` | Include word-level timestamps. |
| `initial_prompt` | string | — | Domain-specific vocabulary hint for Whisper. |
| `min_speakers` | int | — | Lower bound on expected speaker count (≥ 1). |
| `max_speakers` | int | — | Upper bound on expected speaker count (≥ 1). |

### Health check

Before submitting audio, verify the service is ready:

```
GET /api/health
```

Gate on `model.ready == true`. For diarized requests, also check
`model.diarization.ready`.

---

## Example API calls

### Minimal — diarized transcription

```bash
curl -X POST http://localhost:5101/api/transcribe \
  -F "file=@meeting.mp3" \
  -F "diarize=true"
```

### With language and speaker hints

```bash
curl -X POST http://localhost:5101/api/transcribe \
  -F "file=@standup.wav" \
  -F "diarize=true" \
  -F "language=en" \
  -F "min_speakers=3" \
  -F "max_speakers=6"
```

### Python (agent tool pattern)

```python
import requests
from pathlib import Path

STT_BASE_URL = "http://stt-service/api"   # resolved by service mesh / env var

SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".ogg", ".m4a",
    ".webm", ".mp4", ".aac", ".wma", ".aif", ".aiff",
}
MAX_FILE_MB = 500


def transcribe_with_speakers(
    audio_path: str,
    language: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict:
    """
    Upload *audio_path* to the STT service and return a diarized transcript.

    Returns the parsed JSON response (TranscribeResult).
    Raises ValueError for unsupported formats or oversized files.
    Raises requests.HTTPError for API errors.
    """
    path = Path(audio_path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise ValueError(
            f"File is {size_mb:.1f} MB which exceeds the {MAX_FILE_MB} MB limit."
        )

    data: dict = {"diarize": "true"}
    if language:
        data["language"] = language
    if min_speakers is not None:
        data["min_speakers"] = str(min_speakers)
    if max_speakers is not None:
        data["max_speakers"] = str(max_speakers)

    with path.open("rb") as fh:
        response = requests.post(
            f"{STT_BASE_URL}/transcribe",
            files={"file": (path.name, fh, "audio/octet-stream")},
            data=data,
            timeout=600,
        )

    if response.status_code == 422:
        detail = response.json().get("detail", response.text)
        if "STT_HF_TOKEN" in detail:
            # Diarization not configured; fall back to non-diarized call
            data["diarize"] = "false"
            with path.open("rb") as fh:
                response = requests.post(
                    f"{STT_BASE_URL}/transcribe",
                    files={"file": (path.name, fh, "audio/octet-stream")},
                    data=data,
                    timeout=600,
                )

    response.raise_for_status()
    return response.json()
```

---

## Response shape

### `TranscribeResult`

```json
{
  "text": "string — full concatenated transcript",
  "language": "en",
  "language_probability": 0.999,
  "duration": 180.0,
  "segments": [ /* Segment objects */ ],
  "processing_time": 22.4,
  "whisper_time": 14.1,
  "diarization_time": 8.3,
  "speakers": [ /* SpeakerSummary objects */ ]
}
```

`speakers` and `diarization_time` are `null` when `diarize=false`.

### `Segment`

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

`speaker` is `null` when `diarize=false` or when a segment falls in a gap
between diarization turns.

### `SpeakerSummary`

```json
{
  "id": "SPEAKER_00",
  "total_duration": 42.75,
  "segment_count": 8
}
```

Speaker labels (`SPEAKER_00`, `SPEAKER_01`, …) are assigned by pyannote and
are **not** stable across calls. Do not persist them as business identifiers.

---

## Rendering speaker-attributed output for chat users

The agent tool should convert the structured response into a readable format.
A recommended pattern:

```python
def format_transcript(result: dict) -> str:
    """Convert a TranscribeResult dict into a chat-friendly transcript."""
    lines = []
    current_speaker = None

    for seg in result["segments"]:
        speaker = seg.get("speaker") or "Unknown"
        text = seg["text"].strip()
        start = seg["start"]

        if speaker != current_speaker:
            # New speaker turn — emit a header
            minutes, seconds = divmod(int(start), 60)
            lines.append(f"\n**{speaker}** [{minutes:02d}:{seconds:02d}]")
            current_speaker = speaker

        lines.append(text)

    # Append speaker summary if available
    if result.get("speakers"):
        lines.append("\n---\n**Speaker summary**")
        for spk in result["speakers"]:
            mins = int(spk["total_duration"] // 60)
            secs = int(spk["total_duration"] % 60)
            lines.append(
                f"- {spk['id']}: {mins}m {secs}s "
                f"({spk['segment_count']} segments)"
            )

    return "\n".join(lines).strip()
```

**Example output** for a 3-minute two-speaker meeting:

```
**SPEAKER_00** [00:00]
Hello, how are you today?
Let's get started with the interview.

**SPEAKER_01** [00:02]
I'm doing great, thanks for asking.
Sounds good.

---
**Speaker summary**
- SPEAKER_00: 0m 5s (2 segments)
- SPEAKER_01: 0m 4s (2 segments)
```

---

## Error handling

| HTTP status | Agent action |
|-------------|--------------|
| `400` | Surface message to user (bad file type or missing filename). |
| `413` | Inform user the file is too large; suggest compressing or trimming. |
| `422` with `STT_HF_TOKEN` message | Fall back to `diarize=false`; note in response that speaker attribution is unavailable. |
| `422` with speaker constraint message | Fix `min_speakers`/`max_speakers` values or remove them. |
| `500` | Retry once after a short delay; escalate if it persists. |
| Timeout | Inform user; suggest retrying with a shorter clip or during off-peak hours. |

---

## Follow-up notes for `gateway-chat-platform`

The following items are **not** required from `stt-service` (API contract is
complete) but should be addressed in the `gateway-chat-platform` implementation:

1. **Tool registration** — Register the agent tool with the correct input schema
   (file upload + optional language / speaker-count hints).
2. **Speaker re-labelling** — Optionally allow users to assign names to
   `SPEAKER_00`, `SPEAKER_01`, etc. Store the mapping in the chat session
   context, not in the STT service.
3. **Streaming progress** — The STT service is synchronous; the agent tool
   should post an interim "Transcribing…" message while waiting.
4. **Result caching** — Cache results by file hash to avoid re-transcribing
   the same audio across sessions. The STT service does not cache internally.
5. **Access control** — Gate diarized transcription on a feature flag or
   entitlement if GPU resources are limited; plain transcription is always
   available.
6. **Audio pre-processing** — For very long recordings (> 30 min), consider
   splitting server-side before calling the STT service to improve latency and
   reduce memory pressure. No splitting logic exists in `stt-service`.
