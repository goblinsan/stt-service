import logging
import threading
import time
from pathlib import Path

from faster_whisper import WhisperModel

from .config import SUPPORTED_ENGINES, settings
from .models import Segment, SpeakerSummary, TranscribeResult, WordSegment

logger = logging.getLogger("stt.engine")

# Validate engine selection at import time so misconfigurations surface immediately.
if settings.engine not in SUPPORTED_ENGINES:
    raise ValueError(
        f"Unsupported STT_ENGINE: {settings.engine!r}. "
        f"Supported engines: {', '.join(SUPPORTED_ENGINES)}"
    )

_BYTES_PER_MB = 1024 * 1024

# Primary Whisper model (used for all non-diarized requests and when no
# diarize_whisper_model override is configured).
_model: WhisperModel | None = None
_model_lock = threading.Lock()

# Secondary Whisper model used for diarized requests when
# STT_DIARIZE_WHISPER_MODEL is set to a different model size (issue #32).
# Keeping a separate singleton avoids re-loading on every request while still
# allowing a smaller model (e.g. "medium") to co-exist with pyannote without
# blowing the VRAM budget.
_diarize_model: WhisperModel | None = None
_diarize_model_lock = threading.Lock()

_inference_lock = threading.Lock()


def _resolve_device() -> tuple[str, str]:
    device = settings.device
    compute_type = settings.compute_type

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


def _load_whisper(model_size: str) -> WhisperModel:
    """Load a WhisperModel for *model_size* and return it."""
    device, compute_type = _resolve_device()
    logger.info(
        "Loading Whisper model %s on %s (%s) — cache: %s",
        model_size, device, compute_type, settings.model_cache_dir,
    )
    Path(settings.model_cache_dir).mkdir(parents=True, exist_ok=True)
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=settings.model_cache_dir,
    )
    logger.info("Whisper model %s loaded", model_size)
    return model


def get_model() -> WhisperModel:
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        _model = _load_whisper(settings.model_size)
        return _model


def get_diarize_model() -> WhisperModel:
    """Return the Whisper model to use for diarized requests.

    If ``STT_DIARIZE_WHISPER_MODEL`` is set to a *different* model size than
    ``STT_MODEL_SIZE``, a separate singleton is loaded and returned.  This
    allows operators to run a smaller Whisper (e.g. ``medium``) alongside
    pyannote to stay within VRAM budgets (issue #32).

    When ``STT_DIARIZE_WHISPER_MODEL`` is unset or equals ``STT_MODEL_SIZE``,
    the standard ``_model`` singleton is returned.
    """
    global _diarize_model
    diarize_size = settings.diarize_whisper_model
    if not diarize_size or diarize_size == settings.model_size:
        return get_model()

    if _diarize_model is not None:
        return _diarize_model
    with _diarize_model_lock:
        if _diarize_model is not None:
            return _diarize_model
        logger.info(
            "VRAM-tier: loading smaller Whisper model '%s' for diarized requests "
            "(main model is '%s')",
            diarize_size, settings.model_size,
        )
        _diarize_model = _load_whisper(diarize_size)
        return _diarize_model


def is_model_loaded() -> bool:
    return _model is not None


def get_device_info() -> tuple[str, str]:
    return _resolve_device()


def _vram_used_mb() -> int | None:
    """Return currently allocated VRAM in MB, or None if CUDA is unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated(0) / _BYTES_PER_MB)
    except Exception:
        pass
    return None


def transcribe_audio(
    audio_path: str,
    language: str | None = None,
    task: str = "transcribe",
    word_timestamps: bool = False,
    initial_prompt: str | None = None,
    diarize: bool = False,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> TranscribeResult:
    model = get_diarize_model() if diarize else get_model()

    t0 = time.monotonic()

    vram_before_whisper = _vram_used_mb()

    with _inference_lock:
        segments_iter, info = model.transcribe(
            audio_path,
            language=language,
            task=task,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        segments: list[Segment] = []
        full_text_parts: list[str] = []

        for seg in segments_iter:
            words = None
            if word_timestamps and seg.words:
                words = [
                    WordSegment(
                        word=w.word,
                        start=round(w.start, 3),
                        end=round(w.end, 3),
                        probability=round(w.probability, 3),
                    )
                    for w in seg.words
                ]

            segments.append(
                Segment(
                    id=seg.id,
                    start=round(seg.start, 3),
                    end=round(seg.end, 3),
                    text=seg.text.strip(),
                    words=words,
                )
            )
            full_text_parts.append(seg.text.strip())

    whisper_time = round(time.monotonic() - t0, 3)
    vram_after_whisper = _vram_used_mb()
    if vram_before_whisper is not None and vram_after_whisper is not None:
        logger.debug(
            "Whisper inference: %.1fs, VRAM %d→%d MB",
            whisper_time, vram_before_whisper, vram_after_whisper,
        )

    speakers: list[SpeakerSummary] | None = None
    diarization_time: float | None = None

    if diarize:
        if not settings.hf_token:
            raise ValueError(
                "STT_HF_TOKEN must be set to use speaker diarization"
            )
        from .diarization import diarize as run_diarize
        from .merge import assign_speakers

        t_diarize = time.monotonic()
        vram_before_diarize = _vram_used_mb()

        logger.info("Running speaker diarization on %s", audio_path)
        diarization_segs = run_diarize(
            audio_path,
            settings.hf_token,
            settings.model_cache_dir,
            settings.pyannote_model,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        segments = assign_speakers(segments, diarization_segs)

        diarization_time = round(time.monotonic() - t_diarize, 3)
        vram_after_diarize = _vram_used_mb()

        logger.info(
            "Diarization complete: %d speaker turns assigned in %.1fs",
            len(diarization_segs), diarization_time,
        )
        if vram_before_diarize is not None and vram_after_diarize is not None:
            logger.debug(
                "Diarization VRAM %d→%d MB (peak delta %+d MB)",
                vram_before_diarize, vram_after_diarize,
                vram_after_diarize - vram_before_diarize,
            )

        speaker_stats: dict[str, dict] = {}
        for seg in segments:
            if seg.speaker:
                if seg.speaker not in speaker_stats:
                    speaker_stats[seg.speaker] = {"total_duration": 0.0, "segment_count": 0}
                speaker_stats[seg.speaker]["total_duration"] += seg.end - seg.start
                speaker_stats[seg.speaker]["segment_count"] += 1
        speakers = [
            SpeakerSummary(
                id=spk_id,
                total_duration=round(stats["total_duration"], 3),
                segment_count=stats["segment_count"],
            )
            for spk_id, stats in sorted(speaker_stats.items())
        ]

    return TranscribeResult(
        text=" ".join(full_text_parts),
        language=info.language,
        language_probability=round(info.language_probability, 3),
        duration=round(info.duration, 3),
        segments=segments,
        processing_time=round(time.monotonic() - t0, 3),
        whisper_time=whisper_time,
        diarization_time=diarization_time,
        speakers=speakers,
    )

