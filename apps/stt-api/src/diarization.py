"""Speaker diarization using pyannote.audio.

Provides a thread-safe singleton loader for the pyannote
``speaker-diarization-3.1`` pipeline, mirroring the pattern used in
engine.py for the Whisper model.

Idle-timeout unloading (issue #32)
------------------------------------
When ``pyannote_idle_timeout_sec > 0`` the pipeline is automatically unloaded
after that many seconds of inactivity, freeing the ~1.5 GB of VRAM it
occupies.  It will be transparently reloaded on the next diarize request.
"""

import logging
import threading
import time

logger = logging.getLogger("stt.diarization")

_pipeline = None
_pipeline_lock = threading.Lock()
_inference_lock = threading.Lock()

# Monotonic timestamp of the most recent pipeline use.  Updated inside
# _inference_lock so the reaper sees a consistent value.
_last_used: float = 0.0

# Background reaper thread – created lazily, daemon so it never blocks exit.
_reaper_thread: threading.Thread | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reaper(idle_timeout_sec: int) -> None:
    """Background thread: unload the pipeline after it has been idle."""
    global _pipeline, _last_used
    while True:
        time.sleep(max(10, idle_timeout_sec // 6))
        with _pipeline_lock:
            if _pipeline is None:
                # Nothing to unload; keep running in case it is reloaded later.
                continue
            idle = time.monotonic() - _last_used
            if idle >= idle_timeout_sec:
                logger.info(
                    "Pyannote pipeline idle for %.0fs (limit %ds) — unloading to free VRAM",
                    idle, idle_timeout_sec,
                )
                _pipeline = None


def _ensure_reaper(idle_timeout_sec: int) -> None:
    """Start the reaper daemon thread if it hasn't been started yet."""
    global _reaper_thread
    if _reaper_thread is not None and _reaper_thread.is_alive():
        return
    t = threading.Thread(
        target=_reaper,
        args=(idle_timeout_sec,),
        daemon=True,
        name="pyannote-reaper",
    )
    t.start()
    _reaper_thread = t


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_pipeline_loaded() -> bool:
    return _pipeline is not None


def unload_pipeline() -> None:
    """Immediately release the pipeline and its VRAM.

    Safe to call even if the pipeline is not loaded.
    """
    global _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            logger.info("Unloading pyannote pipeline (manual request)")
            _pipeline = None


def get_pipeline(hf_token: str, model_cache_dir: str, pyannote_model: str):
    """Return the (singleton) pyannote diarization pipeline.

    The pipeline is loaded once and cached in-process.  The function is
    thread-safe: concurrent callers block until the first load finishes.

    When ``STT_PYANNOTE_IDLE_TIMEOUT_SEC`` is non-zero a background daemon
    thread will unload the pipeline after the configured idle period so VRAM
    is returned to the OS for co-tenant GPU services.

    Args:
        hf_token: HuggingFace auth token for gated model access.
        model_cache_dir: Base directory for model caches (``HF_HOME`` is
            set at container level to persist downloads here).
        pyannote_model: HuggingFace model ID
            (default ``pyannote/speaker-diarization-3.1``).

    Raises:
        RuntimeError: If ``pyannote.audio`` is not installed.
    """
    global _pipeline, _last_used

    # Fast path (no lock needed for a simple None-check).
    if _pipeline is not None:
        return _pipeline

    with _pipeline_lock:
        if _pipeline is not None:  # double-checked locking
            return _pipeline

        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError(
                "pyannote.audio is not installed. "
                "Add pyannote.audio to requirements.txt and rebuild the image."
            ) from exc

        logger.info("Loading %s — first run downloads model weights", pyannote_model)
        pipeline = Pipeline.from_pretrained(
            pyannote_model,
            use_auth_token=hf_token,
        )

        try:
            import torch

            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
                logger.info("Pyannote pipeline moved to CUDA")
            else:
                logger.info("Pyannote pipeline running on CPU")
        except Exception:
            logger.warning("Could not move pyannote pipeline to CUDA, running on CPU")

        _pipeline = pipeline
        _last_used = time.monotonic()
        logger.info("Pyannote pipeline ready")

        # Start the idle-timeout reaper if configured.
        from .config import settings
        if settings.pyannote_idle_timeout_sec > 0:
            _ensure_reaper(settings.pyannote_idle_timeout_sec)

        return _pipeline


def diarize(
    audio_path: str,
    hf_token: str,
    model_cache_dir: str,
    pyannote_model: str,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[tuple[float, float, str]]:
    """Run speaker diarization on *audio_path*.

    Returns a list of ``(start, end, speaker_label)`` tuples sorted by
    start time.  The inference lock serialises calls so that GPU memory
    is not exhausted when multiple requests arrive concurrently.

    Args:
        audio_path: Path to the audio file.
        hf_token: HuggingFace auth token for gated model access.
        model_cache_dir: Base directory for model caches.
        pyannote_model: HuggingFace model ID.
        min_speakers: Optional lower bound hint on the number of speakers.
        max_speakers: Optional upper bound hint on the number of speakers.
    """
    global _last_used

    pipeline = get_pipeline(hf_token, model_cache_dir, pyannote_model)

    kwargs: dict = {}
    if min_speakers is not None:
        if min_speakers < 1:
            raise ValueError("min_speakers must be >= 1")
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        if max_speakers < 1:
            raise ValueError("max_speakers must be >= 1")
        kwargs["max_speakers"] = max_speakers
    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise ValueError(
            f"min_speakers ({min_speakers}) must be <= max_speakers ({max_speakers})"
        )

    with _inference_lock:
        result = pipeline(audio_path, **kwargs)
        _last_used = time.monotonic()

    return [
        (round(turn.start, 3), round(turn.end, 3), speaker)
        for turn, _, speaker in result.itertracks(yield_label=True)
    ]
