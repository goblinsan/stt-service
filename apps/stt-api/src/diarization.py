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
import inspect
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
_validated_repo_access: set[str] = set()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reaper(idle_timeout_sec: int) -> None:
    """Background thread: unload the pipeline after it has been idle."""
    global _pipeline, _last_used
    # Wake up at most every 10 s, but no more than every 1/6th of the timeout
    # so we never miss the deadline by more than ~17%.
    _REAPER_MIN_SLEEP_SEC = 10
    _REAPER_FRACTION = 6
    sleep_sec = max(_REAPER_MIN_SLEEP_SEC, idle_timeout_sec // _REAPER_FRACTION)
    while True:
        time.sleep(sleep_sec)
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


def _ensure_hf_hub_legacy_auth_alias() -> None:
    """Patch huggingface_hub to accept legacy ``use_auth_token`` calls.

    Some pyannote releases still forward ``use_auth_token`` to
    ``huggingface_hub.hf_hub_download`` even when the installed hub client only
    accepts ``token``. Patch the module before importing pyannote so any
    ``from huggingface_hub import hf_hub_download`` inside pyannote receives the
    compatibility wrapper.
    """
    try:
        import huggingface_hub
    except ImportError:
        return

    download = getattr(huggingface_hub, "hf_hub_download", None)
    if download is None:
        return

    params = inspect.signature(download).parameters
    if "use_auth_token" in params:
        return
    if getattr(download, "_supports_legacy_use_auth_token", False):
        return

    def compat_hf_hub_download(*args, use_auth_token=None, **kwargs):
        if use_auth_token is not None and "token" not in kwargs:
            kwargs["token"] = use_auth_token
        return download(*args, **kwargs)

    compat_hf_hub_download._supports_legacy_use_auth_token = True
    huggingface_hub.hf_hub_download = compat_hf_hub_download


def _load_audio_for_pyannote(audio_path: str) -> dict:
    """Return audio in the in-memory format pyannote recommends.

    This avoids pyannote's internal torchcodec-based file decoding path, which
    has proven brittle across CUDA / FFmpeg combinations on the GPU node.
    """
    try:
        import torch
        import torchaudio
    except ImportError as exc:
        raise RuntimeError(
            "torchaudio is required for pyannote diarization audio loading"
        ) from exc

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.dtype != torch.float32:
        waveform = waveform.to(torch.float32)
    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }


def _required_pyannote_repos(pyannote_model: str) -> list[str]:
    repos = [pyannote_model]
    if pyannote_model == "pyannote/speaker-diarization-3.1":
        repos.append("pyannote/segmentation-3.0")
    return repos


def _validate_pyannote_repo_access(hf_token: str, pyannote_model: str) -> None:
    """Verify the token can access all gated repos needed by pyannote."""
    try:
        from huggingface_hub import model_info
    except ImportError:
        return

    for repo_id in _required_pyannote_repos(pyannote_model):
        if repo_id in _validated_repo_access:
            continue
        try:
            model_info(repo_id=repo_id, token=hf_token)
        except Exception as exc:
            error_name = exc.__class__.__name__
            if error_name in {"GatedRepoError", "RepositoryNotFoundError", "HfHubHTTPError"}:
                raise RuntimeError(
                    "Hugging Face access is not ready for pyannote diarization. "
                    f"Ensure your token can access {repo_id} and accept the usage conditions at "
                    f"https://huggingface.co/{repo_id} . "
                    "For speaker-diarization-3.1 you must also accept the linked "
                    "pyannote/segmentation-3.0 model."
                ) from exc
            raise
        _validated_repo_access.add(repo_id)


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
            _validate_pyannote_repo_access(hf_token, pyannote_model)
            _ensure_hf_hub_legacy_auth_alias()
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError(
                "pyannote.audio is not installed. "
                "Add pyannote.audio to requirements.txt and rebuild the image."
            ) from exc

        logger.info("Loading %s — first run downloads model weights", pyannote_model)
        from_pretrained_kwargs = {}
        from_pretrained_params = inspect.signature(Pipeline.from_pretrained).parameters
        if "token" in from_pretrained_params:
            from_pretrained_kwargs["token"] = hf_token
        elif "use_auth_token" in from_pretrained_params:
            from_pretrained_kwargs["use_auth_token"] = hf_token

        pipeline = Pipeline.from_pretrained(
            pyannote_model,
            **from_pretrained_kwargs,
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

    audio_input = _load_audio_for_pyannote(audio_path)

    with _inference_lock:
        result = pipeline(audio_input, **kwargs)
        _last_used = time.monotonic()

    return [
        (round(turn.start, 3), round(turn.end, 3), speaker)
        for turn, _, speaker in result.itertracks(yield_label=True)
    ]
