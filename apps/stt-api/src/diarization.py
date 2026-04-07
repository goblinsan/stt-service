"""Speaker diarization using pyannote.audio.

Provides a thread-safe singleton loader for the pyannote
``speaker-diarization-3.1`` pipeline, mirroring the pattern used in
engine.py for the Whisper model.
"""

import logging
import threading

logger = logging.getLogger("stt.diarization")

_pipeline = None
_pipeline_lock = threading.Lock()
_inference_lock = threading.Lock()


def is_pipeline_loaded() -> bool:
    return _pipeline is not None


def get_pipeline(hf_token: str, model_cache_dir: str, pyannote_model: str):
    """Return the (singleton) pyannote diarization pipeline.

    The pipeline is loaded once and cached in-process.  The function is
    thread-safe: concurrent callers block until the first load finishes.

    Args:
        hf_token: HuggingFace auth token for gated model access.
        model_cache_dir: Base directory for model caches (``HF_HOME`` is
            set at container level to persist downloads here).
        pyannote_model: HuggingFace model ID
            (default ``pyannote/speaker-diarization-3.1``).

    Raises:
        RuntimeError: If ``pyannote.audio`` is not installed.
    """
    global _pipeline
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
        logger.info("Pyannote pipeline ready")
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

    return [
        (round(turn.start, 3), round(turn.end, 3), speaker)
        for turn, _, speaker in result.itertracks(yield_label=True)
    ]
