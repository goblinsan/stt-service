"""whisper-jax adapter for the STT benchmark harness (issue #10, optional).

Wraps ``whisper-jax`` (JAX/XLA-optimised Whisper) as a :class:`BaseAdapter`.

.. warning::
    This adapter is **optional**.  JAX + CUDA setup can be complex and may not
    work well on all GPUs (e.g. RTX 4060 with CUDA 12.x).  If importing
    ``whisper_jax`` fails, the adapter raises :class:`ImportError` with a
    helpful message pointing to the installation instructions.

Install::

    pip install whisper-jax

Optionally, install JAX with GPU support::

    pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Usage
-----
::

    from tests.benchmark.adapters.whisper_jax import WhisperJaxAdapter

    adapter = WhisperJaxAdapter(model_size="large-v2")
    load_time = adapter.load()
    result = adapter.transcribe("audio.wav")
    adapter.unload()
"""
from __future__ import annotations

import time
from typing import List, Optional

from .base import AdapterResult, BaseAdapter, FeatureSupport, Segment


class WhisperJaxAdapter(BaseAdapter):
    """Adapter wrapping ``whisper-jax`` (JAX/XLA-optimised Whisper).

    This adapter is **optional** — skip it if JAX + CUDA setup is not
    straightforward on your target hardware.

    Args:
        model_size: Whisper model size (e.g. ``"large-v2"``, ``"medium"``).
            Note: ``large-v3`` is not yet supported by whisper-jax.
        dtype: JAX dtype for inference.  Defaults to ``"bfloat16"`` for best
            performance on modern GPUs/TPUs; use ``"float32"`` for CPU.
        batch_size: Chunk batch size for long-form transcription.
    """

    name = "whisper-jax"
    features = FeatureSupport(
        language_detection=True,
        word_timestamps=False,
        translation=True,
        diarization_ready=False,
    )

    def __init__(
        self,
        model_size: str = "large-v2",
        dtype: str = "bfloat16",
        batch_size: int = 16,
    ) -> None:
        self.model_size = model_size
        self.dtype = dtype
        self.batch_size = batch_size
        self._pipeline = None

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def load(self) -> float:
        """Load the whisper-jax model and return the load time in seconds.

        Raises:
            ImportError: If ``whisper-jax`` is not installed or if JAX cannot
                be imported (e.g. incompatible CUDA version).
        """
        try:
            import jax.numpy as jnp  # noqa: PLC0415
            from whisper_jax import FlaxWhisperPipeline  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "whisper-jax is not installed. Install with:\n"
                "  pip install whisper-jax\n"
                "For GPU support, also install:\n"
                "  pip install 'jax[cuda12_pip]' -f "
                "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
            ) from exc

        dtype = getattr(jnp, self.dtype, jnp.bfloat16)

        t0 = time.monotonic()
        self._pipeline = FlaxWhisperPipeline(
            f"openai/whisper-{self.model_size}",
            dtype=dtype,
            batch_size=self.batch_size,
        )
        return time.monotonic() - t0

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> AdapterResult:
        """Transcribe *audio_path* using whisper-jax.

        Args:
            audio_path: Path to the audio file.
            language: ISO 639-1 code, or ``None`` for auto-detection.
                whisper-jax will detect the language automatically when
                this is ``None``.

        Returns:
            :class:`AdapterResult` with text, segments, and timing.
            Note: word-level timestamps are **not** supported by whisper-jax.

        Raises:
            RuntimeError: If :meth:`load` has not been called first.
        """
        if self._pipeline is None:
            raise RuntimeError(
                "Model not loaded — call load() before transcribe()"
            )

        kwargs: dict = {
            "task": "transcribe",
            "return_timestamps": True,
        }
        if language:
            kwargs["language"] = language

        t0 = time.monotonic()
        raw = self._pipeline(audio_path, **kwargs)
        processing_time = round(time.monotonic() - t0, 3)

        # whisper-jax returns {"text": ..., "chunks": [{"text": ..., "timestamp": (s, e)}, ...]}
        full_text: str = raw.get("text", "").strip()
        chunks = raw.get("chunks", [])

        segments: List[Segment] = []
        for idx, chunk in enumerate(chunks):
            ts = chunk.get("timestamp", (0.0, 0.0)) or (0.0, 0.0)
            start = round(float(ts[0] or 0.0), 3)
            end = round(float(ts[1] or 0.0), 3)
            text = chunk.get("text", "").strip()
            segments.append(Segment(id=idx, start=start, end=end, text=text))

        duration: Optional[float] = segments[-1].end if segments else None

        return AdapterResult(
            text=full_text,
            segments=segments,
            processing_time=processing_time,
            duration=duration,
        )

    def unload(self) -> None:
        """Release the JAX pipeline.

        Note: JAX does not expose an explicit memory-release API; deleting the
        pipeline object is the best we can do here.
        """
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
