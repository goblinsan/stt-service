"""faster-whisper adapter for the STT benchmark harness (issue #6).

Wraps ``faster-whisper`` (CTranslate2-based) as a :class:`BaseAdapter`.
This is the **baseline** adapter — all other engines are compared against it.

Supported models: ``large-v3``, ``medium``, ``small``, ``distil-large-v3``.

Usage
-----
::

    from tests.benchmark.adapters.faster_whisper import FasterWhisperAdapter

    adapter = FasterWhisperAdapter(model_size="large-v3", device="cuda")
    load_time = adapter.load()
    result = adapter.transcribe("audio.wav", language="en")
    adapter.unload()
"""
from __future__ import annotations

import time
from typing import List, Optional

from .base import AdapterResult, BaseAdapter, FeatureSupport, Segment, WordTimestamp


class FasterWhisperAdapter(BaseAdapter):
    """Adapter wrapping ``faster-whisper`` (CTranslate2 backend).

    Args:
        model_size: One of ``"large-v3"``, ``"medium"``, ``"small"``,
            ``"distil-large-v3"``, etc.
        device: ``"cuda"``, ``"cpu"``, or ``"auto"`` (auto-detects GPU).
        compute_type: CTranslate2 quantization type.  Defaults to
            ``"float16"`` on CUDA and ``"int8"`` on CPU when ``"auto"``.
        download_root: Directory where models are cached.  Defaults to
            the value used by the production stt-api service (``/data/models``).
    """

    name = "faster-whisper"
    features = FeatureSupport(
        language_detection=True,
        word_timestamps=True,
        translation=True,
        diarization_ready=True,
    )

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "auto",
        download_root: str = "/data/models",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self._model = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_device(self) -> tuple[str, str]:
        device = self.device
        compute_type = self.compute_type

        if device == "auto":
            try:
                import torch  # noqa: PLC0415

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        return device, compute_type

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def load(self) -> float:
        """Load the faster-whisper model and return the load time in seconds."""
        from faster_whisper import WhisperModel  # noqa: PLC0415

        device, compute_type = self._resolve_device()
        t0 = time.monotonic()
        self._model = WhisperModel(
            self.model_size,
            device=device,
            compute_type=compute_type,
            download_root=self.download_root,
        )
        return time.monotonic() - t0

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> AdapterResult:
        """Transcribe *audio_path* using faster-whisper.

        Args:
            audio_path: Path to the audio file.
            language: ISO 639-1 code, or ``None`` for auto-detection.

        Returns:
            :class:`AdapterResult` with text, segments, timing, and language
            metadata populated.

        Raises:
            RuntimeError: If :meth:`load` has not been called first.
        """
        if self._model is None:
            raise RuntimeError(
                "Model not loaded — call load() before transcribe()"
            )

        t0 = time.monotonic()

        segments_iter, info = self._model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        segments: List[Segment] = []
        full_text_parts: List[str] = []

        for seg in segments_iter:
            words = None
            if seg.words:
                words = [
                    WordTimestamp(
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

        processing_time = round(time.monotonic() - t0, 3)

        return AdapterResult(
            text=" ".join(full_text_parts),
            segments=segments,
            processing_time=processing_time,
            language=info.language,
            language_probability=round(info.language_probability, 3),
            duration=round(info.duration, 3),
        )

    def unload(self) -> None:
        """Release the model and free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch  # noqa: PLC0415

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
