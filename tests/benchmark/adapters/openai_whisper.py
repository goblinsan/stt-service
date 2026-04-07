"""openai-whisper adapter for the STT benchmark harness (issue #8).

Wraps the original OpenAI ``whisper`` package (PyTorch implementation) as a
:class:`BaseAdapter`.  This serves as the **reference accuracy** baseline —
slower than faster-whisper or whisper.cpp but the canonical implementation.

Install: ``pip install openai-whisper``

Supported models: ``large-v3``, ``large-v2``, ``medium``, ``small``, ``base``,
``tiny`` (and their ``.en`` English-only variants).

Usage
-----
::

    from tests.benchmark.adapters.openai_whisper import OpenAIWhisperAdapter

    adapter = OpenAIWhisperAdapter(model_size="large-v3", device="cuda")
    load_time = adapter.load()
    result = adapter.transcribe("audio.wav", language="en")
    adapter.unload()
"""
from __future__ import annotations

import time
from typing import List, Optional

from .base import AdapterResult, BaseAdapter, FeatureSupport, Segment, WordTimestamp


class OpenAIWhisperAdapter(BaseAdapter):
    """Adapter wrapping the original ``openai-whisper`` PyTorch implementation.

    Args:
        model_size: Whisper model size (e.g. ``"large-v3"``, ``"medium"``).
        device: PyTorch device string — ``"cuda"``, ``"cpu"``, or ``None``
            to auto-detect (uses CUDA when available).
        download_root: Directory where models are cached.  Defaults to the
            standard whisper cache directory when ``None``.
        fp16: Whether to use FP16 inference on GPU.  Has no effect on CPU.
    """

    name = "openai-whisper"
    features = FeatureSupport(
        language_detection=True,
        word_timestamps=True,
        translation=True,
        diarization_ready=True,
    )

    def __init__(
        self,
        model_size: str = "large-v3",
        device: Optional[str] = None,
        download_root: Optional[str] = None,
        fp16: bool = True,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.download_root = download_root
        self.fp16 = fp16
        self._model = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_device(self) -> str:
        if self.device is not None:
            return self.device
        try:
            import torch  # noqa: PLC0415

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def load(self) -> float:
        """Load the openai-whisper model and return the load time in seconds.

        Raises:
            ImportError: If ``openai-whisper`` is not installed.
        """
        import whisper  # noqa: PLC0415

        device = self._resolve_device()
        t0 = time.monotonic()
        kwargs: dict = {"device": device}
        if self.download_root is not None:
            kwargs["download_root"] = self.download_root
        self._model = whisper.load_model(self.model_size, **kwargs)
        return time.monotonic() - t0

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> AdapterResult:
        """Transcribe *audio_path* using openai-whisper.

        Args:
            audio_path: Path to the audio file.
            language: ISO 639-1 code, or ``None`` for auto-detection.

        Returns:
            :class:`AdapterResult` with text, segments, and language metadata.

        Raises:
            RuntimeError: If :meth:`load` has not been called first.
        """
        if self._model is None:
            raise RuntimeError(
                "Model not loaded — call load() before transcribe()"
            )

        device = self._resolve_device()
        use_fp16 = self.fp16 and device != "cpu"

        t0 = time.monotonic()
        raw = self._model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            fp16=use_fp16,
            verbose=False,
        )
        processing_time = round(time.monotonic() - t0, 3)

        segments: List[Segment] = []
        full_text_parts: List[str] = []

        for idx, seg in enumerate(raw.get("segments", [])):
            text = seg.get("text", "").strip()

            words: Optional[List[WordTimestamp]] = None
            raw_words = seg.get("words")
            if raw_words:
                words = [
                    WordTimestamp(
                        word=w.get("word", "").strip(),
                        start=round(w.get("start", 0.0), 3),
                        end=round(w.get("end", 0.0), 3),
                        probability=round(w.get("probability", 1.0), 3),
                    )
                    for w in raw_words
                ]

            segments.append(
                Segment(
                    id=idx,
                    start=round(seg.get("start", 0.0), 3),
                    end=round(seg.get("end", 0.0), 3),
                    text=text,
                    words=words,
                )
            )
            full_text_parts.append(text)

        detected_language: Optional[str] = raw.get("language")
        # openai-whisper does not expose a per-inference language probability;
        # we set it to None as the field is optional in AdapterResult.
        duration: Optional[float] = None
        if raw.get("segments"):
            last_seg = raw["segments"][-1]
            duration = round(last_seg.get("end", 0.0), 3)

        return AdapterResult(
            text=raw.get("text", "").strip(),
            segments=segments,
            processing_time=processing_time,
            language=detected_language,
            duration=duration,
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
