"""whisper.cpp adapter for the STT benchmark harness (issue #7).

Wraps ``whisper.cpp`` via the ``pywhispercpp`` Python bindings.
GGUF-quantized models (Q5_0, Q8_0, etc.) are supported, which may
outperform faster-whisper on inference speed with lower VRAM usage.

Install: ``pip install pywhispercpp``

Usage
-----
::

    from tests.benchmark.adapters.whisper_cpp import WhisperCppAdapter

    adapter = WhisperCppAdapter(model_path="/data/models/ggml/ggml-large-v3-q5_0.bin")
    load_time = adapter.load()
    result = adapter.transcribe("audio.wav", language="en")
    adapter.unload()
"""
from __future__ import annotations

import time
from typing import List, Optional

from .base import AdapterResult, BaseAdapter, FeatureSupport, Segment, WordTimestamp


class WhisperCppAdapter(BaseAdapter):
    """Adapter wrapping ``whisper.cpp`` via ``pywhispercpp``.

    Args:
        model_path: Absolute path to a GGUF/GGML model file, **or** a
            model name accepted by ``pywhispercpp`` (e.g. ``"large"``).
            GGUF-quantized variants (Q5_0, Q8_0) give the best speed/quality
            trade-off on a single GPU.
        n_threads: Number of CPU threads to use during decoding.
            Defaults to 4.
        use_gpu: Whether to enable GPU offloading (CUDA / Metal).
            Defaults to ``True``.
    """

    name = "whisper.cpp"
    features = FeatureSupport(
        language_detection=True,
        word_timestamps=True,
        translation=True,
        diarization_ready=False,
    )

    def __init__(
        self,
        model_path: str = "large",
        n_threads: int = 4,
        use_gpu: bool = True,
    ) -> None:
        self.model_path = model_path
        self.n_threads = n_threads
        self.use_gpu = use_gpu
        self._model = None

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def load(self) -> float:
        """Load the whisper.cpp model and return the load time in seconds.

        Raises:
            ImportError: If ``pywhispercpp`` is not installed.
        """
        from pywhispercpp.model import Model  # noqa: PLC0415

        t0 = time.monotonic()
        self._model = Model(
            self.model_path,
            n_threads=self.n_threads,
        )
        return time.monotonic() - t0

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> AdapterResult:
        """Transcribe *audio_path* using whisper.cpp.

        Args:
            audio_path: Path to the audio file.  whisper.cpp requires
                16 kHz mono WAV input; the binding handles conversion for
                common formats automatically.
            language: ISO 639-1 code (e.g. ``"en"``), or ``None`` for
                automatic language detection.

        Returns:
            :class:`AdapterResult` with text, segments, and timing.

        Raises:
            RuntimeError: If :meth:`load` has not been called first.
        """
        if self._model is None:
            raise RuntimeError(
                "Model not loaded — call load() before transcribe()"
            )

        # Build pywhispercpp transcription parameters.
        params: dict = {}
        if language:
            params["language"] = language

        t0 = time.monotonic()
        raw_segments = self._model.transcribe(audio_path, **params)
        processing_time = round(time.monotonic() - t0, 3)

        segments: List[Segment] = []
        full_text_parts: List[str] = []

        for idx, seg in enumerate(raw_segments):
            # pywhispercpp segment attributes: t0, t1, text (+ optional words)
            start = round(getattr(seg, "t0", 0) / 100.0, 3)
            end = round(getattr(seg, "t1", 0) / 100.0, 3)
            text = seg.text.strip() if hasattr(seg, "text") else str(seg).strip()

            words: Optional[List[WordTimestamp]] = None
            raw_words = getattr(seg, "tokens", None) or getattr(seg, "words", None)
            if raw_words:
                try:
                    words = [
                        WordTimestamp(
                            word=getattr(w, "text", str(w)).strip(),
                            start=round(getattr(w, "t0", 0) / 100.0, 3),
                            end=round(getattr(w, "t1", 0) / 100.0, 3),
                            probability=round(getattr(w, "p", 1.0), 3),
                        )
                        for w in raw_words
                    ]
                except Exception:  # noqa: BLE001
                    words = None

            segments.append(Segment(id=idx, start=start, end=end, text=text, words=words))
            full_text_parts.append(text)

        # pywhispercpp does not expose a duration field on the result;
        # approximate it from the last segment end time.
        duration: Optional[float] = segments[-1].end if segments else None

        # Language detection result is available on the model instance after
        # transcription.
        detected_language: Optional[str] = None
        try:
            detected_language = self._model.lang_str
        except AttributeError:
            pass

        return AdapterResult(
            text=" ".join(full_text_parts),
            segments=segments,
            processing_time=processing_time,
            language=detected_language or language,
            duration=duration,
        )

    def unload(self) -> None:
        """Release model resources."""
        if self._model is not None:
            del self._model
            self._model = None
