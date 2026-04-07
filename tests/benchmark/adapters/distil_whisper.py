"""distil-whisper adapter for the STT benchmark harness (issue #9).

Wraps ``distil-large-v3`` (and other distil-whisper variants) via the
HuggingFace ``transformers`` pipeline.  Expected characteristics:

* 5-6× faster than full Whisper large-v3
* Slightly lower accuracy (~3-5% WER increase on clean speech)
* Significantly lower VRAM (~1.5 GB vs ~3 GB for large-v3)
* Good candidate for diarization mode where Whisper runs alongside pyannote

Install: ``pip install transformers accelerate torch``

A faster-whisper backend variant is also available for environments that
already have ``faster-whisper`` installed (set ``backend="faster-whisper"``).

Usage
-----
::

    from tests.benchmark.adapters.distil_whisper import DistilWhisperAdapter

    # HuggingFace transformers backend (default)
    adapter = DistilWhisperAdapter(model_id="distil-whisper/distil-large-v3")
    load_time = adapter.load()
    result = adapter.transcribe("audio.wav")
    adapter.unload()

    # faster-whisper backend
    adapter = DistilWhisperAdapter(
        model_id="distil-large-v3",
        backend="faster-whisper",
    )
"""
from __future__ import annotations

import time
from typing import List, Optional

from .base import AdapterResult, BaseAdapter, FeatureSupport, Segment, WordTimestamp

_HF_BACKEND = "huggingface"
_FW_BACKEND = "faster-whisper"


class DistilWhisperAdapter(BaseAdapter):
    """Adapter for ``distil-whisper`` models.

    Supports two backends:

    * ``"huggingface"`` (default): Uses the HuggingFace ``transformers``
      ``AutoModelForSpeechSeq2Seq`` pipeline.  Best compatibility and access
      to the latest distil-whisper releases.
    * ``"faster-whisper"``: Uses ``faster-whisper``'s CTranslate2 backend
      with the ``distil-large-v3`` model size.  Fastest option when CTranslate2
      is already available.

    Args:
        model_id: HuggingFace model identifier (e.g.
            ``"distil-whisper/distil-large-v3"``) for the ``"huggingface"``
            backend, **or** the faster-whisper model size string (e.g.
            ``"distil-large-v3"``) for the ``"faster-whisper"`` backend.
        backend: One of ``"huggingface"`` or ``"faster-whisper"``.
        device: ``"cuda"``, ``"cpu"``, or ``None`` to auto-detect.
        torch_dtype: PyTorch dtype for the HuggingFace backend.  Defaults to
            ``"float16"`` on CUDA and ``"float32"`` on CPU.
        download_root: Model cache directory.  Used by the faster-whisper
            backend; HuggingFace uses ``HF_HOME`` / ``TRANSFORMERS_CACHE``.
    """

    name = "distil-whisper"
    features = FeatureSupport(
        language_detection=True,
        word_timestamps=True,
        translation=False,
        diarization_ready=True,
    )

    def __init__(
        self,
        model_id: str = "distil-whisper/distil-large-v3",
        backend: str = _HF_BACKEND,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        download_root: str = "/data/models",
    ) -> None:
        if backend not in (_HF_BACKEND, _FW_BACKEND):
            raise ValueError(
                f"backend must be '{_HF_BACKEND}' or '{_FW_BACKEND}', got {backend!r}"
            )
        self.model_id = model_id
        self.backend = backend
        self.device = device
        self.torch_dtype = torch_dtype
        self.download_root = download_root
        self._model = None
        self._pipeline = None

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
        """Load the distil-whisper model and return the load time in seconds."""
        t0 = time.monotonic()
        if self.backend == _FW_BACKEND:
            self._load_faster_whisper()
        else:
            self._load_huggingface()
        return time.monotonic() - t0

    def _load_faster_whisper(self) -> None:
        from faster_whisper import WhisperModel  # noqa: PLC0415

        device = self._resolve_device()
        compute_type = "float16" if device == "cuda" else "int8"
        self._model = WhisperModel(
            self.model_id,
            device=device,
            compute_type=compute_type,
            download_root=self.download_root,
        )

    def _load_huggingface(self) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline  # noqa: PLC0415

        device = self._resolve_device()

        if self.torch_dtype is not None:
            dtype = getattr(torch, self.torch_dtype)
        else:
            dtype = torch.float16 if device == "cuda" else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(self.model_id)

        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=device,
        )

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> AdapterResult:
        """Transcribe *audio_path* using distil-whisper.

        Args:
            audio_path: Path to the audio file.
            language: ISO 639-1 code, or ``None`` for auto-detection.

        Returns:
            :class:`AdapterResult` with text, segments, and timing.

        Raises:
            RuntimeError: If :meth:`load` has not been called first.
        """
        if self.backend == _FW_BACKEND:
            return self._transcribe_faster_whisper(audio_path, language)
        return self._transcribe_huggingface(audio_path, language)

    def _transcribe_faster_whisper(
        self,
        audio_path: str,
        language: Optional[str],
    ) -> AdapterResult:
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

        return AdapterResult(
            text=" ".join(full_text_parts),
            segments=segments,
            processing_time=round(time.monotonic() - t0, 3),
            language=info.language,
            language_probability=round(info.language_probability, 3),
            duration=round(info.duration, 3),
        )

    def _transcribe_huggingface(
        self,
        audio_path: str,
        language: Optional[str],
    ) -> AdapterResult:
        if self._pipeline is None:
            raise RuntimeError(
                "Pipeline not loaded — call load() before transcribe()"
            )

        generate_kwargs: dict = {
            "return_timestamps": True,
        }
        if language:
            generate_kwargs["language"] = language

        t0 = time.monotonic()
        raw = self._pipeline(
            audio_path,
            generate_kwargs=generate_kwargs,
        )
        processing_time = round(time.monotonic() - t0, 3)

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
        """Release model resources and free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
