"""Abstract base class for STT engine adapters.

Each adapter wraps an STT engine behind a common interface::

    transcribe(audio_path, language=None) → AdapterResult

Adapters handle engine-specific setup, model loading, and teardown.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class WordTimestamp:
    """Word-level timestamp from a transcription."""

    word: str
    start: float
    end: float
    probability: float


@dataclass
class Segment:
    """A single transcription segment (sentence or phrase)."""

    id: int
    start: float
    end: float
    text: str
    words: Optional[List[WordTimestamp]] = None
    speaker: Optional[str] = None


@dataclass
class AdapterResult:
    """Result returned by an engine adapter's ``transcribe()`` method."""

    text: str
    segments: List[Segment]
    processing_time: float
    language: Optional[str] = None
    language_probability: Optional[float] = None
    duration: Optional[float] = None
    #: Optional — not all engines expose token-level streaming boundaries.
    first_token_latency: Optional[float] = None


@dataclass
class FeatureSupport:
    """Declares which optional features an adapter supports."""

    language_detection: bool = False
    word_timestamps: bool = False
    translation: bool = False
    diarization_ready: bool = False


class BaseAdapter(ABC):
    """Abstract base class for STT engine adapters.

    Sub-classes must set :attr:`name` and override :meth:`load` and
    :meth:`transcribe`.  The optional :attr:`features` dataclass declares
    which capabilities the engine supports; it is used to populate the
    feature-support matrix in benchmark reports.
    """

    #: Human-readable identifier used in reports (e.g. "faster-whisper").
    name: str = "unknown"

    #: Feature support declaration — override in subclass.
    features: FeatureSupport = FeatureSupport()

    @abstractmethod
    def load(self) -> float:
        """Load the model and return the load time in seconds.

        Called once before benchmarking begins.  Implementations should
        initialise all engine-specific resources here so that
        :meth:`transcribe` measures only inference latency.
        """
        ...

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> AdapterResult:
        """Transcribe *audio_path* and return an :class:`AdapterResult`.

        Args:
            audio_path: Absolute or relative path to the audio file.
            language: ISO 639-1 language code, or ``None`` for auto-detect.

        Returns:
            :class:`AdapterResult` populated with at minimum ``text``,
            ``segments``, and ``processing_time``.
        """
        ...

    def unload(self) -> None:
        """Release model resources (optional).

        Called after benchmarking completes.  Implementations should free
        GPU memory here so the next adapter starts with a clean slate.
        """
