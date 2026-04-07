"""Engine adapter interface for the STT benchmark harness.

Built-in adapters (each wraps a specific STT engine):

* :class:`~tests.benchmark.adapters.faster_whisper.FasterWhisperAdapter`
  — faster-whisper (CTranslate2), the production baseline (issue #6).
* :class:`~tests.benchmark.adapters.whisper_cpp.WhisperCppAdapter`
  — whisper.cpp via ``pywhispercpp`` (issue #7).
* :class:`~tests.benchmark.adapters.openai_whisper.OpenAIWhisperAdapter`
  — original OpenAI ``openai-whisper`` PyTorch reference (issue #8).
* :class:`~tests.benchmark.adapters.distil_whisper.DistilWhisperAdapter`
  — distil-large-v3 via HuggingFace or faster-whisper backend (issue #9).
* :class:`~tests.benchmark.adapters.whisper_jax.WhisperJaxAdapter`
  — JAX/XLA-optimised Whisper (optional, issue #10).
"""

from .base import AdapterResult, BaseAdapter, FeatureSupport, Segment, WordTimestamp
from .distil_whisper import DistilWhisperAdapter
from .faster_whisper import FasterWhisperAdapter
from .openai_whisper import OpenAIWhisperAdapter
from .whisper_cpp import WhisperCppAdapter
from .whisper_jax import WhisperJaxAdapter

__all__ = [
    # Base types
    "AdapterResult",
    "BaseAdapter",
    "FeatureSupport",
    "Segment",
    "WordTimestamp",
    # Concrete adapters
    "DistilWhisperAdapter",
    "FasterWhisperAdapter",
    "OpenAIWhisperAdapter",
    "WhisperCppAdapter",
    "WhisperJaxAdapter",
]
