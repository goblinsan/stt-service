"""Engine adapter interface for the STT benchmark harness."""

from .base import AdapterResult, BaseAdapter, FeatureSupport, Segment, WordTimestamp

__all__ = [
    "AdapterResult",
    "BaseAdapter",
    "FeatureSupport",
    "Segment",
    "WordTimestamp",
]
