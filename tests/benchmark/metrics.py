"""Metrics collector for benchmark results (issue #3).

Collects :class:`~tests.benchmark.runner.RunResult` objects produced by the
benchmark runner, aggregates them per engine/model combination, and serialises
the aggregated statistics to JSON for downstream analysis and report
generation.

Usage
-----
::

    from tests.benchmark.metrics import MetricsCollector
    from tests.benchmark.runner import BenchmarkRunner

    collector = MetricsCollector()
    runner = BenchmarkRunner()

    for sample in dataset:
        result = runner.run(adapter, sample.audio_path, sample.reference)
        collector.add_result(result)

    metrics = collector.aggregate(
        engine="faster-whisper",
        model="large-v3",
        results=collector.results,
        model_load_time=load_time,
        language_detection=True,
        word_timestamps=True,
        translation=True,
        diarization_ready=True,
    )
    collector.save_json("tests/benchmark/results/faster-whisper.json", [metrics])
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from .runner import RunResult


@dataclass
class EngineMetrics:
    """Aggregated metrics for a single engine + model combination.

    Populated by :meth:`MetricsCollector.aggregate` from a list of
    :class:`~tests.benchmark.runner.RunResult` objects.
    """

    engine: str
    model: str

    # Accuracy -----------------------------------------------------------------
    wer_avg: Optional[float] = None
    wer_min: Optional[float] = None
    wer_max: Optional[float] = None
    cer_avg: Optional[float] = None

    # Latency ------------------------------------------------------------------
    wall_time_avg: float = 0.0
    wall_time_min: float = 0.0
    wall_time_max: float = 0.0
    #: Average first-token latency — ``None`` when unsupported by the engine.
    first_token_latency_avg: Optional[float] = None

    # Real-time factor ---------------------------------------------------------
    rtf_avg: Optional[float] = None

    # GPU memory ---------------------------------------------------------------
    #: Peak VRAM observed across all runs (MiB).
    vram_peak_mb: Optional[int] = None

    # Model loading ------------------------------------------------------------
    model_load_time: Optional[float] = None

    # Feature support matrix ---------------------------------------------------
    language_detection: bool = False
    word_timestamps: bool = False
    translation: bool = False
    diarization_ready: bool = False

    # Meta ---------------------------------------------------------------------
    run_count: int = 0


@dataclass
class MetricsCollector:
    """Accumulate :class:`RunResult` objects and produce aggregated metrics.

    Typical workflow::

        collector = MetricsCollector()
        # ... add results from multiple runs ...
        metrics = collector.aggregate("faster-whisper", "large-v3", collector.results)
        collector.save_json("results/faster-whisper.json", [metrics])
    """

    results: List[RunResult] = field(default_factory=list)

    def add_result(self, result: RunResult) -> None:
        """Append a single run result to the collection."""
        self.results.append(result)

    def aggregate(
        self,
        engine: str,
        model: str,
        results: List[RunResult],
        *,
        model_load_time: Optional[float] = None,
        language_detection: bool = False,
        word_timestamps: bool = False,
        translation: bool = False,
        diarization_ready: bool = False,
    ) -> EngineMetrics:
        """Aggregate a list of :class:`RunResult` objects into :class:`EngineMetrics`.

        Args:
            engine: Engine identifier (e.g. ``"faster-whisper"``).
            model: Model size / variant (e.g. ``"large-v3"``).
            results: Non-empty list of run results to aggregate.
            model_load_time: Time in seconds to load the model (measured once
                before the benchmark runs).
            language_detection: Whether the engine supports automatic language
                detection.
            word_timestamps: Whether the engine supports word-level timestamps.
            translation: Whether the engine supports translation tasks.
            diarization_ready: Whether the engine exposes word timestamps
                suitable for pairing with a diarization pipeline.

        Raises:
            ValueError: If *results* is empty.
        """
        if not results:
            raise ValueError("Cannot aggregate an empty results list")

        wers = [r.wer for r in results if r.wer is not None]
        cers = [r.cer for r in results if r.cer is not None]
        wall_times = [r.wall_time for r in results]
        rtfs = [r.rtf for r in results if r.rtf is not None]
        first_token_latencies = [
            r.first_token_latency
            for r in results
            if r.first_token_latency is not None
        ]
        vram_peaks = [r.vram_peak_mb for r in results if r.vram_peak_mb is not None]

        return EngineMetrics(
            engine=engine,
            model=model,
            wer_avg=statistics.mean(wers) if wers else None,
            wer_min=min(wers) if wers else None,
            wer_max=max(wers) if wers else None,
            cer_avg=statistics.mean(cers) if cers else None,
            wall_time_avg=statistics.mean(wall_times),
            wall_time_min=min(wall_times),
            wall_time_max=max(wall_times),
            first_token_latency_avg=(
                statistics.mean(first_token_latencies)
                if first_token_latencies
                else None
            ),
            rtf_avg=statistics.mean(rtfs) if rtfs else None,
            vram_peak_mb=max(vram_peaks) if vram_peaks else None,
            model_load_time=model_load_time,
            language_detection=language_detection,
            word_timestamps=word_timestamps,
            translation=translation,
            diarization_ready=diarization_ready,
            run_count=len(results),
        )

    def save_json(self, path: str | Path, metrics: List[EngineMetrics]) -> None:
        """Serialise a list of :class:`EngineMetrics` to a JSON file.

        The file is created (including any missing parent directories) if it
        does not already exist.  Existing files are overwritten.

        The JSON schema is::

            {
              "generated_at": "<ISO-8601 UTC timestamp>",
              "engines": [ <EngineMetrics>, ... ]
            }
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "engines": [asdict(m) for m in metrics],
        }

        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    @staticmethod
    def load_json(path: str | Path) -> dict:
        """Load a metrics JSON file previously written by :meth:`save_json`.

        Returns:
            The parsed dictionary with ``generated_at`` and ``engines`` keys.
        """
        with Path(path).open(encoding="utf-8") as fh:
            return json.load(fh)
