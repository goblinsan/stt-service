"""Full benchmark suite orchestrator (issue #12).

Runs every adapter × model configuration against every audio sample in the
manifest, clears GPU memory between runs, handles failures gracefully, and
persists results to JSON files for downstream report generation.

Usage (CLI)
-----------
::

    python -m tests.benchmark.suite \\
        --data-dir  tests/benchmark/data \\
        --output    tests/benchmark/results \\
        --device    cuda

Usage (library)
---------------
::

    from tests.benchmark.suite import FullBenchmarkSuite, AdapterConfig
    from tests.benchmark.adapters import FasterWhisperAdapter

    configs = [
        AdapterConfig(
            adapter_cls=FasterWhisperAdapter,
            model_size="large-v3",
            device="cuda",
        ),
    ]
    suite = FullBenchmarkSuite(
        configs=configs,
        data_dir="tests/benchmark/data",
        output_dir="tests/benchmark/results",
    )
    suite_result = suite.run()
    print(f"Completed {suite_result.total_runs} runs, "
          f"{len(suite_result.failures)} failures")
"""
from __future__ import annotations

import gc
import json
import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .adapters.base import BaseAdapter
from .metrics import MetricsCollector
from .runner import BenchmarkRunner, RunResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AdapterConfig:
    """Specifies one adapter + model combination to include in the suite.

    Args:
        adapter_cls: The :class:`~tests.benchmark.adapters.base.BaseAdapter`
            subclass to instantiate.
        model_size: Model size / identifier passed to the adapter constructor
            as a keyword argument.  The exact keyword depends on the adapter
            (e.g. ``model_size`` for faster-whisper, ``model_id`` for
            distil-whisper, ``model_path`` for whisper.cpp).
        model_kwarg: Name of the constructor keyword that receives
            *model_size*.  Defaults to ``"model_size"``.
        device: Device string (``"cuda"``, ``"cpu"``).  Passed to the adapter
            constructor when the adapter accepts a ``device`` parameter.
        extra_kwargs: Additional keyword arguments forwarded verbatim to the
            adapter constructor.
    """

    adapter_cls: Type[BaseAdapter]
    model_size: str
    model_kwarg: str = "model_size"
    device: str = "cuda"
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def build(self) -> BaseAdapter:
        """Instantiate and return the adapter."""
        kwargs: Dict[str, Any] = {self.model_kwarg: self.model_size, **self.extra_kwargs}
        # Inject device only when the adapter accepts it.
        try:
            return self.adapter_cls(device=self.device, **kwargs)
        except TypeError:
            return self.adapter_cls(**kwargs)


# ---------------------------------------------------------------------------
# Result / failure dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RunFailure:
    """Records a single run that raised an exception.

    Attributes:
        adapter_name: Human-readable engine name.
        model_size: Model variant that was running.
        sample_id: Dataset sample ID.
        audio_path: Path to the audio file.
        error_type: Exception class name.
        error_message: ``str(exception)`` — first line.
        traceback: Full traceback as a string.
    """

    adapter_name: str
    model_size: str
    sample_id: str
    audio_path: str
    error_type: str
    error_message: str
    traceback: str


@dataclass
class SuiteResult:
    """Aggregated outcome of a full benchmark suite run.

    Attributes:
        total_runs: Total number of (adapter, model, sample) combinations
            attempted.
        successful_runs: Number that completed without an exception.
        failures: List of :class:`RunFailure` objects for every exception
            that was caught and handled.
        results_by_adapter: Maps ``"<adapter_name>/<model_size>"`` to the
            list of :class:`~tests.benchmark.runner.RunResult` objects for
            that combination.
    """

    total_runs: int = 0
    successful_runs: int = 0
    failures: List[RunFailure] = field(default_factory=list)
    results_by_adapter: Dict[str, List[RunResult]] = field(default_factory=dict)

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    @property
    def success_rate(self) -> float:
        if self.total_runs == 0:
            return 0.0
        return self.successful_runs / self.total_runs


# ---------------------------------------------------------------------------
# GPU memory helpers
# ---------------------------------------------------------------------------

def _clear_gpu_memory() -> None:
    """Release unused GPU memory (best-effort).

    Calls ``torch.cuda.empty_cache()`` if CUDA is available, then triggers a
    CPython garbage-collection cycle to drop any unreferenced Python objects
    that hold GPU tensors.
    """
    gc.collect()
    try:
        # Deferred import: torch is optional; omitting it lets the benchmark
        # run on CPU-only machines without installing PyTorch.
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — graceful no-op when torch is absent
        pass


# ---------------------------------------------------------------------------
# Suite orchestrator
# ---------------------------------------------------------------------------

class FullBenchmarkSuite:
    """Runs all adapter × model × sample combinations and collects results.

    Args:
        configs: List of :class:`AdapterConfig` entries to benchmark.
        data_dir: Path to the directory containing the manifest and audio
            files.
        output_dir: Directory where per-adapter JSON result files are
            written.  Created if it does not exist.
        vram_poll_interval: Interval (seconds) between VRAM samples during
            each inference run.
        save_on_adapter_complete: When ``True`` (default), each adapter's
            aggregated metrics are saved to JSON as soon as all of its runs
            are finished.  This means partial results survive an early abort.
        manifest_filename: Name of the manifest JSON file inside *data_dir*.
    """

    def __init__(
        self,
        configs: List[AdapterConfig],
        data_dir: str | Path = "tests/benchmark/data",
        output_dir: str | Path = "tests/benchmark/results",
        vram_poll_interval: float = 0.1,
        save_on_adapter_complete: bool = True,
        manifest_filename: str = "manifest.json",
    ) -> None:
        self.configs = configs
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.vram_poll_interval = vram_poll_interval
        self.save_on_adapter_complete = save_on_adapter_complete
        self.manifest_filename = manifest_filename

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> List[dict]:
        manifest_path = self.data_dir / self.manifest_filename
        with manifest_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data["samples"]

    def _reference_text(self, sample: dict) -> Optional[str]:
        """Return the reference transcript for a sample, or ``None``."""
        ref_filename = sample.get("reference")
        if not ref_filename:
            return None
        ref_path = self.data_dir / ref_filename
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
        return None

    def _run_config(
        self,
        config: AdapterConfig,
        samples: List[dict],
        suite_result: SuiteResult,
    ) -> None:
        """Execute all samples for one *config* and update *suite_result*."""
        key = f"{config.adapter_cls.name}/{config.model_size}"
        adapter: Optional[BaseAdapter] = None
        run_results: List[RunResult] = []
        runner = BenchmarkRunner(vram_poll_interval=self.vram_poll_interval)
        model_load_time: Optional[float] = None

        # ---- Load model -----------------------------------------------
        try:
            adapter = config.build()
            logger.info("Loading %s / %s …", adapter.name, config.model_size)
            model_load_time = adapter.load()
            logger.info(
                "  Model loaded in %.1f s", model_load_time
            )
        except Exception as exc:  # noqa: BLE001 — any load failure is recorded and skipped
            _record_failure(
                suite_result,
                adapter_name=config.adapter_cls.name,
                model_size=config.model_size,
                sample_id="<model_load>",
                audio_path="",
                exc=exc,
            )
            logger.warning(
                "FAILED to load %s/%s: %s",
                config.adapter_cls.name,
                config.model_size,
                exc,
            )
            return

        # ---- Iterate over samples -------------------------------------
        for sample in samples:
            suite_result.total_runs += 1
            sample_id: str = sample["id"]
            audio_path = str(self.data_dir / sample["filename"])
            reference = self._reference_text(sample)

            try:
                logger.info(
                    "  Running %s on %s …", adapter.name, sample_id
                )
                result = runner.run(adapter, audio_path, reference)
                run_results.append(result)
                suite_result.successful_runs += 1
                logger.info(
                    "    WER=%.3f  RTF=%.2fx  wall=%.2fs",
                    result.wer if result.wer is not None else float("nan"),
                    result.rtf if result.rtf is not None else float("nan"),
                    result.wall_time,
                )
            except Exception as exc:  # noqa: BLE001 — per-sample failures are recorded and skipped
                _record_failure(
                    suite_result,
                    adapter_name=adapter.name,
                    model_size=config.model_size,
                    sample_id=sample_id,
                    audio_path=audio_path,
                    exc=exc,
                )
                logger.warning(
                    "    FAILED %s on %s: %s",
                    adapter.name,
                    sample_id,
                    exc,
                )
            finally:
                # Always attempt GPU memory cleanup between samples.
                _clear_gpu_memory()

        # ---- Unload model / release GPU memory -----------------------
        try:
            adapter.unload()
        except Exception:  # noqa: BLE001
            pass
        _clear_gpu_memory()

        # ---- Save partial results ------------------------------------
        suite_result.results_by_adapter[key] = run_results

        if self.save_on_adapter_complete and run_results:
            self._save_adapter_results(
                adapter=adapter,
                config=config,
                run_results=run_results,
                model_load_time=model_load_time,
            )

    def _save_adapter_results(
        self,
        adapter: BaseAdapter,
        config: AdapterConfig,
        run_results: List[RunResult],
        model_load_time: Optional[float],
    ) -> None:
        collector = MetricsCollector(results=run_results)
        metrics = collector.aggregate(
            engine=adapter.name,
            model=config.model_size,
            results=run_results,
            model_load_time=model_load_time,
            language_detection=adapter.features.language_detection,
            word_timestamps=adapter.features.word_timestamps,
            translation=adapter.features.translation,
            diarization_ready=adapter.features.diarization_ready,
        )
        safe_name = adapter.name.replace("/", "_").replace(" ", "_")
        safe_model = config.model_size.replace("/", "_").replace(" ", "_")
        out_path = self.output_dir / f"{safe_name}_{safe_model}.json"
        collector.save_json(out_path, [metrics])
        logger.info("  Saved results → %s", out_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> SuiteResult:
        """Execute the full suite and return a :class:`SuiteResult`.

        Iterates over :attr:`configs` in order.  For each configuration it:

        1. Loads the model.
        2. Runs inference against every sample in the manifest.
        3. Unloads the model and clears GPU memory.
        4. Optionally saves aggregated metrics to JSON.

        Any exception during model loading or per-sample inference is caught,
        logged, and recorded in :attr:`SuiteResult.failures`.  The suite
        continues with the next configuration / sample regardless.

        Returns:
            :class:`SuiteResult` containing counts, failures, and per-adapter
            run results.
        """
        samples = self._load_manifest()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        suite_result = SuiteResult()

        for config in self.configs:
            self._run_config(config, samples, suite_result)

        self._save_failure_log(suite_result)

        logger.info(
            "Suite complete — %d/%d runs successful, %d failures",
            suite_result.successful_runs,
            suite_result.total_runs,
            suite_result.failure_count,
        )
        return suite_result

    def _save_failure_log(self, suite_result: SuiteResult) -> None:
        """Write failures to ``<output_dir>/failures.json`` when non-empty."""
        if not suite_result.failures:
            return
        failure_path = self.output_dir / "failures.json"
        data = {
            "total_runs": suite_result.total_runs,
            "successful_runs": suite_result.successful_runs,
            "failure_count": suite_result.failure_count,
            "failures": [
                {
                    "adapter_name": f.adapter_name,
                    "model_size": f.model_size,
                    "sample_id": f.sample_id,
                    "audio_path": f.audio_path,
                    "error_type": f.error_type,
                    "error_message": f.error_message,
                    "traceback": f.traceback,
                }
                for f in suite_result.failures
            ],
        }
        failure_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info(
            "Failure log written → %s (%d failures)",
            failure_path,
            suite_result.failure_count,
        )


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _record_failure(
    suite_result: SuiteResult,
    adapter_name: str,
    model_size: str,
    sample_id: str,
    audio_path: str,
    exc: Exception,
) -> None:
    suite_result.failures.append(
        RunFailure(
            adapter_name=adapter_name,
            model_size=model_size,
            sample_id=sample_id,
            audio_path=audio_path,
            error_type=type(exc).__name__,
            error_message=str(exc).splitlines()[0] if str(exc) else repr(exc),
            traceback=traceback.format_exc(),
        )
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """CLI entry-point for the full benchmark suite.

    Runs a default set of :class:`AdapterConfig` objects — one per engine ×
    model combination.  Pass ``--help`` for all options.
    """
    import argparse
    import sys

    from .adapters import (
        DistilWhisperAdapter,
        FasterWhisperAdapter,
        OpenAIWhisperAdapter,
        WhisperCppAdapter,
    )

    parser = argparse.ArgumentParser(
        description="Run the full STT benchmark suite (issue #12)",
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        default=Path("tests/benchmark/data"),
        help="Directory containing manifest.json and audio files",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("tests/benchmark/results"),
        help="Directory where per-adapter JSON result files are saved",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Compute device (default: cuda)",
    )
    args = parser.parse_args(argv)

    device: str = args.device
    if device == "auto":
        try:
            import torch  # noqa: PLC0415
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    configs: List[AdapterConfig] = [
        # faster-whisper — large-v3 and medium
        AdapterConfig(FasterWhisperAdapter, "large-v3", device=device),
        AdapterConfig(FasterWhisperAdapter, "medium", device=device),
        # distil-whisper via faster-whisper backend
        AdapterConfig(
            DistilWhisperAdapter,
            "distil-large-v3",
            model_kwarg="model_id",
            device=device,
            extra_kwargs={"backend": "faster-whisper"},
        ),
        # openai-whisper — large-v3 and medium
        AdapterConfig(OpenAIWhisperAdapter, "large-v3", device=device),
        AdapterConfig(OpenAIWhisperAdapter, "medium", device=device),
        # whisper.cpp — model paths expected under /data/models/whisper-cpp/
        AdapterConfig(
            WhisperCppAdapter,
            "/data/models/whisper-cpp/ggml-large-v3.bin",
            model_kwarg="model_path",
            device=device,
        ),
    ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    suite = FullBenchmarkSuite(
        configs=configs,
        data_dir=args.data_dir,
        output_dir=args.output,
    )
    result = suite.run()

    print(
        f"\nSuite complete — {result.successful_runs}/{result.total_runs} "
        f"runs successful, {result.failure_count} failures"
    )
    if result.failures:
        print(f"Failure log: {args.output / 'failures.json'}")
    return 0 if result.failure_count == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
