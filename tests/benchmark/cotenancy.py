"""Co-tenancy benchmark module (issue #14).

Measures STT adapter performance while other GPU services occupy VRAM.
Documents whether VRAM contention causes OOM errors or merely slowdown,
and by how much inference latency increases under each contention scenario.

Usage (library)
---------------
::

    from tests.benchmark.cotenancy import CoTenancyBenchmark, CoTenancyScenario
    from tests.benchmark.adapters import FasterWhisperAdapter

    scenarios = [
        CoTenancyScenario(name="baseline",    tenant_vram_mb=0),
        CoTenancyScenario(name="pyannote",    tenant_vram_mb=1500),
        CoTenancyScenario(name="llm-idle",    tenant_vram_mb=4096),
    ]

    benchmark = CoTenancyBenchmark(
        adapter_configs=[
            {"cls": FasterWhisperAdapter, "model_size": "large-v3"},
        ],
        scenarios=scenarios,
        audio_samples=[
            {"audio_path": "tests/benchmark/data/ls_clean_en_001.wav",
             "reference": "...",
             "sample_id": "ls_clean_en_001"},
        ],
    )
    results = benchmark.run()
    report_md = benchmark.generate_report(results)
    print(report_md)

Usage (CLI)
-----------
::

    python -m tests.benchmark.cotenancy \\
        --data-dir  tests/benchmark/data \\
        --results   tests/benchmark/results \\
        --output    tests/benchmark/results/cotenancy_report.md \\
        --device    cuda
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
from .runner import BenchmarkRunner, RunResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------

class OutcomeKind:
    """String constants for co-tenancy run outcomes."""

    SUCCESS = "success"
    OOM = "oom"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Configuration / result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CoTenancyScenario:
    """Describes a single co-tenancy scenario.

    Args:
        name: Human-readable label (e.g. ``"pyannote-idle"``).
        tenant_vram_mb: How many MiB of VRAM the synthetic co-tenant should
            occupy.  Set to ``0`` for the baseline (no contention).
        description: Optional longer description of what the co-tenant
            represents.
    """

    name: str
    tenant_vram_mb: int = 0
    description: str = ""


@dataclass
class CoTenancyRunResult:
    """Result of one adapter run under one co-tenancy scenario.

    Attributes:
        adapter_name: Engine name.
        model_size: Model variant.
        sample_id: Dataset sample identifier.
        scenario_name: The :attr:`CoTenancyScenario.name` in use.
        tenant_vram_mb: Requested co-tenant VRAM (may not be fully allocated
            if the GPU ran out of memory).
        outcome: One of :data:`OutcomeKind` constants.
        wall_time: Inference wall-clock time in seconds.  ``None`` on failure.
        baseline_wall_time: Wall time from the baseline (no contention) run
            for the same adapter + sample, if available.
        slowdown_factor: ``wall_time / baseline_wall_time``.  Values > 1
            indicate slowdown; ``None`` when baseline is unavailable.
        wer: Word Error Rate.  ``None`` when not available.
        vram_peak_mb: Peak VRAM during inference.  ``None`` on CPU-only runs.
        error_type: Exception class name when *outcome* is not ``"success"``.
        error_message: Short error description.
    """

    adapter_name: str
    model_size: str
    sample_id: str
    scenario_name: str
    tenant_vram_mb: int
    outcome: str
    wall_time: Optional[float] = None
    baseline_wall_time: Optional[float] = None
    slowdown_factor: Optional[float] = None
    wer: Optional[float] = None
    vram_peak_mb: Optional[int] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class CoTenancySuiteResult:
    """Aggregated result of a full co-tenancy benchmark run.

    Attributes:
        runs: All individual :class:`CoTenancyRunResult` objects.
        total: Total runs attempted.
        success_count: Runs that completed without error.
        oom_count: Runs that hit an OOM error.
        error_count: Runs that failed with a non-OOM error.
    """

    runs: List[CoTenancyRunResult] = field(default_factory=list)
    total: int = 0
    success_count: int = 0
    oom_count: int = 0
    error_count: int = 0


# ---------------------------------------------------------------------------
# Synthetic VRAM tenant
# ---------------------------------------------------------------------------

class _VramTenant:
    """Allocates and holds a block of GPU VRAM as a synthetic co-tenant.

    On platforms without CUDA / torch the allocation silently becomes a no-op
    so that the benchmark can still run (measuring CPU impact instead).

    Args:
        vram_mb: Megabytes of VRAM to allocate.
    """

    def __init__(self, vram_mb: int) -> None:
        self._vram_mb = vram_mb
        self._tensor = None

    def __enter__(self) -> "_VramTenant":
        if self._vram_mb > 0:
            self._allocate()
        return self

    def __exit__(self, *_: Any) -> None:
        self._release()

    def _allocate(self) -> None:
        try:
            import torch  # noqa: PLC0415

            if not torch.cuda.is_available():
                return
            # Allocate a float32 tensor of roughly vram_mb MiB.
            # float32 = 4 bytes; 1 MiB = 2^20 bytes → n_elements = vram_mb * 2^20 / 4
            n_elements = self._vram_mb * (1024 * 1024) // 4
            self._tensor = torch.zeros(n_elements, dtype=torch.float32, device="cuda")
            logger.debug("Co-tenant allocated ~%d MiB of VRAM", self._vram_mb)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not allocate %d MiB VRAM for co-tenant: %s",
                self._vram_mb,
                exc,
            )
            self._tensor = None

    def _release(self) -> None:
        if self._tensor is not None:
            del self._tensor
            self._tensor = None
        gc.collect()
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _is_oom_error(exc: Exception) -> bool:
    """Return ``True`` when *exc* looks like an out-of-memory GPU error."""
    exc_type = type(exc).__name__
    msg = str(exc).lower()
    # Whole-phrase matches (not just substring to avoid false positives).
    oom_phrases = ("out of memory", "cuda out of memory", "outofmemory")
    if exc_type in ("RuntimeError", "torch.cuda.OutOfMemoryError", "OutOfMemoryError"):
        if any(phrase in msg for phrase in oom_phrases):
            return True
    return any(phrase in msg for phrase in oom_phrases)


# ---------------------------------------------------------------------------
# Co-tenancy benchmark orchestrator
# ---------------------------------------------------------------------------

class CoTenancyBenchmark:
    """Runs adapter(s) under different co-tenancy scenarios.

    For each scenario the benchmark:

    1. Allocates a synthetic VRAM block of :attr:`~CoTenancyScenario.tenant_vram_mb` MiB.
    2. Runs each adapter against each audio sample with the VRAM block held.
    3. Records whether the outcome is success, OOM, or another error.
    4. Computes slowdown relative to the baseline (zero contention) run.

    Args:
        adapter_configs: Each entry is a ``dict`` with keys ``"cls"``
            (:class:`BaseAdapter` subclass), ``"model_size"`` (str), and
            optionally ``"device"`` (str, default ``"cuda"``) and any
            adapter-specific kwargs under ``"extra_kwargs"``.
        scenarios: Ordered list of :class:`CoTenancyScenario` to test.
            **The first scenario should have** ``tenant_vram_mb=0`` **so that
            baseline timings are established.**
        audio_samples: List of sample dicts with keys ``"audio_path"``,
            ``"sample_id"``, and optionally ``"reference"`` (transcript str).
        vram_poll_interval: VRAM monitoring poll interval for
            :class:`~tests.benchmark.runner.BenchmarkRunner`.
    """

    def __init__(
        self,
        adapter_configs: List[Dict[str, Any]],
        scenarios: List[CoTenancyScenario],
        audio_samples: List[Dict[str, Any]],
        vram_poll_interval: float = 0.1,
    ) -> None:
        self.adapter_configs = adapter_configs
        self.scenarios = scenarios
        self.audio_samples = audio_samples
        self.vram_poll_interval = vram_poll_interval
        # Keyed by (adapter_name, model_size, sample_id) → baseline wall_time
        self._baselines: Dict[tuple, float] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_adapter(self, cfg: Dict[str, Any]) -> BaseAdapter:
        cls: Type[BaseAdapter] = cfg["cls"]
        model_size: str = cfg.get("model_size", "")
        device: str = cfg.get("device", "cuda")
        extra: Dict[str, Any] = cfg.get("extra_kwargs", {})
        model_kwarg: str = cfg.get("model_kwarg", "model_size")
        kwargs = {model_kwarg: model_size, **extra}
        try:
            return cls(device=device, **kwargs)
        except TypeError:
            return cls(**kwargs)

    def _run_one(
        self,
        adapter: BaseAdapter,
        model_size: str,
        sample: Dict[str, Any],
        scenario: CoTenancyScenario,
        runner: BenchmarkRunner,
    ) -> CoTenancyRunResult:
        """Execute one adapter-sample-scenario combination."""
        sample_id: str = sample["sample_id"]
        audio_path: str = sample["audio_path"]
        reference: Optional[str] = sample.get("reference")
        baseline_key = (adapter.name, model_size, sample_id)

        base_result = CoTenancyRunResult(
            adapter_name=adapter.name,
            model_size=model_size,
            sample_id=sample_id,
            scenario_name=scenario.name,
            tenant_vram_mb=scenario.tenant_vram_mb,
            outcome=OutcomeKind.SUCCESS,
        )

        try:
            with _VramTenant(scenario.tenant_vram_mb):
                run_result: RunResult = runner.run(adapter, audio_path, reference)

            base_result.wall_time = run_result.wall_time
            base_result.wer = run_result.wer
            base_result.vram_peak_mb = run_result.vram_peak_mb

            # Retrieve any previously recorded baseline BEFORE overwriting it.
            prior_baseline_wt = self._baselines.get(baseline_key)

            # Record baseline timing (zero-contention scenario).
            if scenario.tenant_vram_mb == 0:
                self._baselines[baseline_key] = run_result.wall_time

            # Compute slowdown only against a *prior* baseline run.
            if prior_baseline_wt and run_result.wall_time and prior_baseline_wt > 0:
                base_result.baseline_wall_time = prior_baseline_wt
                base_result.slowdown_factor = round(
                    run_result.wall_time / prior_baseline_wt, 3
                )

        except Exception as exc:  # noqa: BLE001
            outcome = OutcomeKind.OOM if _is_oom_error(exc) else OutcomeKind.ERROR
            base_result.outcome = outcome
            base_result.error_type = type(exc).__name__
            base_result.error_message = str(exc).splitlines()[0] if str(exc) else repr(exc)
            logger.warning(
                "  %s on %s (%s): %s — %s",
                adapter.name,
                sample_id,
                scenario.name,
                outcome.upper(),
                base_result.error_message,
            )

        return base_result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> CoTenancySuiteResult:
        """Execute the full co-tenancy benchmark and return a result summary.

        For each adapter configuration the model is loaded once per scenario
        group, then unloaded between scenarios to avoid cross-scenario
        interference.

        Returns:
            :class:`CoTenancySuiteResult` containing all individual run results
            and aggregate counts.
        """
        suite = CoTenancySuiteResult()
        runner = BenchmarkRunner(vram_poll_interval=self.vram_poll_interval)

        for cfg in self.adapter_configs:
            model_size: str = cfg.get("model_size", "")
            for scenario in self.scenarios:
                logger.info(
                    "Scenario '%s' — tenant_vram=%d MiB", scenario.name, scenario.tenant_vram_mb
                )
                adapter: Optional[BaseAdapter] = None

                # Load model outside the tenant context so model weights are
                # already in VRAM when the tenant is allocated (mimics real
                # co-tenancy where both services are idle-loaded).
                try:
                    adapter = self._build_adapter(cfg)
                    load_time = adapter.load()
                    logger.info(
                        "  Loaded %s/%s in %.1f s", adapter.name, model_size, load_time
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "  FAILED to load %s/%s for scenario '%s': %s",
                        cfg["cls"].name,
                        model_size,
                        scenario.name,
                        exc,
                    )
                    for sample in self.audio_samples:
                        suite.total += 1
                        suite.error_count += 1
                        outcome = OutcomeKind.OOM if _is_oom_error(exc) else OutcomeKind.ERROR
                        suite.runs.append(
                            CoTenancyRunResult(
                                adapter_name=cfg["cls"].name,
                                model_size=model_size,
                                sample_id=sample["sample_id"],
                                scenario_name=scenario.name,
                                tenant_vram_mb=scenario.tenant_vram_mb,
                                outcome=outcome,
                                error_type=type(exc).__name__,
                                error_message=str(exc).splitlines()[0],
                            )
                        )
                    continue

                for sample in self.audio_samples:
                    suite.total += 1
                    run_result = self._run_one(adapter, model_size, sample, scenario, runner)
                    suite.runs.append(run_result)
                    if run_result.outcome == OutcomeKind.SUCCESS:
                        suite.success_count += 1
                    elif run_result.outcome == OutcomeKind.OOM:
                        suite.oom_count += 1
                    else:
                        suite.error_count += 1

                # Unload before next scenario so VRAM is clean.
                try:
                    adapter.unload()
                except Exception:  # noqa: BLE001
                    pass
                gc.collect()
                try:
                    import torch  # noqa: PLC0415
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass

        return suite

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_report(suite: CoTenancySuiteResult) -> str:
        """Generate a Markdown co-tenancy report from a :class:`CoTenancySuiteResult`.

        The report contains:

        * A summary table showing outcome, average wall time, and slowdown
          factor per (adapter, model, scenario).
        * An OOM table listing every run that hit an out-of-memory error.
        * A conclusion section summarising findings.

        Args:
            suite: Result from :meth:`run`.

        Returns:
            Multi-section Markdown string.
        """
        lines: List[str] = []
        lines.append("# Co-Tenancy Benchmark Report")
        lines.append("")
        lines.append(
            f"Total runs: {suite.total} — "
            f"success: {suite.success_count}, "
            f"OOM: {suite.oom_count}, "
            f"other errors: {suite.error_count}"
        )
        lines.append("")

        # ---- Aggregate per (adapter, model, scenario) ----------------
        from collections import defaultdict
        bucket: Dict[tuple, List[CoTenancyRunResult]] = defaultdict(list)
        for r in suite.runs:
            key = (r.adapter_name, r.model_size, r.scenario_name, r.tenant_vram_mb)
            bucket[key].append(r)

        lines.append("## Per-Scenario Summary")
        lines.append("")
        cols = [
            "Engine", "Model", "Scenario", "Tenant MiB",
            "Outcome", "Wall avg s", "Slowdown", "OOM?",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

        for key in sorted(bucket.keys()):
            adapter_name, model_size, scenario_name, tenant_mb = key
            runs = bucket[key]
            successes = [r for r in runs if r.outcome == OutcomeKind.SUCCESS]
            oom_runs = [r for r in runs if r.outcome == OutcomeKind.OOM]

            if successes:
                avg_wall = sum(r.wall_time for r in successes if r.wall_time) / len(successes)
                slowdown_vals = [r.slowdown_factor for r in successes if r.slowdown_factor is not None]
                avg_slowdown = (
                    f"{sum(slowdown_vals) / len(slowdown_vals):.2f}x"
                    if slowdown_vals else "—"
                )
                outcome_str = "✓ success"
                wall_str = f"{avg_wall:.2f}"
            elif oom_runs:
                outcome_str = "✗ OOM"
                wall_str = "—"
                avg_slowdown = "—"
            else:
                outcome_str = "✗ error"
                wall_str = "—"
                avg_slowdown = "—"

            oom_flag = "⚠ OOM" if oom_runs else ""

            lines.append(
                f"| {adapter_name} | {model_size} | {scenario_name} | {tenant_mb} "
                f"| {outcome_str} | {wall_str} | {avg_slowdown} | {oom_flag} |"
            )

        lines.append("")

        # ---- OOM detail table ----------------------------------------
        oom_runs_all = [r for r in suite.runs if r.outcome == OutcomeKind.OOM]
        if oom_runs_all:
            lines.append("## OOM Incidents")
            lines.append("")
            lines.append("| Engine | Model | Scenario | Tenant MiB | Sample | Error |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for r in oom_runs_all:
                msg = (r.error_message or "").replace("|", "\\|")
                lines.append(
                    f"| {r.adapter_name} | {r.model_size} | {r.scenario_name} "
                    f"| {r.tenant_vram_mb} | {r.sample_id} | {msg} |"
                )
            lines.append("")

        # ---- Conclusion -----------------------------------------------
        lines.append("## Conclusion")
        lines.append("")
        if suite.oom_count == 0 and suite.error_count == 0:
            lines.append(
                "All tested adapters ran successfully under all co-tenancy scenarios. "
                "No OOM errors were observed."
            )
        elif suite.oom_count > 0:
            lines.append(
                f"⚠ **{suite.oom_count} OOM error(s)** were observed. "
                "Engines affected by OOM require VRAM reduction strategies "
                "(e.g. smaller model, INT8 quantisation, or exclusive GPU allocation) "
                "before deployment alongside co-tenant services."
            )
        if suite.error_count > 0:
            lines.append(
                f"Additionally, {suite.error_count} non-OOM error(s) occurred — "
                "see the per-scenario table for details."
            )

        # Slowdown summary
        slowdown_vals_all = [
            r.slowdown_factor for r in suite.runs
            if r.slowdown_factor is not None and r.tenant_vram_mb > 0
        ]
        if slowdown_vals_all:
            avg_sd = sum(slowdown_vals_all) / len(slowdown_vals_all)
            max_sd = max(slowdown_vals_all)
            lines.append(
                f"\nAverage slowdown under VRAM contention: **{avg_sd:.2f}x** "
                f"(max: {max_sd:.2f}x). "
                + (
                    "VRAM contention causes measurable slowdown."
                    if avg_sd > 1.05
                    else "Slowdown is negligible."
                )
            )
        lines.append("")

        return "\n".join(lines)

    def save_report(self, suite: CoTenancySuiteResult, path: str | Path) -> None:
        """Write the Markdown co-tenancy report to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.generate_report(suite), encoding="utf-8")
        logger.info("Co-tenancy report written → %s", path)

    def save_json(self, suite: CoTenancySuiteResult, path: str | Path) -> None:
        """Write all :class:`CoTenancyRunResult` objects to a JSON file."""
        import time

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total": suite.total,
            "success_count": suite.success_count,
            "oom_count": suite.oom_count,
            "error_count": suite.error_count,
            "runs": [
                {
                    "adapter_name": r.adapter_name,
                    "model_size": r.model_size,
                    "sample_id": r.sample_id,
                    "scenario_name": r.scenario_name,
                    "tenant_vram_mb": r.tenant_vram_mb,
                    "outcome": r.outcome,
                    "wall_time": r.wall_time,
                    "baseline_wall_time": r.baseline_wall_time,
                    "slowdown_factor": r.slowdown_factor,
                    "wer": r.wer,
                    "vram_peak_mb": r.vram_peak_mb,
                    "error_type": r.error_type,
                    "error_message": r.error_message,
                }
                for r in suite.runs
            ],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Co-tenancy JSON saved → %s", path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """CLI for the co-tenancy benchmark (issue #14).

    Reads the manifest, selects the top 3 engines (by adapter name), and
    runs them against baseline + representative contention scenarios.
    """
    import argparse
    import sys

    from .adapters import FasterWhisperAdapter, DistilWhisperAdapter, OpenAIWhisperAdapter

    parser = argparse.ArgumentParser(
        description="Run co-tenancy GPU benchmark (issue #14)",
    )
    parser.add_argument("--data-dir", "-d", type=Path,
                        default=Path("tests/benchmark/data"))
    parser.add_argument("--results", "-r", type=Path,
                        default=Path("tests/benchmark/results"))
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args(argv)

    output_path: Path = args.output or (args.results / "cotenancy_report.md")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load a small subset of samples to keep co-tenancy runs manageable.
    manifest_path = args.data_dir / "manifest.json"
    with manifest_path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)

    # Use only clean short-form samples for co-tenancy tests.
    selected_samples = [
        {
            "sample_id": s["id"],
            "audio_path": str(args.data_dir / s["filename"]),
            "reference": (
                (args.data_dir / s["reference"]).read_text(encoding="utf-8").strip()
                if (args.data_dir / s["reference"]).exists()
                else None
            ),
        }
        for s in manifest["samples"]
        if s["category"] in ("clean_single_speaker", "noisy") and s["duration_s"] <= 30
    ][:3]  # limit to 3 samples per adapter/scenario

    adapter_configs: List[Dict[str, Any]] = [
        {"cls": FasterWhisperAdapter, "model_size": "large-v3", "device": args.device},
        {
            "cls": DistilWhisperAdapter,
            "model_size": "distil-large-v3",
            "model_kwarg": "model_id",
            "device": args.device,
            "extra_kwargs": {"backend": "faster-whisper"},
        },
        {"cls": OpenAIWhisperAdapter, "model_size": "medium", "device": args.device},
    ]

    scenarios: List[CoTenancyScenario] = [
        CoTenancyScenario("baseline",      tenant_vram_mb=0,
                          description="No co-tenant; establishes baseline latency"),
        CoTenancyScenario("pyannote-idle", tenant_vram_mb=1500,
                          description="~1.5 GiB occupied by idle pyannote model"),
        CoTenancyScenario("sam-idle",      tenant_vram_mb=2048,
                          description="~2 GiB occupied by idle SAM model"),
        CoTenancyScenario("llm-idle",      tenant_vram_mb=4096,
                          description="~4 GiB occupied by idle LLM"),
    ]

    benchmark = CoTenancyBenchmark(
        adapter_configs=adapter_configs,
        scenarios=scenarios,
        audio_samples=selected_samples,
    )
    suite_result = benchmark.run()

    benchmark.save_report(suite_result, output_path)
    benchmark.save_json(suite_result, args.results / "cotenancy.json")

    print(
        f"\nCo-tenancy benchmark complete — "
        f"{suite_result.success_count}/{suite_result.total} runs successful, "
        f"{suite_result.oom_count} OOM, {suite_result.error_count} other errors"
    )
    print(f"Report: {output_path}")
    return 0 if suite_result.oom_count + suite_result.error_count == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
