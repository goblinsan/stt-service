"""Comparison report generator (issue #4).

Reads per-engine JSON result files produced by :mod:`tests.benchmark.metrics`
and generates a Markdown comparison table.

Usage (CLI)
-----------
::

    python -m tests.benchmark.report \\
        --input  tests/benchmark/results/ \\
        --output tests/benchmark/results/report.md

Usage (library)
---------------
::

    from tests.benchmark.report import generate_report
    from tests.benchmark.metrics import MetricsCollector

    data = MetricsCollector.load_json("results/faster-whisper.json")
    md = generate_report(data["engines"])
    Path("results/report.md").write_text(md)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_float(value: Optional[float], decimals: int = 3, suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{value:.{decimals}f}{suffix}"


def _fmt_int(value: Optional[int]) -> str:
    if value is None:
        return "—"
    return str(value)


def _fmt_bool(value: bool) -> str:
    return "✓" if value else "✗"


def _row(cells: List[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _header(cols: List[str]) -> List[str]:
    return [_row(cols), _row(["---"] * len(cols))]


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(metrics_list: List[dict]) -> str:
    """Generate a Markdown comparison report from a list of engine metrics dicts.

    The *metrics_list* items should conform to the schema produced by
    :func:`dataclasses.asdict` applied to :class:`~tests.benchmark.metrics.EngineMetrics`.

    Args:
        metrics_list: List of engine metrics dictionaries.

    Returns:
        A multi-section Markdown string ready to be written to a ``.md`` file.
    """
    if not metrics_list:
        return "No results available.\n"

    lines: List[str] = []

    # -------------------------------------------------------------------------
    # Section 1 — Performance comparison
    # -------------------------------------------------------------------------
    lines.append("# STT Engine Benchmark Report")
    lines.append("")
    lines.append("## Performance Comparison")
    lines.append("")

    perf_cols = ["Engine", "Model", "WER avg", "CER avg", "RTF avg", "VRAM MB", "Load s", "Runs"]
    lines.extend(_header(perf_cols))

    for m in sorted(metrics_list, key=lambda x: (x.get("wer_avg") or 1.0)):
        lines.append(_row([
            m.get("engine", "?"),
            m.get("model", "?"),
            _fmt_float(m.get("wer_avg"), decimals=3),
            _fmt_float(m.get("cer_avg"), decimals=3),
            _fmt_float(m.get("rtf_avg"), decimals=2, suffix="x"),
            _fmt_int(m.get("vram_peak_mb")),
            _fmt_float(m.get("model_load_time"), decimals=1),
            str(m.get("run_count", 0)),
        ]))

    lines.append("")

    # -------------------------------------------------------------------------
    # Section 2 — Feature support matrix
    # -------------------------------------------------------------------------
    lines.append("## Feature Support Matrix")
    lines.append("")

    feat_cols = ["Engine", "Model", "Lang Detect", "Word Timestamps", "Translation", "Diarize-Ready"]
    lines.extend(_header(feat_cols))

    for m in sorted(metrics_list, key=lambda x: (x.get("engine", ""), x.get("model", ""))):
        lines.append(_row([
            m.get("engine", "?"),
            m.get("model", "?"),
            _fmt_bool(m.get("language_detection", False)),
            _fmt_bool(m.get("word_timestamps", False)),
            _fmt_bool(m.get("translation", False)),
            _fmt_bool(m.get("diarization_ready", False)),
        ]))

    lines.append("")

    # -------------------------------------------------------------------------
    # Section 3 — Latency details
    # -------------------------------------------------------------------------
    lines.append("## Latency Details")
    lines.append("")

    lat_cols = ["Engine", "Model", "Wall avg s", "Wall min s", "Wall max s", "WER min", "WER max", "First Token s"]
    lines.extend(_header(lat_cols))

    for m in sorted(metrics_list, key=lambda x: x.get("wall_time_avg", 0.0)):
        lines.append(_row([
            m.get("engine", "?"),
            m.get("model", "?"),
            _fmt_float(m.get("wall_time_avg"), decimals=2),
            _fmt_float(m.get("wall_time_min"), decimals=2),
            _fmt_float(m.get("wall_time_max"), decimals=2),
            _fmt_float(m.get("wer_min"), decimals=3),
            _fmt_float(m.get("wer_max"), decimals=3),
            _fmt_float(m.get("first_token_latency_avg"), decimals=2),
        ]))

    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entry-point for the report generator.

    Reads all ``*.json`` files from *--input* directory and writes a Markdown
    report to *--output* (default: ``<input>/report.md``).
    """
    parser = argparse.ArgumentParser(
        description="Generate Markdown comparison report from benchmark JSON results",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("tests/benchmark/results"),
        help="Directory containing benchmark JSON result files (default: tests/benchmark/results)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output Markdown file (default: <input>/report.md)",
    )
    args = parser.parse_args(argv)

    input_dir: Path = args.input
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    output_path: Path = args.output or (input_dir / "report.md")

    # Collect metrics from all JSON files in the input directory.
    metrics_list: List[dict] = []
    for json_file in sorted(input_dir.glob("*.json")):
        try:
            with json_file.open(encoding="utf-8") as fh:
                data = json.load(fh)
            # Support both multi-engine wrapper format and bare single-engine.
            if "engines" in data:
                metrics_list.extend(data["engines"])
            elif "engine" in data:
                metrics_list.append(data)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: Failed to read {json_file}: {exc}", file=sys.stderr)

    if not metrics_list:
        print("WARNING: No metrics data found in JSON files", file=sys.stderr)

    report = generate_report(metrics_list)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Report written to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
