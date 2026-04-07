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

    # -------------------------------------------------------------------------
    # Section 4 — Analysis: best-in-class and dominated engines
    # -------------------------------------------------------------------------
    lines.extend(_generate_analysis_section(metrics_list))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis helpers (issue #13)
# ---------------------------------------------------------------------------

def _generate_analysis_section(metrics_list: List[dict]) -> List[str]:
    """Generate a Markdown analysis section from a list of engine metrics.

    Identifies:
    * Best overall WER
    * Best speed (highest RTF avg)
    * Best VRAM efficiency (lowest VRAM peak)
    * Best diarization-pairing candidate (lowest VRAM among diarize-ready
      engines that also support word timestamps)
    * Strictly dominated engines (worse than some other engine on *all* axes)

    Returns a list of Markdown lines to be joined by ``\\n``.
    """
    lines: List[str] = []
    lines.append("## Analysis")
    lines.append("")

    if not metrics_list:
        lines.append("No data available for analysis.")
        lines.append("")
        return lines

    def _label(m: dict) -> str:
        return f"{m.get('engine', '?')} / {m.get('model', '?')}"

    # --- Best WER -----------------------------------------------------------
    wer_candidates = [m for m in metrics_list if m.get("wer_avg") is not None]
    if wer_candidates:
        best_wer = min(wer_candidates, key=lambda m: m["wer_avg"])
        lines.append(
            f"**Best WER:** {_label(best_wer)} "
            f"(WER avg = {_fmt_float(best_wer.get('wer_avg'), decimals=3)})"
        )
    else:
        lines.append("**Best WER:** — (no WER data available)")

    # --- Best speed (RTF) ---------------------------------------------------
    rtf_candidates = [m for m in metrics_list if m.get("rtf_avg") is not None]
    if rtf_candidates:
        best_rtf = max(rtf_candidates, key=lambda m: m["rtf_avg"])
        lines.append(
            f"**Best Speed (RTF):** {_label(best_rtf)} "
            f"(RTF avg = {_fmt_float(best_rtf.get('rtf_avg'), decimals=2)}x)"
        )
    else:
        lines.append("**Best Speed (RTF):** — (no RTF data available)")

    # --- Best VRAM efficiency -----------------------------------------------
    vram_candidates = [m for m in metrics_list if m.get("vram_peak_mb") is not None]
    if vram_candidates:
        best_vram = min(vram_candidates, key=lambda m: m["vram_peak_mb"])
        lines.append(
            f"**Best VRAM Efficiency:** {_label(best_vram)} "
            f"(peak = {_fmt_int(best_vram.get('vram_peak_mb'))} MiB)"
        )
    else:
        lines.append("**Best VRAM Efficiency:** — (no VRAM data available)")

    # --- Best diarization-pairing candidate ---------------------------------
    # Criteria: diarization_ready AND word_timestamps, then lowest VRAM
    diar_candidates = [
        m for m in metrics_list
        if m.get("diarization_ready") and m.get("word_timestamps")
    ]
    diar_with_vram = [m for m in diar_candidates if m.get("vram_peak_mb") is not None]
    if diar_with_vram:
        best_diar = min(diar_with_vram, key=lambda m: m["vram_peak_mb"])
        lines.append(
            f"**Best Diarization Pairing:** {_label(best_diar)} "
            f"(low VRAM + word timestamps; peak = {_fmt_int(best_diar.get('vram_peak_mb'))} MiB)"
        )
    elif diar_candidates:
        # No VRAM data — pick lowest WER among diarize-ready engines
        diar_wer = [m for m in diar_candidates if m.get("wer_avg") is not None]
        if diar_wer:
            best_diar = min(diar_wer, key=lambda m: m["wer_avg"])
            lines.append(
                f"**Best Diarization Pairing:** {_label(best_diar)} "
                f"(diarize-ready + word timestamps; best WER = "
                f"{_fmt_float(best_diar.get('wer_avg'), decimals=3)})"
            )
        else:
            lines.append(
                f"**Best Diarization Pairing:** {_label(diar_candidates[0])} "
                f"(diarize-ready + word timestamps)"
            )
    else:
        lines.append(
            "**Best Diarization Pairing:** — "
            "(no engine is both diarize-ready and supports word timestamps)"
        )

    lines.append("")

    # --- Dominated engines --------------------------------------------------
    dominated = _find_dominated_engines(metrics_list)
    if dominated:
        lines.append("### Strictly Dominated Engines")
        lines.append("")
        lines.append(
            "The following engines are *strictly dominated* — "
            "there exists at least one other engine that is better (or equal) "
            "on every measured axis (WER, RTF, VRAM) and strictly better on at least one:"
        )
        lines.append("")
        for dom_label, dom_by_label in dominated:
            lines.append(f"- **{dom_label}** dominated by **{dom_by_label}**")
        lines.append("")
    else:
        lines.append(
            "_No engine is strictly dominated on all measured axes._"
        )
        lines.append("")

    return lines


def _find_dominated_engines(metrics_list: List[dict]) -> List[tuple]:
    """Return ``(dominated_label, dominated_by_label)`` pairs.

    An engine A is strictly dominated by engine B when:
    * B has WER ≤ A (lower is better) — or either is missing WER
    * B has RTF ≥ A (higher is better) — or either is missing RTF
    * B has VRAM ≤ A (lower is better) — or either is missing VRAM
    * B is strictly better than A on *at least one* of the three axes where
      both have data.

    Engines with no data on any axis cannot be dominated.
    """
    dominated: List[tuple] = []
    for i, a in enumerate(metrics_list):
        a_label = f"{a.get('engine', '?')} / {a.get('model', '?')}"
        for j, b in enumerate(metrics_list):
            if i == j:
                continue
            b_label = f"{b.get('engine', '?')} / {b.get('model', '?')}"
            if _dominates(b, a):
                dominated.append((a_label, b_label))
                break  # first dominator is enough
    return dominated


def _dominates(b: dict, a: dict) -> bool:
    """Return True when *b* dominates *a* on all shared axes."""
    axes_compared = 0
    strictly_better = False

    # WER: lower is better
    a_wer = a.get("wer_avg")
    b_wer = b.get("wer_avg")
    if a_wer is not None and b_wer is not None:
        if b_wer > a_wer:
            return False
        axes_compared += 1
        if b_wer < a_wer:
            strictly_better = True

    # RTF: higher is better
    a_rtf = a.get("rtf_avg")
    b_rtf = b.get("rtf_avg")
    if a_rtf is not None and b_rtf is not None:
        if b_rtf < a_rtf:
            return False
        axes_compared += 1
        if b_rtf > a_rtf:
            strictly_better = True

    # VRAM: lower is better
    a_vram = a.get("vram_peak_mb")
    b_vram = b.get("vram_peak_mb")
    if a_vram is not None and b_vram is not None:
        if b_vram > a_vram:
            return False
        axes_compared += 1
        if b_vram < a_vram:
            strictly_better = True

    return axes_compared > 0 and strictly_better


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
