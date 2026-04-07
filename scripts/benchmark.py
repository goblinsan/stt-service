#!/usr/bin/env python3
"""Benchmark diarization latency against the STT API (issue #30).

Usage
-----
    python scripts/benchmark.py [OPTIONS] <audio_file> [<audio_file> ...]

Options
-------
    --url URL       Base URL of the STT API  (default: http://localhost:5101)
    --runs N        Number of timed runs per file/mode (default: 1)
    --diarize       Also run diarized transcription and compare timings
    --language LANG Force language (default: auto)

Output
------
Prints a Markdown table with per-file timing breakdowns:

    file            | duration_s | mode        | processing_s | whisper_s | diarize_s | realtime_x
    --------------- | ---------- | ----------- | ------------ | --------- | --------- | ----------
    sample_30s.wav  |      30.4  | baseline    |         5.2  |       5.2 |         - |       5.8x
    sample_30s.wav  |      30.4  | +diarize    |        14.1  |       5.3 |       8.8 |       2.2x

Exit code is non-zero if any request fails.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests is not installed — run: pip install requests")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def transcribe(
    url: str,
    audio_path: Path,
    diarize: bool = False,
    language: str | None = None,
) -> dict:
    """POST *audio_path* to ``/api/transcribe`` and return the JSON response."""
    with audio_path.open("rb") as fh:
        resp = requests.post(
            f"{url.rstrip('/')}/api/transcribe",
            files={"file": (audio_path.name, fh, "audio/wav")},
            data={
                "diarize": "true" if diarize else "false",
                **({"language": language} if language else {}),
            },
            timeout=600,
        )
    if not resp.ok:
        raise RuntimeError(
            f"API error {resp.status_code} for {audio_path.name}: {resp.text[:200]}"
        )
    return resp.json()


def _fmt(value: float | None, unit: str = "") -> str:
    if value is None:
        return "   -"
    return f"{value:7.1f}{unit}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark STT API latency across audio files and modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("files", nargs="+", metavar="audio_file", type=Path)
    parser.add_argument("--url", default="http://localhost:5101", help="API base URL")
    parser.add_argument("--runs", type=int, default=1, help="Timed runs per file/mode")
    parser.add_argument("--diarize", action="store_true", help="Also benchmark diarized mode")
    parser.add_argument("--language", default=None, help="Force language code")
    args = parser.parse_args(argv)

    # Check API is reachable
    try:
        health = requests.get(f"{args.url.rstrip('/')}/api/health", timeout=10).json()
        print(f"Connected to STT API — model: {health['model']['model_size']}, "
              f"ready: {health['model']['ready']}\n")
    except Exception as exc:
        print(f"ERROR: Cannot reach API at {args.url}: {exc}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    errors = 0

    modes = ["baseline"]
    if args.diarize:
        modes.append("+diarize")

    for audio_path in args.files:
        if not audio_path.exists():
            print(f"WARNING: file not found: {audio_path}", file=sys.stderr)
            errors += 1
            continue

        for mode in modes:
            use_diarize = mode == "+diarize"
            run_times: list[float] = []
            whisper_times: list[float] = []
            diarize_times: list[float] = []
            audio_duration: float | None = None

            for run in range(args.runs):
                try:
                    result = transcribe(
                        args.url, audio_path, diarize=use_diarize, language=args.language
                    )
                except RuntimeError as exc:
                    print(f"  ERROR run {run + 1}: {exc}", file=sys.stderr)
                    errors += 1
                    continue

                audio_duration = result.get("duration")
                run_times.append(result["processing_time"])
                if result.get("whisper_time") is not None:
                    whisper_times.append(result["whisper_time"])
                if result.get("diarization_time") is not None:
                    diarize_times.append(result["diarization_time"])

            if not run_times:
                continue

            proc = statistics.mean(run_times)
            rows.append({
                "file": audio_path.name,
                "duration_s": audio_duration,
                "mode": mode,
                "processing_s": proc,
                "whisper_s": statistics.mean(whisper_times) if whisper_times else None,
                "diarize_s": statistics.mean(diarize_times) if diarize_times else None,
                "realtime_x": (audio_duration / proc) if audio_duration and proc > 0 else None,
                "runs": len(run_times),
            })

    # ---------------------------------------------------------------------------
    # Print results table
    # ---------------------------------------------------------------------------
    if not rows:
        print("No results collected.", file=sys.stderr)
        return 1 if errors else 0

    header = (
        f"{'file':<30} {'dur_s':>6} {'mode':>10} {'proc_s':>8} "
        f"{'whisper_s':>10} {'diarize_s':>10} {'RT_x':>6} {'runs':>5}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    baseline: dict[str, float] = {}  # file → baseline processing_s for overhead calc
    for row in rows:
        fname = row["file"]
        if row["mode"] == "baseline":
            baseline[fname] = row["processing_s"]

        overhead = ""
        if row["mode"] == "+diarize" and fname in baseline and baseline[fname] > 0:
            pct = (row["processing_s"] / baseline[fname] - 1) * 100
            overhead = f"  (+{pct:.0f}% vs baseline)"

        print(
            f"{fname:<30} {_fmt(row['duration_s'])}"
            f" {row['mode']:>10}"
            f" {_fmt(row['processing_s'])}"
            f" {_fmt(row['whisper_s'])}"
            f" {_fmt(row['diarize_s'])}"
            f" {_fmt(row['realtime_x'], 'x')}"
            f" {row['runs']:>5}"
            f"{overhead}"
        )

    print(sep)

    # Diarization overhead summary
    if args.diarize:
        overheads = []
        for row in rows:
            fname = row["file"]
            if row["mode"] == "+diarize" and fname in baseline and baseline[fname] > 0:
                pct = (row["processing_s"] / baseline[fname] - 1) * 100
                overheads.append(pct)
        if overheads:
            avg_overhead = statistics.mean(overheads)
            target = 50.0
            verdict = "✓ within target" if avg_overhead <= target else "✗ exceeds target"
            print(
                f"\nDiarization overhead: avg {avg_overhead:.0f}% "
                f"(target < {target:.0f}%) — {verdict}"
            )

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
