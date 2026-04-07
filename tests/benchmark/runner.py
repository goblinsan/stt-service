"""Benchmark runner for STT engine adapters (issue #1).

Usage
-----
::

    from tests.benchmark.runner import BenchmarkRunner
    from my_adapter import MyAdapter

    runner = BenchmarkRunner()
    result = runner.run(
        adapter=MyAdapter(),
        audio_path="tests/benchmark/data/sample_en.wav",
        reference="hello world this is a test",
    )
    print(result)

``BenchmarkRunner.run`` returns a :class:`RunResult` containing the transcript
text, segments, wall-clock time, GPU memory peak, and computed WER/CER.

First-token latency is propagated from the adapter result if the engine
exposes it; otherwise it is ``None`` (engines that use a synchronous
request/response model will not populate this field).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    import jiwer
except ImportError:
    jiwer = None  # type: ignore[assignment]

from .adapters.base import AdapterResult, BaseAdapter, Segment


@dataclass
class RunResult:
    """Result of a single benchmark run against one audio sample."""

    adapter_name: str
    audio_path: str

    # Transcript ---------------------------------------------------------------
    text: str
    segments: List[Segment]

    # Accuracy -----------------------------------------------------------------
    #: Word Error Rate — ``None`` when no reference transcript was supplied or
    #: when ``jiwer`` is not installed.
    wer: Optional[float] = None
    #: Character Error Rate — same availability as :attr:`wer`.
    cer: Optional[float] = None

    # Latency ------------------------------------------------------------------
    wall_time: float = 0.0
    #: Nullable — only populated when the engine supports streaming token
    #: boundaries.
    first_token_latency: Optional[float] = None

    # GPU memory ---------------------------------------------------------------
    #: Peak VRAM allocated during inference (MiB).  ``None`` on CPU-only runs
    #: or when PyTorch is not installed.
    vram_peak_mb: Optional[int] = None

    # Language -----------------------------------------------------------------
    language: Optional[str] = None
    language_probability: Optional[float] = None

    # Duration / real-time factor ----------------------------------------------
    audio_duration: Optional[float] = None
    #: Real-time factor: ``audio_duration / wall_time``.  Higher = faster.
    rtf: Optional[float] = None


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def _make_wer_transform():
    """Return a jiwer Compose transform for normalised WER computation."""
    return jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ReduceToListOfListOfWords(),
    ])


def _make_cer_transform():
    """Return a jiwer Compose transform for normalised CER computation."""
    return jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfChars(),
    ])


def compute_wer(hypothesis: str, reference: str) -> Optional[float]:
    """Compute Word Error Rate using ``jiwer``.

    Returns ``None`` when ``jiwer`` is not installed.
    """
    if jiwer is None:
        return None
    return jiwer.wer(
        reference,
        hypothesis,
        reference_transform=_make_wer_transform(),
        hypothesis_transform=_make_wer_transform(),
    )


def compute_cer(hypothesis: str, reference: str) -> Optional[float]:
    """Compute Character Error Rate using ``jiwer``.

    Returns ``None`` when ``jiwer`` is not installed.
    """
    if jiwer is None:
        return None
    return jiwer.cer(
        reference,
        hypothesis,
        reference_transform=_make_cer_transform(),
        hypothesis_transform=_make_cer_transform(),
    )


# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------

def _poll_vram(
    interval: float,
    peak_box: list,
    stop_event: threading.Event,
) -> None:
    """Poll GPU memory allocation in a background thread and record the peak.

    Stores the peak allocated MiB into ``peak_box[0]`` when the stop event
    fires.  Uses ``torch.cuda.max_memory_allocated`` which tracks the
    running maximum since the last reset.
    """
    peak: Optional[int] = None
    try:
        import torch  # noqa: PLC0415

        while not stop_event.is_set():
            if torch.cuda.is_available():
                allocated = torch.cuda.max_memory_allocated(0) // (1024 * 1024)
                if peak is None or allocated > peak:
                    peak = allocated
            stop_event.wait(interval)
    except Exception:  # noqa: BLE001
        pass
    peak_box.append(peak)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Runs a single STT adapter against an audio file and optional reference.

    Args:
        vram_poll_interval: Seconds between VRAM samples in the background
            monitoring thread.  Lower values are more accurate but add
            marginal overhead.
    """

    def __init__(self, vram_poll_interval: float = 0.1) -> None:
        self.vram_poll_interval = vram_poll_interval

    def run(
        self,
        adapter: BaseAdapter,
        audio_path: str | Path,
        reference: Optional[str] = None,
    ) -> RunResult:
        """Run *adapter* against *audio_path* and return a :class:`RunResult`.

        Args:
            adapter: An STT engine adapter implementing :class:`BaseAdapter`.
            audio_path: Path to the audio file.
            reference: Reference transcript for WER/CER calculation.  When
                ``None``, accuracy fields in the result are left as ``None``.

        Returns:
            :class:`RunResult` with all available metrics populated.
        """
        audio_path = str(audio_path)

        # Reset PyTorch peak memory counter so we measure only this run.
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(0)
        except Exception:  # noqa: BLE001
            pass

        # Start background VRAM monitor.
        stop_event = threading.Event()
        peak_box: list = []
        vram_thread = threading.Thread(
            target=_poll_vram,
            args=(self.vram_poll_interval, peak_box, stop_event),
            daemon=True,
        )
        vram_thread.start()

        # Run inference.
        t0 = time.monotonic()
        result: AdapterResult = adapter.transcribe(audio_path)
        wall_time = time.monotonic() - t0

        # Stop VRAM monitor.
        stop_event.set()
        vram_thread.join(timeout=2.0)

        # Resolve peak VRAM — prefer torch's authoritative counter.
        vram_peak_mb: Optional[int] = None
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                vram_peak_mb = torch.cuda.max_memory_allocated(0) // (1024 * 1024)
        except Exception:  # noqa: BLE001
            pass
        if vram_peak_mb is None and peak_box:
            vram_peak_mb = peak_box[0]

        # Accuracy metrics (require reference + jiwer).
        wer = compute_wer(result.text, reference) if reference else None
        cer = compute_cer(result.text, reference) if reference else None

        # Real-time factor.
        duration = result.duration
        rtf = (duration / wall_time) if duration and wall_time > 0 else None

        return RunResult(
            adapter_name=adapter.name,
            audio_path=audio_path,
            text=result.text,
            segments=result.segments,
            wer=wer,
            cer=cer,
            wall_time=wall_time,
            first_token_latency=result.first_token_latency,
            vram_peak_mb=vram_peak_mb,
            language=result.language,
            language_probability=result.language_probability,
            audio_duration=duration,
            rtf=rtf,
        )
