"""Merge Whisper transcript segments with pyannote diarization segments.

Assigns a ``speaker`` label to each Whisper segment using a
longest-overlap strategy.

Edge cases handled
------------------
* **Speaker change mid-sentence** — the speaker whose diarization
  interval has the greatest total overlap with the Whisper segment wins.
* **Short pauses / gaps in diarization** — segments with no diarization
  coverage keep ``speaker = None``.
* **Overlapping speech** — the speaker with the highest cumulative
  overlap for a given Whisper segment is assigned.
"""

from __future__ import annotations

from .models import Segment


def assign_speakers(
    transcript_segments: list[Segment],
    diarization_segments: list[tuple[float, float, str]],
) -> list[Segment]:
    """Return new ``Segment`` objects with ``speaker`` populated.

    For each Whisper segment the function sums the overlap (in seconds)
    contributed by every diarization turn for each speaker, then assigns
    the speaker with the greatest total overlap.  If no diarization turn
    overlaps with a given segment, ``speaker`` remains ``None``.

    Args:
        transcript_segments: Whisper segments produced by
            :func:`stt.engine.transcribe_audio`.
        diarization_segments: ``(start, end, speaker_label)`` tuples
            produced by :func:`stt.diarization.diarize`.

    Returns:
        A new list of ``Segment`` instances with the ``speaker`` field
        set (or ``None`` when no diarization coverage exists).
    """
    result: list[Segment] = []

    for seg in transcript_segments:
        overlap_by_speaker: dict[str, float] = {}

        for d_start, d_end, speaker in diarization_segments:
            # Skip turns that don't overlap with this segment at all.
            if d_end <= seg.start or d_start >= seg.end:
                continue
            overlap = min(seg.end, d_end) - max(seg.start, d_start)
            overlap_by_speaker[speaker] = (
                overlap_by_speaker.get(speaker, 0.0) + overlap
            )

        best_speaker: str | None = (
            max(overlap_by_speaker, key=overlap_by_speaker.__getitem__)
            if overlap_by_speaker
            else None
        )
        result.append(seg.model_copy(update={"speaker": best_speaker}))

    return result
