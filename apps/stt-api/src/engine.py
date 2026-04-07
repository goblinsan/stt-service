import logging
import threading
import time
from pathlib import Path

from faster_whisper import WhisperModel

from .config import settings
from .models import Segment, SpeakerSummary, TranscribeResult, WordSegment

logger = logging.getLogger("stt.engine")

_model: WhisperModel | None = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()


def _resolve_device() -> tuple[str, str]:
    device = settings.device
    compute_type = settings.compute_type

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


def get_model() -> WhisperModel:
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        device, compute_type = _resolve_device()
        logger.info(
            "Loading model %s on %s (%s) — cache: %s",
            settings.model_size, device, compute_type, settings.model_cache_dir
        )
        Path(settings.model_cache_dir).mkdir(parents=True, exist_ok=True)
        _model = WhisperModel(
            settings.model_size,
            device=device,
            compute_type=compute_type,
            download_root=settings.model_cache_dir,
        )
        logger.info("Model loaded")
        return _model


def is_model_loaded() -> bool:
    return _model is not None


def get_device_info() -> tuple[str, str]:
    return _resolve_device()


def transcribe_audio(
    audio_path: str,
    language: str | None = None,
    task: str = "transcribe",
    word_timestamps: bool = False,
    initial_prompt: str | None = None,
    diarize: bool = False,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> TranscribeResult:
    model = get_model()

    t0 = time.monotonic()

    with _inference_lock:
        segments_iter, info = model.transcribe(
            audio_path,
            language=language,
            task=task,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        segments: list[Segment] = []
        full_text_parts: list[str] = []

        for seg in segments_iter:
            words = None
            if word_timestamps and seg.words:
                words = [
                    WordSegment(
                        word=w.word,
                        start=round(w.start, 3),
                        end=round(w.end, 3),
                        probability=round(w.probability, 3),
                    )
                    for w in seg.words
                ]

            segments.append(
                Segment(
                    id=seg.id,
                    start=round(seg.start, 3),
                    end=round(seg.end, 3),
                    text=seg.text.strip(),
                    words=words,
                )
            )
            full_text_parts.append(seg.text.strip())

    speakers: list[SpeakerSummary] | None = None

    if diarize:
        if not settings.hf_token:
            raise ValueError(
                "STT_HF_TOKEN must be set to use speaker diarization"
            )
        from .diarization import diarize as run_diarize
        from .merge import assign_speakers

        logger.info("Running speaker diarization on %s", audio_path)
        diarization_segs = run_diarize(
            audio_path,
            settings.hf_token,
            settings.model_cache_dir,
            settings.pyannote_model,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        segments = assign_speakers(segments, diarization_segs)
        logger.info(
            "Diarization complete: %d speaker turns assigned", len(diarization_segs)
        )

        speaker_stats: dict[str, dict] = {}
        for seg in segments:
            if seg.speaker:
                if seg.speaker not in speaker_stats:
                    speaker_stats[seg.speaker] = {"total_duration": 0.0, "segment_count": 0}
                speaker_stats[seg.speaker]["total_duration"] += seg.end - seg.start
                speaker_stats[seg.speaker]["segment_count"] += 1
        speakers = [
            SpeakerSummary(
                id=spk_id,
                total_duration=round(stats["total_duration"], 3),
                segment_count=stats["segment_count"],
            )
            for spk_id, stats in sorted(speaker_stats.items())
        ]

    return TranscribeResult(
        text=" ".join(full_text_parts),
        language=info.language,
        language_probability=round(info.language_probability, 3),
        duration=round(info.duration, 3),
        segments=segments,
        processing_time=round(time.monotonic() - t0, 3),
        speakers=speakers,
    )
