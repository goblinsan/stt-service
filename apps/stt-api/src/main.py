import asyncio
import logging
import os
import tempfile

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .engine import get_device_info, get_diarize_model, get_model, is_model_loaded, transcribe_audio
from .models import DiarizationInfo, HealthResponse, ModelInfo, TranscribeResult

logger = logging.getLogger("stt.api")

_BYTES_PER_MB = 1024 * 1024

app = FastAPI(title="STT Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".mp4", ".aac", ".wma", ".aif", ".aiff"}


@app.on_event("startup")
async def startup():
    logger.info("STT API starting — warming up models in background")

    async def _load_whisper():
        try:
            await asyncio.to_thread(get_model)
            logger.info("Whisper model warmup complete")
            # If a separate diarize model is configured, warm it up too.
            if settings.diarize_whisper_model and settings.diarize_whisper_model != settings.model_size:
                await asyncio.to_thread(get_diarize_model)
                logger.info("Diarize Whisper model warmup complete (%s)", settings.diarize_whisper_model)
        except Exception:
            logger.exception("Whisper model warmup failed")

    async def _load_pyannote():
        if not settings.hf_token:
            return
        if not settings.warmup_pyannote:
            logger.info(
                "Pyannote warmup disabled (STT_WARMUP_PYANNOTE=false) — "
                "pipeline will load lazily on first diarize request"
            )
            return
        from .diarization import get_pipeline
        try:
            await asyncio.to_thread(
                get_pipeline,
                settings.hf_token,
                settings.model_cache_dir,
                settings.pyannote_model,
            )
            logger.info("Pyannote pipeline warmup complete")
        except Exception:
            logger.exception("Pyannote pipeline warmup failed")

    asyncio.create_task(_load_whisper())
    asyncio.create_task(_load_pyannote())


@app.get("/api/health", response_model=HealthResponse)
async def health():
    from .diarization import is_pipeline_loaded

    device, compute_type = get_device_info()
    diarization_info: DiarizationInfo | None = None
    if settings.hf_token:
        diarization_info = DiarizationInfo(
            model=settings.pyannote_model,
            ready=is_pipeline_loaded(),
        )
    return HealthResponse(
        status="ok",
        version="0.1.0",
        model=ModelInfo(
            model_size=settings.model_size,
            device=device,
            compute_type=compute_type,
            ready=is_model_loaded(),
            diarization=diarization_info,
        ),
    )


@app.get("/api/info")
async def info():
    from .diarization import is_pipeline_loaded

    device, compute_type = get_device_info()
    gpu_info = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "vram_total_mb": round(torch.cuda.get_device_properties(0).total_mem / _BYTES_PER_MB),
                "vram_used_mb": round(torch.cuda.memory_allocated(0) / _BYTES_PER_MB),
                "vram_reserved_mb": round(torch.cuda.memory_reserved(0) / _BYTES_PER_MB),
            }
    except ImportError:
        pass

    return {
        "model_size": settings.model_size,
        "device": device,
        "compute_type": compute_type,
        "model_loaded": is_model_loaded(),
        "gpu": gpu_info,
        "max_upload_mb": settings.max_upload_mb,
        "diarization": {
            "available": bool(settings.hf_token),
            "model": settings.pyannote_model,
            "ready": is_pipeline_loaded(),
            "idle_timeout_sec": settings.pyannote_idle_timeout_sec,
            "whisper_model_override": settings.diarize_whisper_model,
        },
    }


@app.post("/api/transcribe", response_model=TranscribeResult)
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    task: str = Form("transcribe"),
    word_timestamps: bool = Form(False),
    initial_prompt: str | None = Form(None),
    diarize: bool = Form(False),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
):
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    if task not in ("transcribe", "translate"):
        raise HTTPException(400, "task must be 'transcribe' or 'translate'")

    if diarize and not settings.hf_token:
        raise HTTPException(
            422,
            "Speaker diarization is not configured. "
            "Set the STT_HF_TOKEN environment variable to enable it.",
        )

    content = await file.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max {settings.max_upload_mb} MB")

    suffix = ext if ext else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        logger.info(
            "Transcribing %s (%d bytes, lang=%s, task=%s, diarize=%s, min_speakers=%s, max_speakers=%s)",
            file.filename, len(content), language, task, diarize, min_speakers, max_speakers,
        )
        get_model()

        result = transcribe_audio(
            audio_path=tmp_path,
            language=language if language else None,
            task=task,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        logger.info(
            "Done: %.1fs audio in %.1fs total (whisper=%.1fs%s), %.1fx realtime, lang=%s",
            result.duration,
            result.processing_time,
            result.whisper_time or 0.0,
            f", diarize={result.diarization_time:.1f}s" if result.diarization_time is not None else "",
            result.duration / result.processing_time if result.processing_time > 0 else 0,
            result.language,
        )
        return result
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(500, f"Transcription failed: {str(e)}")
    finally:
        os.unlink(tmp_path)
