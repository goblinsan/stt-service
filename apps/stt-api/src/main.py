import logging
import os
import tempfile
import time

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .engine import get_device_info, get_model, is_model_loaded, transcribe_audio
from .models import HealthResponse, ModelInfo, TranscribeResult

logger = logging.getLogger("stt.api")

app = FastAPI(title="STT Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".mp4", ".aac", ".wma"}


@app.on_event("startup")
async def startup():
    logger.info("STT API starting — model will load on first request")


@app.get("/api/health", response_model=HealthResponse)
async def health():
    device, compute_type = get_device_info()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        model=ModelInfo(
            model_size=settings.model_size,
            device=device,
            compute_type=compute_type,
            ready=is_model_loaded(),
        ),
    )


@app.get("/api/info")
async def info():
    device, compute_type = get_device_info()
    gpu_info = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "vram_total_mb": round(torch.cuda.get_device_properties(0).total_mem / 1024 / 1024),
                "vram_used_mb": round(torch.cuda.memory_allocated(0) / 1024 / 1024),
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
    }


@app.post("/api/transcribe", response_model=TranscribeResult)
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    task: str = Form("transcribe"),
    word_timestamps: bool = Form(False),
    initial_prompt: str | None = Form(None),
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

    content = await file.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max {settings.max_upload_mb} MB")

    suffix = ext if ext else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Ensure model is loaded (lazy init)
        logger.info("Transcribing %s (%d bytes, lang=%s, task=%s)", file.filename, len(content), language, task)
        get_model()

        result = transcribe_audio(
            audio_path=tmp_path,
            language=language if language else None,
            task=task,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
        )
        logger.info(
            "Done: %.1fs audio in %.1fs (%.1fx realtime), lang=%s",
            result.duration, result.processing_time,
            result.duration / result.processing_time if result.processing_time > 0 else 0,
            result.language,
        )
        return result
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(500, f"Transcription failed: {str(e)}")
    finally:
        os.unlink(tmp_path)
