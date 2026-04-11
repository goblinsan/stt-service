import asyncio
import logging
import os
import tempfile

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .engine import (
    get_device_info,
    get_diarize_model,
    get_model,
    is_diarize_model_loaded,
    is_model_loaded,
    transcribe_audio,
    unload_models,
)
from .models import (
    AsyncJobAccepted,
    AsyncJobStatusResponse,
    DiarizationInfo,
    HealthResponse,
    ModelInfo,
    RemoteTranscribeRequest,
    TranscribeResult,
)
from .jobs import get_job, submit_job
from .remote_source import RemoteSourceError, download_remote_audio

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
        if not settings.warmup_whisper:
            logger.info(
                "Whisper warmup disabled (STT_WARMUP_WHISPER=false) — "
                "model will load lazily on first request"
            )
            return
        try:
            await asyncio.to_thread(get_model)
            logger.info("Whisper model warmup complete")
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
            props = torch.cuda.get_device_properties(0)
            total_memory = getattr(props, "total_memory", getattr(props, "total_mem", 0))
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "vram_total_mb": round(total_memory / _BYTES_PER_MB),
                "vram_used_mb": round(torch.cuda.memory_allocated(0) / _BYTES_PER_MB),
                "vram_reserved_mb": round(torch.cuda.memory_reserved(0) / _BYTES_PER_MB),
            }
    except ImportError:
        pass
    except Exception:
        logger.exception("GPU introspection failed in /api/info")

    return {
        "engine": settings.engine,
        "model_size": settings.model_size,
        "device": device,
        "compute_type": compute_type,
        "model_loaded": is_model_loaded(),
        "diarize_model_loaded": is_diarize_model_loaded(),
        "gpu": gpu_info,
        "max_upload_mb": settings.max_upload_mb,
        "remote_source": {
            "enabled": True,
            "allowed_hosts": settings.remote_source_allowed_hosts,
            "timeout_sec": settings.remote_source_timeout_sec,
        },
        "diarization": {
            "available": bool(settings.hf_token),
            "model": settings.pyannote_model,
            "ready": is_pipeline_loaded(),
            "idle_timeout_sec": settings.pyannote_idle_timeout_sec,
            "whisper_model_override": settings.diarize_whisper_model,
        },
    }


@app.post("/api/models/unload")
async def unload_loaded_models():
    from .diarization import unload_pipeline

    unload_models()
    unload_pipeline()
    return {
        "ok": True,
        "model_loaded": is_model_loaded(),
        "diarize_model_loaded": is_diarize_model_loaded(),
        "diarization_ready": False,
    }


def _validate_filename(filename: str) -> str:
    if not filename:
        raise HTTPException(400, "No filename provided")

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    return ext


def _validate_transcribe_options(task: str, diarize: bool):
    if task not in ("transcribe", "translate"):
        raise HTTPException(400, "task must be 'transcribe' or 'translate'")

    if diarize and not settings.hf_token:
        raise HTTPException(
            422,
            "Speaker diarization is not configured. "
            "Set the STT_HF_TOKEN environment variable to enable it.",
        )


def _write_temp_audio(content: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        return tmp.name


async def _run_local_file_job(
    *,
    tmp_path: str,
    filename: str,
    content_bytes: int,
    language: str | None,
    task: str,
    word_timestamps: bool,
    initial_prompt: str | None,
    diarize: bool,
    min_speakers: int | None,
    max_speakers: int | None,
) -> TranscribeResult:
    try:
        return await asyncio.to_thread(
            _run_transcription,
            tmp_path,
            filename,
            content_bytes,
            language,
            task,
            word_timestamps,
            initial_prompt,
            diarize,
            min_speakers,
            max_speakers,
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def _run_remote_source_job(request: RemoteTranscribeRequest) -> TranscribeResult:
    provisional_filename = request.filename or os.path.basename(request.source_url)
    _validate_filename(provisional_filename)

    try:
        tmp_path, resolved_filename, total_bytes = await asyncio.to_thread(
            download_remote_audio,
            request.source_url,
            request.filename,
            settings.max_upload_mb * _BYTES_PER_MB,
            settings.remote_source_timeout_sec,
            settings.remote_source_allowed_hosts,
        )
        _validate_filename(resolved_filename)
    except RemoteSourceError as e:
        raise HTTPException(e.status_code, str(e))

    return await _run_local_file_job(
        tmp_path=tmp_path,
        filename=resolved_filename,
        content_bytes=total_bytes,
        language=request.language,
        task=request.task,
        word_timestamps=request.word_timestamps,
        initial_prompt=request.initial_prompt,
        diarize=request.diarize,
        min_speakers=request.min_speakers,
        max_speakers=request.max_speakers,
    )


def _run_transcription(
    tmp_path: str,
    filename: str,
    content_bytes: int,
    language: str | None,
    task: str,
    word_timestamps: bool,
    initial_prompt: str | None,
    diarize: bool,
    min_speakers: int | None,
    max_speakers: int | None,
) -> TranscribeResult:
    logger.info(
        "Transcribing %s (%d bytes, lang=%s, task=%s, diarize=%s, min_speakers=%s, max_speakers=%s)",
        filename,
        content_bytes,
        language,
        task,
        diarize,
        min_speakers,
        max_speakers,
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
    _validate_transcribe_options(task, diarize)

    filename = file.filename or ""
    ext = _validate_filename(filename)

    content = await file.read()
    if len(content) > settings.max_upload_mb * _BYTES_PER_MB:
        raise HTTPException(413, f"File too large. Max {settings.max_upload_mb} MB")

    suffix = ext if ext else ".wav"
    tmp_path = _write_temp_audio(content, suffix)

    try:
        return await _run_local_file_job(
            tmp_path=tmp_path,
            filename=filename,
            content_bytes=len(content),
            language=language,
            task=task,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(500, f"Transcription failed: {str(e)}")


@app.post("/api/transcribe-from-url", response_model=TranscribeResult)
async def transcribe_from_url(request: RemoteTranscribeRequest):
    _validate_transcribe_options(request.task, request.diarize)

    try:
        return await _run_remote_source_job(request)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Remote transcription failed")
        raise HTTPException(500, f"Transcription failed: {str(e)}")


@app.post("/api/transcribe-async", response_model=AsyncJobAccepted, status_code=202)
async def transcribe_async(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    task: str = Form("transcribe"),
    word_timestamps: bool = Form(False),
    initial_prompt: str | None = Form(None),
    diarize: bool = Form(False),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
):
    _validate_transcribe_options(task, diarize)

    filename = file.filename or ""
    ext = _validate_filename(filename)
    content = await file.read()
    if len(content) > settings.max_upload_mb * _BYTES_PER_MB:
        raise HTTPException(413, f"File too large. Max {settings.max_upload_mb} MB")

    tmp_path = _write_temp_audio(content, ext if ext else ".wav")
    return await submit_job(
        filename=filename,
        runner=lambda: _run_local_file_job(
            tmp_path=tmp_path,
            filename=filename,
            content_bytes=len(content),
            language=language,
            task=task,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        ),
        retention_seconds=settings.async_job_retention_sec,
    )


@app.post("/api/transcribe-from-url-async", response_model=AsyncJobAccepted, status_code=202)
async def transcribe_from_url_async(request: RemoteTranscribeRequest):
    _validate_transcribe_options(request.task, request.diarize)

    provisional_filename = request.filename or os.path.basename(request.source_url)
    _validate_filename(provisional_filename)

    return await submit_job(
        filename=provisional_filename,
        runner=lambda: _run_remote_source_job(request),
        retention_seconds=settings.async_job_retention_sec,
    )


@app.get("/api/jobs/{job_id}", response_model=AsyncJobStatusResponse)
async def get_async_job_status(job_id: str):
    job = await get_job(job_id, settings.async_job_retention_sec)
    if not job:
        raise HTTPException(404, "Job not found")
    return job
