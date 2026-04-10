from pydantic_settings import BaseSettings

# Engines selectable via STT_ENGINE.  Only "faster-whisper" is fully supported
# in production; additional values are reserved for future adapters.
SUPPORTED_ENGINES: tuple[str, ...] = ("faster-whisper",)


class Settings(BaseSettings):
    model_size: str = "large-v3"
    device: str = "auto"
    compute_type: str = "auto"
    model_cache_dir: str = "/data/models"
    max_upload_mb: int = 500
    host: str = "0.0.0.0"
    port: int = 5100
    log_level: str = "info"
    # STT backend engine.  "faster-whisper" is the production default.
    # Set STT_ENGINE to select an alternative adapter.
    engine: str = "faster-whisper"
    # Speaker diarization (pyannote.audio)
    hf_token: str | None = None
    pyannote_model: str = "pyannote/speaker-diarization-3.1"
    # VRAM management (issue #32)
    # When set, diarized requests use this Whisper model size instead of
    # model_size.  Useful when large-v3 + pyannote would exceed VRAM budget;
    # e.g. STT_DIARIZE_WHISPER_MODEL=medium keeps combined usage ~3.5 GB.
    diarize_whisper_model: str | None = None
    # Seconds of inactivity before the pyannote pipeline is unloaded to free
    # VRAM.  Set to 0 to never unload (legacy behaviour).
    pyannote_idle_timeout_sec: int = 300
    # Pre-warm the pyannote pipeline at startup.  Disable to save VRAM until
    # the first diarize request arrives (lazy-load strategy).
    warmup_pyannote: bool = True
    # Remote-source ingest for large presigned uploads.
    remote_source_timeout_sec: int = 600
    remote_source_allowed_hosts: str | None = None
    async_job_retention_sec: int = 3600

    model_config = {"env_prefix": "STT_"}


settings = Settings()
