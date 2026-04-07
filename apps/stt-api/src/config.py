from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_size: str = "large-v3"
    device: str = "auto"
    compute_type: str = "auto"
    model_cache_dir: str = "/data/models"
    max_upload_mb: int = 500
    host: str = "0.0.0.0"
    port: int = 5100
    log_level: str = "info"
    # Speaker diarization (pyannote.audio)
    hf_token: str | None = None
    pyannote_model: str = "pyannote/speaker-diarization-3.1"

    model_config = {"env_prefix": "STT_"}


settings = Settings()
