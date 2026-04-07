from pydantic import BaseModel, Field


class TranscribeRequest(BaseModel):
    language: str | None = Field(None, description="ISO 639-1 code or None for auto-detect")
    task: str = Field("transcribe", pattern="^(transcribe|translate)$")
    word_timestamps: bool = Field(False, description="Include word-level timestamps")
    initial_prompt: str | None = Field(None, description="Optional prompt to condition the model")
    diarize: bool = Field(False, description="Run speaker diarization (requires STT_HF_TOKEN)")


class WordSegment(BaseModel):
    word: str
    start: float
    end: float
    probability: float


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: list[WordSegment] | None = None
    speaker: str | None = None


class SpeakerSummary(BaseModel):
    id: str
    total_duration: float
    segment_count: int


class TranscribeResult(BaseModel):
    text: str
    language: str
    language_probability: float
    duration: float
    segments: list[Segment]
    processing_time: float
    whisper_time: float | None = None
    diarization_time: float | None = None
    speakers: list[SpeakerSummary] | None = None


class DiarizationInfo(BaseModel):
    model: str
    ready: bool


class ModelInfo(BaseModel):
    model_size: str
    device: str
    compute_type: str
    ready: bool
    diarization: DiarizationInfo | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    model: ModelInfo
