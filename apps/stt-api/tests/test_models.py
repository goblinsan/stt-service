"""Tests for Pydantic models, merge logic, and diarization helpers."""
import pytest

from src.models import Segment, SpeakerSummary, TranscribeResult, WordSegment


class TestSpeakerSummaryModel:
    def test_fields(self):
        s = SpeakerSummary(id="SPEAKER_00", total_duration=12.5, segment_count=3)
        assert s.id == "SPEAKER_00"
        assert s.total_duration == 12.5
        assert s.segment_count == 3

    def test_zero_duration(self):
        s = SpeakerSummary(id="SPEAKER_01", total_duration=0.0, segment_count=0)
        assert s.total_duration == 0.0


class TestTranscribeResultWithSpeakers:
    def _make_result(self, speakers=None):
        return TranscribeResult(
            text="hello world",
            language="en",
            language_probability=0.99,
            duration=5.0,
            segments=[
                Segment(id=0, start=0.0, end=2.5, text="hello", speaker="SPEAKER_00"),
                Segment(id=1, start=2.5, end=5.0, text="world", speaker="SPEAKER_01"),
            ],
            processing_time=1.0,
            speakers=speakers,
        )

    def test_speakers_none_by_default(self):
        result = TranscribeResult(
            text="hi",
            language="en",
            language_probability=0.9,
            duration=1.0,
            segments=[],
            processing_time=0.5,
        )
        assert result.speakers is None

    def test_speakers_populated(self):
        speakers = [
            SpeakerSummary(id="SPEAKER_00", total_duration=2.5, segment_count=1),
            SpeakerSummary(id="SPEAKER_01", total_duration=2.5, segment_count=1),
        ]
        result = self._make_result(speakers=speakers)
        assert result.speakers is not None
        assert len(result.speakers) == 2
        assert result.speakers[0].id == "SPEAKER_00"
        assert result.speakers[1].id == "SPEAKER_01"

    def test_serialization_includes_speakers(self):
        speakers = [SpeakerSummary(id="SPEAKER_00", total_duration=5.0, segment_count=2)]
        result = self._make_result(speakers=speakers)
        data = result.model_dump()
        assert "speakers" in data
        assert data["speakers"][0]["id"] == "SPEAKER_00"
        assert data["speakers"][0]["total_duration"] == 5.0
        assert data["speakers"][0]["segment_count"] == 2

    def test_serialization_omits_speakers_when_none(self):
        result = self._make_result(speakers=None)
        data = result.model_dump(exclude_none=True)
        assert "speakers" not in data


class TestAssignSpeakers:
    """Tests for merge.assign_speakers."""

    def _seg(self, id, start, end, text="x"):
        return Segment(id=id, start=start, end=end, text=text)

    def test_simple_assignment(self):
        from src.merge import assign_speakers

        segments = [self._seg(0, 0.0, 2.0), self._seg(1, 2.0, 4.0)]
        diarization = [(0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")]
        result = assign_speakers(segments, diarization)
        assert result[0].speaker == "SPEAKER_00"
        assert result[1].speaker == "SPEAKER_01"

    def test_no_coverage_returns_none(self):
        from src.merge import assign_speakers

        segments = [self._seg(0, 5.0, 6.0)]
        diarization = [(0.0, 2.0, "SPEAKER_00")]
        result = assign_speakers(segments, diarization)
        assert result[0].speaker is None

    def test_longest_overlap_wins(self):
        from src.merge import assign_speakers

        # Segment spans 0-4; SPEAKER_00 covers 0-1 (1s), SPEAKER_01 covers 1-4 (3s)
        segments = [self._seg(0, 0.0, 4.0)]
        diarization = [(0.0, 1.0, "SPEAKER_00"), (1.0, 4.0, "SPEAKER_01")]
        result = assign_speakers(segments, diarization)
        assert result[0].speaker == "SPEAKER_01"


class TestDiarizeSignature:
    """Verify that diarize() accepts min_speakers / max_speakers kwargs."""

    def test_signature(self):
        import inspect
        from src.diarization import diarize

        params = inspect.signature(diarize).parameters
        assert "min_speakers" in params
        assert "max_speakers" in params
        assert params["min_speakers"].default is None
        assert params["max_speakers"].default is None


class TestDiarizeValidation:
    """Verify that diarize() validates speaker count arguments before calling the pipeline."""

    def _call_diarize(self, min_s, max_s):
        # We don't have a real pipeline, but we can verify that ValueError is raised
        # before the pipeline is even loaded (by testing with a mock).
        from unittest.mock import MagicMock, patch
        from src.diarization import diarize

        mock_pipeline = MagicMock()
        with patch("src.diarization.get_pipeline", return_value=mock_pipeline):
            diarize(
                audio_path="/dev/null",
                hf_token="tok",
                model_cache_dir="/tmp",
                pyannote_model="test/model",
                min_speakers=min_s,
                max_speakers=max_s,
            )

    def test_min_speakers_zero_raises(self):
        import pytest
        with pytest.raises(ValueError, match="min_speakers must be >= 1"):
            self._call_diarize(min_s=0, max_s=None)

    def test_max_speakers_zero_raises(self):
        import pytest
        with pytest.raises(ValueError, match="max_speakers must be >= 1"):
            self._call_diarize(min_s=None, max_s=0)

    def test_min_greater_than_max_raises(self):
        import pytest
        with pytest.raises(ValueError, match="min_speakers.*must be <= max_speakers"):
            self._call_diarize(min_s=5, max_s=2)

    def test_valid_constraints_pass(self):
        from unittest.mock import MagicMock, patch
        from src.diarization import diarize

        mock_pipeline = MagicMock()
        mock_pipeline.return_value.itertracks.return_value = []
        with patch("src.diarization.get_pipeline", return_value=mock_pipeline):
            result = diarize(
                audio_path="/dev/null",
                hf_token="tok",
                model_cache_dir="/tmp",
                pyannote_model="test/model",
                min_speakers=2,
                max_speakers=4,
            )
        assert result == []
        mock_pipeline.assert_called_once_with("/dev/null", min_speakers=2, max_speakers=4)
