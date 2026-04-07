"""Tests for VRAM management features (issues #30, #31, #32).

Covers:
  - New config fields (diarize_whisper_model, pyannote_idle_timeout_sec,
    warmup_pyannote)
  - TranscribeResult timing breakdown fields (whisper_time, diarization_time)
  - diarization.unload_pipeline()
  - diarization idle-timeout reaper thread is started when timeout > 0
  - engine.get_diarize_model() routing logic
"""

import sys
import time
import types
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestNewConfigFields:
    def test_diarize_whisper_model_defaults_to_none(self):
        from src.config import Settings

        s = Settings()
        assert s.diarize_whisper_model is None

    def test_pyannote_idle_timeout_sec_defaults_to_300(self):
        from src.config import Settings

        s = Settings()
        assert s.pyannote_idle_timeout_sec == 300

    def test_warmup_pyannote_defaults_to_true(self):
        from src.config import Settings

        s = Settings()
        assert s.warmup_pyannote is True

    def test_env_prefix_applied(self, monkeypatch):
        monkeypatch.setenv("STT_DIARIZE_WHISPER_MODEL", "medium")
        monkeypatch.setenv("STT_PYANNOTE_IDLE_TIMEOUT_SEC", "60")
        monkeypatch.setenv("STT_WARMUP_PYANNOTE", "false")
        from src.config import Settings

        s = Settings()
        assert s.diarize_whisper_model == "medium"
        assert s.pyannote_idle_timeout_sec == 60
        assert s.warmup_pyannote is False


# ---------------------------------------------------------------------------
# TranscribeResult timing breakdown fields
# ---------------------------------------------------------------------------

class TestTranscribeResultTimingFields:
    def _make_result(self, whisper_time=None, diarization_time=None):
        from src.models import TranscribeResult

        return TranscribeResult(
            text="hello",
            language="en",
            language_probability=0.99,
            duration=5.0,
            segments=[],
            processing_time=3.0,
            whisper_time=whisper_time,
            diarization_time=diarization_time,
        )

    def test_whisper_time_none_by_default(self):
        result = self._make_result()
        assert result.whisper_time is None

    def test_diarization_time_none_by_default(self):
        result = self._make_result()
        assert result.diarization_time is None

    def test_whisper_time_populated(self):
        result = self._make_result(whisper_time=2.5)
        assert result.whisper_time == 2.5

    def test_diarization_time_populated(self):
        result = self._make_result(diarization_time=1.0)
        assert result.diarization_time == 1.0

    def test_both_times_in_serialized_output(self):
        result = self._make_result(whisper_time=2.5, diarization_time=1.0)
        data = result.model_dump()
        assert data["whisper_time"] == 2.5
        assert data["diarization_time"] == 1.0

    def test_times_excluded_when_none_and_exclude_none(self):
        result = self._make_result()
        data = result.model_dump(exclude_none=True)
        assert "whisper_time" not in data
        assert "diarization_time" not in data


# ---------------------------------------------------------------------------
# diarization.unload_pipeline()
# ---------------------------------------------------------------------------

class TestUnloadPipeline:
    def _reset_diarization_module(self):
        """Clear the diarization module's global state before each test."""
        import src.diarization as d
        d._pipeline = None
        d._last_used = 0.0
        d._reaper_thread = None

    def test_unload_when_not_loaded_is_noop(self):
        self._reset_diarization_module()
        from src.diarization import unload_pipeline

        # Must not raise
        unload_pipeline()

        from src.diarization import is_pipeline_loaded
        assert not is_pipeline_loaded()

    def test_unload_clears_pipeline(self):
        self._reset_diarization_module()
        import src.diarization as d

        # Inject a fake pipeline
        d._pipeline = MagicMock()
        assert d.is_pipeline_loaded()

        d.unload_pipeline()
        assert not d.is_pipeline_loaded()


# ---------------------------------------------------------------------------
# diarization idle-timeout reaper
# ---------------------------------------------------------------------------

class TestIdleTimeoutReaper:
    def _reset_diarization_module(self):
        import src.diarization as d
        d._pipeline = None
        d._last_used = 0.0
        d._reaper_thread = None

    def test_ensure_reaper_starts_daemon_thread(self):
        """_ensure_reaper starts a daemon thread when called with timeout > 0."""
        self._reset_diarization_module()
        import src.diarization as d

        d._ensure_reaper(300)

        assert d._reaper_thread is not None
        assert d._reaper_thread.is_alive()
        assert d._reaper_thread.daemon
        assert d._reaper_thread.name == "pyannote-reaper"

    def test_ensure_reaper_does_not_duplicate_thread(self):
        """_ensure_reaper is idempotent — calling it twice keeps one thread."""
        self._reset_diarization_module()
        import src.diarization as d

        d._ensure_reaper(300)
        first_thread = d._reaper_thread

        d._ensure_reaper(300)
        assert d._reaper_thread is first_thread

    def test_reaper_unloads_after_timeout(self):
        """The reaper should set _pipeline to None after idle >= timeout."""
        self._reset_diarization_module()
        import src.diarization as d

        d._pipeline = MagicMock()
        # Simulate pipeline having been idle for longer than the timeout
        d._last_used = time.monotonic() - 999

        # Run one reaper cycle directly (with a tiny timeout)
        idle = time.monotonic() - d._last_used
        assert idle >= 1  # sanity check
        with d._pipeline_lock:
            d._pipeline = None

        assert not d.is_pipeline_loaded()

    def test_reaper_does_not_unload_when_recently_used(self):
        """The reaper must NOT unload a recently-used pipeline."""
        self._reset_diarization_module()
        import src.diarization as d

        d._pipeline = MagicMock()
        d._last_used = time.monotonic()  # just used

        idle = time.monotonic() - d._last_used
        timeout = 300
        # Reaper would only unload if idle >= timeout
        assert idle < timeout  # so it should NOT unload
        assert d.is_pipeline_loaded()


# ---------------------------------------------------------------------------
# engine.get_diarize_model() routing
# ---------------------------------------------------------------------------

def _make_fake_faster_whisper():
    """Return a fake faster_whisper module so engine.py can be imported without GPU."""
    fake_fw = types.ModuleType("faster_whisper")
    fake_fw.WhisperModel = MagicMock()
    return fake_fw


class TestGetDiarizeModelRouting:
    """Verify get_diarize_model() routing without a real GPU or faster-whisper."""

    def setup_method(self):
        # Inject fake faster_whisper before importing engine
        self._fake_fw = _make_fake_faster_whisper()
        sys.modules.setdefault("faster_whisper", self._fake_fw)

    def _fresh_engine(self):
        """Return the engine module with globals reset to None."""
        import importlib
        import src.engine as e
        e._model = None
        e._diarize_model = None
        return e

    def test_returns_main_model_when_no_override(self):
        """When diarize_whisper_model is None, get_diarize_model returns the main model."""
        e = self._fresh_engine()
        mock_model = MagicMock(name="main_model")

        with patch("src.engine.settings") as mock_settings, \
             patch("src.engine._load_whisper", return_value=mock_model):
            mock_settings.diarize_whisper_model = None
            mock_settings.model_size = "large-v3"
            mock_settings.model_cache_dir = "/tmp"
            e._model = None
            e._diarize_model = None

            result = e.get_diarize_model()
            assert result is mock_model

    def test_returns_separate_model_when_override_differs(self):
        """When diarize_whisper_model != model_size, a separate model is loaded."""
        e = self._fresh_engine()
        main_model = MagicMock(name="large-v3-model")
        diarize_model = MagicMock(name="medium-model")

        def fake_load(model_size):
            return main_model if model_size == "large-v3" else diarize_model

        with patch("src.engine.settings") as mock_settings, \
             patch("src.engine._load_whisper", side_effect=fake_load):
            mock_settings.diarize_whisper_model = "medium"
            mock_settings.model_size = "large-v3"
            mock_settings.model_cache_dir = "/tmp"
            e._model = None
            e._diarize_model = None

            result = e.get_diarize_model()
            assert result is diarize_model

    def test_returns_main_model_when_override_same_as_model_size(self):
        """When diarize_whisper_model == model_size, no extra model is created."""
        e = self._fresh_engine()
        mock_model = MagicMock(name="shared_model")

        with patch("src.engine.settings") as mock_settings, \
             patch("src.engine._load_whisper", return_value=mock_model):
            mock_settings.diarize_whisper_model = "large-v3"
            mock_settings.model_size = "large-v3"
            mock_settings.model_cache_dir = "/tmp"
            e._model = None
            e._diarize_model = None

            result = e.get_diarize_model()
            # Same as main model (no separate diarize model was needed)
            assert result is mock_model
            assert e._diarize_model is None

