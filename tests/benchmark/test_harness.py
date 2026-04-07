"""Unit tests for the STT benchmark harness (issues #1, #3, #4).

Tests do not require a GPU, a running STT service, or real audio files.
A ``StubAdapter`` provides deterministic responses for all runner/metrics tests.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from tests.benchmark.adapters.base import (
    AdapterResult,
    BaseAdapter,
    FeatureSupport,
    Segment,
    WordTimestamp,
)
from tests.benchmark.metrics import EngineMetrics, MetricsCollector
from tests.benchmark.report import generate_report
from tests.benchmark.runner import BenchmarkRunner, RunResult, compute_cer, compute_wer


# ---------------------------------------------------------------------------
# Stub adapter
# ---------------------------------------------------------------------------

class StubAdapter(BaseAdapter):
    """Deterministic adapter that returns a fixed result without any I/O."""

    name = "stub-engine"
    features = FeatureSupport(
        language_detection=True,
        word_timestamps=True,
        translation=False,
        diarization_ready=True,
    )

    def __init__(
        self,
        text: str = "hello world",
        processing_time: float = 1.0,
        duration: float = 5.0,
        language: str = "en",
        first_token_latency: Optional[float] = None,
    ) -> None:
        self._text = text
        self._processing_time = processing_time
        self._duration = duration
        self._language = language
        self._first_token_latency = first_token_latency

    def load(self) -> float:
        return 0.01

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> AdapterResult:
        return AdapterResult(
            text=self._text,
            segments=[Segment(id=0, start=0.0, end=self._duration, text=self._text)],
            processing_time=self._processing_time,
            language=self._language,
            language_probability=0.99,
            duration=self._duration,
            first_token_latency=self._first_token_latency,
        )


# ---------------------------------------------------------------------------
# Adapter base-class tests
# ---------------------------------------------------------------------------

class TestAdapterBaseClass:
    def test_stub_adapter_is_concrete(self):
        adapter = StubAdapter()
        assert adapter.name == "stub-engine"

    def test_feature_support_defaults_to_false(self):
        fs = FeatureSupport()
        assert not fs.language_detection
        assert not fs.word_timestamps
        assert not fs.translation
        assert not fs.diarization_ready

    def test_feature_support_fields(self):
        fs = FeatureSupport(language_detection=True, word_timestamps=True)
        assert fs.language_detection
        assert fs.word_timestamps
        assert not fs.translation

    def test_word_timestamp_fields(self):
        wt = WordTimestamp(word="hello", start=0.0, end=0.5, probability=0.98)
        assert wt.word == "hello"
        assert wt.start == 0.0
        assert wt.end == 0.5
        assert wt.probability == 0.98

    def test_segment_optional_fields(self):
        seg = Segment(id=0, start=0.0, end=1.0, text="hi")
        assert seg.words is None
        assert seg.speaker is None

    def test_adapter_result_optional_fields(self):
        result = AdapterResult(text="test", segments=[], processing_time=0.5)
        assert result.language is None
        assert result.language_probability is None
        assert result.duration is None
        assert result.first_token_latency is None

    def test_unload_is_noop_by_default(self):
        adapter = StubAdapter()
        adapter.unload()  # must not raise


# ---------------------------------------------------------------------------
# Accuracy helper tests
# ---------------------------------------------------------------------------

class TestComputeWER:
    def test_perfect_match(self):
        wer = compute_wer("hello world", "hello world")
        assert wer == pytest.approx(0.0)

    def test_complete_mismatch(self):
        wer = compute_wer("foo bar", "hello world")
        assert wer == pytest.approx(1.0)

    def test_case_insensitive(self):
        wer = compute_wer("Hello World", "hello world")
        assert wer == pytest.approx(0.0)

    def test_one_word_error(self):
        # "hello world" vs "hello earth" — 1 substitution / 2 words = 0.5
        wer = compute_wer("hello earth", "hello world")
        assert wer == pytest.approx(0.5)

    def test_punctuation_ignored(self):
        wer = compute_wer("hello, world!", "hello world")
        assert wer == pytest.approx(0.0)

    def test_empty_hypothesis(self):
        wer = compute_wer("", "hello world")
        assert wer == pytest.approx(1.0)


class TestComputeCER:
    def test_perfect_match(self):
        cer = compute_cer("hello", "hello")
        assert cer == pytest.approx(0.0)

    def test_one_char_error(self):
        cer = compute_cer("helo", "hello")
        # 1 deletion / 5 chars = 0.2
        assert cer == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# BenchmarkRunner tests
# ---------------------------------------------------------------------------

class TestBenchmarkRunner:
    def test_basic_run_no_reference(self):
        runner = BenchmarkRunner()
        adapter = StubAdapter(text="hello world", processing_time=0.5, duration=5.0)
        result = runner.run(adapter, "/fake/audio.wav")

        assert isinstance(result, RunResult)
        assert result.adapter_name == "stub-engine"
        assert result.text == "hello world"
        assert result.wer is None
        assert result.cer is None
        assert result.wall_time > 0
        assert result.audio_duration == 5.0
        assert result.language == "en"

    def test_run_with_perfect_reference(self):
        runner = BenchmarkRunner()
        adapter = StubAdapter(text="hello world")
        result = runner.run(adapter, "/fake/audio.wav", reference="hello world")

        assert result.wer == pytest.approx(0.0)
        assert result.cer == pytest.approx(0.0)

    def test_run_with_partial_reference(self):
        runner = BenchmarkRunner()
        adapter = StubAdapter(text="hello earth")
        result = runner.run(adapter, "/fake/audio.wav", reference="hello world")

        assert result.wer == pytest.approx(0.5)

    def test_rtf_computed(self):
        runner = BenchmarkRunner()
        # duration=5.0, wall_time will be ~0s (stub), so RTF >> 1 is expected
        adapter = StubAdapter(duration=5.0, processing_time=0.5)
        result = runner.run(adapter, "/fake/audio.wav")

        assert result.rtf is not None
        assert result.rtf > 0

    def test_rtf_none_when_duration_missing(self):
        runner = BenchmarkRunner()

        class NoDurationAdapter(StubAdapter):
            def transcribe(self, audio_path, language=None):
                r = super().transcribe(audio_path, language)
                r.duration = None
                return r

        result = runner.run(NoDurationAdapter(), "/fake/audio.wav")
        assert result.rtf is None

    def test_first_token_latency_propagated(self):
        runner = BenchmarkRunner()
        adapter = StubAdapter(first_token_latency=0.123)
        result = runner.run(adapter, "/fake/audio.wav")
        assert result.first_token_latency == pytest.approx(0.123)

    def test_first_token_latency_none_when_not_supported(self):
        runner = BenchmarkRunner()
        adapter = StubAdapter(first_token_latency=None)
        result = runner.run(adapter, "/fake/audio.wav")
        assert result.first_token_latency is None

    def test_segments_populated(self):
        runner = BenchmarkRunner()
        adapter = StubAdapter(text="hello world", duration=5.0)
        result = runner.run(adapter, "/fake/audio.wav")
        assert len(result.segments) == 1
        assert result.segments[0].text == "hello world"

    def test_vram_peak_none_without_cuda(self):
        runner = BenchmarkRunner()
        adapter = StubAdapter()
        result = runner.run(adapter, "/fake/audio.wav")
        # In CI without GPU, vram_peak_mb should be None (no CUDA available)
        assert result.vram_peak_mb is None


# ---------------------------------------------------------------------------
# MetricsCollector tests
# ---------------------------------------------------------------------------

class TestMetricsCollector:
    def _make_result(self, wer=0.1, cer=0.05, wall_time=1.0, duration=5.0, vram_mb=None):
        return RunResult(
            adapter_name="stub-engine",
            audio_path="/fake/audio.wav",
            text="hello world",
            segments=[],
            wer=wer,
            cer=cer,
            wall_time=wall_time,
            audio_duration=duration,
            rtf=(duration / wall_time) if duration and wall_time > 0 else None,
            vram_peak_mb=vram_mb,
        )

    def test_add_and_aggregate(self):
        collector = MetricsCollector()
        collector.add_result(self._make_result(wer=0.1, wall_time=1.0))
        collector.add_result(self._make_result(wer=0.2, wall_time=2.0))

        metrics = collector.aggregate(
            engine="stub",
            model="large",
            results=collector.results,
        )

        assert metrics.engine == "stub"
        assert metrics.model == "large"
        assert metrics.run_count == 2
        assert metrics.wer_avg == pytest.approx(0.15)
        assert metrics.wer_min == pytest.approx(0.1)
        assert metrics.wer_max == pytest.approx(0.2)
        assert metrics.wall_time_avg == pytest.approx(1.5)
        assert metrics.wall_time_min == pytest.approx(1.0)
        assert metrics.wall_time_max == pytest.approx(2.0)

    def test_aggregate_with_no_wer(self):
        collector = MetricsCollector()
        collector.add_result(self._make_result(wer=None, cer=None))

        metrics = collector.aggregate("stub", "small", collector.results)
        assert metrics.wer_avg is None
        assert metrics.cer_avg is None

    def test_aggregate_empty_raises(self):
        collector = MetricsCollector()
        with pytest.raises(ValueError, match="empty"):
            collector.aggregate("stub", "small", [])

    def test_vram_peak_max(self):
        collector = MetricsCollector()
        collector.add_result(self._make_result(vram_mb=1024))
        collector.add_result(self._make_result(vram_mb=2048))
        collector.add_result(self._make_result(vram_mb=1500))

        metrics = collector.aggregate("stub", "large", collector.results)
        assert metrics.vram_peak_mb == 2048

    def test_vram_peak_none_when_missing(self):
        collector = MetricsCollector()
        collector.add_result(self._make_result(vram_mb=None))

        metrics = collector.aggregate("stub", "large", collector.results)
        assert metrics.vram_peak_mb is None

    def test_first_token_latency_avg(self):
        results = [
            RunResult(
                adapter_name="stub",
                audio_path="/fake/audio.wav",
                text="hi",
                segments=[],
                wall_time=1.0,
                first_token_latency=0.1,
            ),
            RunResult(
                adapter_name="stub",
                audio_path="/fake/audio.wav",
                text="hi",
                segments=[],
                wall_time=1.0,
                first_token_latency=0.3,
            ),
        ]
        collector = MetricsCollector(results=results)
        metrics = collector.aggregate("stub", "small", results)
        assert metrics.first_token_latency_avg == pytest.approx(0.2)

    def test_feature_support_fields(self):
        collector = MetricsCollector()
        collector.add_result(self._make_result())

        metrics = collector.aggregate(
            "stub",
            "large",
            collector.results,
            language_detection=True,
            word_timestamps=True,
            translation=False,
            diarization_ready=True,
        )
        assert metrics.language_detection is True
        assert metrics.word_timestamps is True
        assert metrics.translation is False
        assert metrics.diarization_ready is True

    def test_model_load_time_stored(self):
        collector = MetricsCollector()
        collector.add_result(self._make_result())
        metrics = collector.aggregate("stub", "large", collector.results, model_load_time=5.2)
        assert metrics.model_load_time == pytest.approx(5.2)

    def test_save_and_load_json(self, tmp_path):
        collector = MetricsCollector()
        collector.add_result(self._make_result(wer=0.05))
        metrics = collector.aggregate("stub", "large", collector.results)

        output_path = tmp_path / "results.json"
        collector.save_json(output_path, [metrics])

        assert output_path.exists()
        data = MetricsCollector.load_json(output_path)
        assert "generated_at" in data
        assert len(data["engines"]) == 1
        assert data["engines"][0]["engine"] == "stub"
        assert data["engines"][0]["model"] == "large"

    def test_save_creates_parent_dirs(self, tmp_path):
        collector = MetricsCollector()
        collector.add_result(self._make_result())
        metrics = collector.aggregate("stub", "large", collector.results)

        nested_path = tmp_path / "deep" / "dir" / "results.json"
        collector.save_json(nested_path, [metrics])
        assert nested_path.exists()

    def test_rtf_avg(self):
        collector = MetricsCollector()
        # duration=10, wall_time=2 → RTF=5
        collector.add_result(self._make_result(wall_time=2.0, duration=10.0))
        # duration=10, wall_time=5 → RTF=2
        collector.add_result(self._make_result(wall_time=5.0, duration=10.0))

        metrics = collector.aggregate("stub", "large", collector.results)
        assert metrics.rtf_avg == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# Report generator tests
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def _metrics_dict(self, engine="faster-whisper", model="large-v3", **overrides):
        base = {
            "engine": engine,
            "model": model,
            "wer_avg": 0.05,
            "wer_min": 0.03,
            "wer_max": 0.08,
            "cer_avg": 0.02,
            "wall_time_avg": 3.5,
            "wall_time_min": 3.0,
            "wall_time_max": 4.0,
            "first_token_latency_avg": None,
            "rtf_avg": 8.5,
            "vram_peak_mb": 4096,
            "model_load_time": 12.3,
            "language_detection": True,
            "word_timestamps": True,
            "translation": True,
            "diarization_ready": True,
            "run_count": 3,
        }
        base.update(overrides)
        return base

    def test_empty_input(self):
        report = generate_report([])
        assert "No results" in report

    def test_report_contains_engine_name(self):
        report = generate_report([self._metrics_dict()])
        assert "faster-whisper" in report

    def test_report_contains_model(self):
        report = generate_report([self._metrics_dict()])
        assert "large-v3" in report

    def test_report_contains_wer(self):
        report = generate_report([self._metrics_dict(wer_avg=0.05)])
        assert "0.050" in report

    def test_report_contains_rtf(self):
        report = generate_report([self._metrics_dict(rtf_avg=8.5)])
        assert "8.50" in report

    def test_report_feature_check_marks(self):
        report = generate_report([self._metrics_dict(
            language_detection=True,
            word_timestamps=True,
            translation=False,
            diarization_ready=True,
        )])
        assert "✓" in report
        assert "✗" in report

    def test_report_has_three_sections(self):
        report = generate_report([self._metrics_dict()])
        assert "## Performance Comparison" in report
        assert "## Feature Support Matrix" in report
        assert "## Latency Details" in report

    def test_none_values_formatted_as_dash(self):
        report = generate_report([self._metrics_dict(
            wer_avg=None,
            rtf_avg=None,
            vram_peak_mb=None,
            model_load_time=None,
            first_token_latency_avg=None,
        )])
        assert "—" in report

    def test_multiple_engines_sorted_by_wer(self):
        metrics = [
            self._metrics_dict(engine="engine-b", wer_avg=0.10),
            self._metrics_dict(engine="engine-a", wer_avg=0.05),
        ]
        report = generate_report(metrics)
        # engine-a (lower WER) should appear before engine-b in performance table
        perf_section = report.split("## Feature Support")[0]
        assert perf_section.index("engine-a") < perf_section.index("engine-b")

    def test_report_written_to_file(self, tmp_path):
        out = tmp_path / "report.md"
        out.write_text(generate_report([self._metrics_dict()]), encoding="utf-8")
        content = out.read_text(encoding="utf-8")
        assert "faster-whisper" in content

    def test_report_cli_creates_file(self, tmp_path):
        """report.main() should write a report.md into the results directory."""
        from tests.benchmark.report import main as report_main

        # Create a minimal JSON file the CLI can pick up
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        json_file = results_dir / "stub.json"
        collector = MetricsCollector()
        results = [
            RunResult(
                adapter_name="stub",
                audio_path="/fake/audio.wav",
                text="hi",
                segments=[],
                wall_time=1.0,
                wer=0.1,
            )
        ]
        metrics = collector.aggregate("stub", "small", results)
        collector.save_json(json_file, [metrics])

        rc = report_main(["--input", str(results_dir), "--output", str(tmp_path / "report.md")])
        assert rc == 0
        assert (tmp_path / "report.md").exists()

    def test_report_cli_missing_input_dir(self, tmp_path):
        from tests.benchmark.report import main as report_main

        rc = report_main(["--input", str(tmp_path / "nonexistent")])
        assert rc == 1


# ---------------------------------------------------------------------------
# Manifest / dataset tests (issue #2)
# ---------------------------------------------------------------------------

class TestDataManifest:
    @pytest.fixture
    def manifest(self):
        manifest_path = Path(__file__).parent / "data" / "manifest.json"
        with manifest_path.open(encoding="utf-8") as fh:
            return json.load(fh)

    def test_manifest_is_valid_json(self, manifest):
        assert "samples" in manifest
        assert isinstance(manifest["samples"], list)

    def test_manifest_has_enough_samples(self, manifest):
        assert len(manifest["samples"]) >= 10, "Dataset must have at least 10 samples"

    def test_each_sample_has_required_fields(self, manifest):
        required = {"id", "filename", "reference", "category", "language", "duration_s"}
        for sample in manifest["samples"]:
            missing = required - set(sample.keys())
            assert not missing, f"Sample {sample.get('id')} missing fields: {missing}"

    def test_sample_ids_are_unique(self, manifest):
        ids = [s["id"] for s in manifest["samples"]]
        assert len(ids) == len(set(ids)), "Sample IDs must be unique"

    def test_filenames_are_unique(self, manifest):
        filenames = [s["filename"] for s in manifest["samples"]]
        assert len(filenames) == len(set(filenames)), "Filenames must be unique"

    def test_categories_are_known(self, manifest):
        known = {
            "clean_single_speaker",
            "noisy",
            "accented_english",
            "non_english",
            "multi_speaker",
            "long_form",
        }
        for sample in manifest["samples"]:
            assert sample["category"] in known, (
                f"Sample {sample['id']} has unknown category: {sample['category']}"
            )

    def test_durations_are_positive(self, manifest):
        for sample in manifest["samples"]:
            assert sample["duration_s"] > 0, (
                f"Sample {sample['id']} has non-positive duration"
            )

    def test_covers_non_english_languages(self, manifest):
        languages = {s["language"] for s in manifest["samples"]}
        non_english = languages - {"en"}
        assert non_english, "Dataset must include at least one non-English sample"

    def test_covers_long_form(self, manifest):
        long_form = [s for s in manifest["samples"] if s["category"] == "long_form"]
        assert long_form, "Dataset must include at least one long-form sample"
        assert any(s["duration_s"] >= 60 for s in long_form), (
            "At least one long-form sample must be >= 60 seconds"
        )


# ---------------------------------------------------------------------------
# Engine adapter contract tests (issues #6–#10)
#
# These tests validate that each concrete adapter implementation conforms to
# the BaseAdapter contract WITHOUT requiring any real models or GPU resources.
# The adapters are tested in isolation using unittest.mock to intercept the
# underlying engine calls, so the full test suite runs in CI without any
# optional dependencies installed.
# ---------------------------------------------------------------------------

import sys
import types
import unittest.mock as mock
from typing import Any


def _make_fake_fw_model(text="hello world", language="en", duration=5.0):
    """Return a mock faster-whisper WhisperModel."""
    fake_seg = mock.MagicMock()
    fake_seg.id = 0
    fake_seg.start = 0.0
    fake_seg.end = duration
    fake_seg.text = f" {text}"
    fake_word = mock.MagicMock()
    fake_word.word = text.split()[0]
    fake_word.start = 0.0
    fake_word.end = 0.5
    fake_word.probability = 0.99
    fake_seg.words = [fake_word]

    fake_info = mock.MagicMock()
    fake_info.language = language
    fake_info.language_probability = 0.98
    fake_info.duration = duration

    model = mock.MagicMock()
    model.transcribe.return_value = (iter([fake_seg]), fake_info)
    return model


def _mock_faster_whisper_module(fake_model: Any = None) -> dict:
    """Return a sys.modules patch dict that stubs out ``faster_whisper``."""
    if fake_model is None:
        fake_model = _make_fake_fw_model()
    fake_fw = types.ModuleType("faster_whisper")
    fake_fw.WhisperModel = mock.MagicMock(return_value=fake_model)
    return {"faster_whisper": fake_fw}


class TestFasterWhisperAdapter:
    """Contract tests for FasterWhisperAdapter (issue #6)."""

    def _make_adapter(self):
        from tests.benchmark.adapters.faster_whisper import FasterWhisperAdapter
        return FasterWhisperAdapter(model_size="large-v3", device="cpu")

    def test_name(self):
        adapter = self._make_adapter()
        assert adapter.name == "faster-whisper"

    def test_features(self):
        adapter = self._make_adapter()
        assert adapter.features.language_detection
        assert adapter.features.word_timestamps
        assert adapter.features.translation
        assert adapter.features.diarization_ready

    def test_load_returns_float(self):
        adapter = self._make_adapter()
        with mock.patch.dict(sys.modules, _mock_faster_whisper_module()):
            load_time = adapter.load()
        assert isinstance(load_time, float)
        assert load_time >= 0

    def test_transcribe_without_load_raises(self):
        adapter = self._make_adapter()
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            adapter.transcribe("/fake/audio.wav")

    def test_transcribe_returns_adapter_result(self):
        from tests.benchmark.adapters.base import AdapterResult

        adapter = self._make_adapter()
        fake_model = _make_fake_fw_model(text="hello world", language="en", duration=5.0)
        with mock.patch.dict(sys.modules, _mock_faster_whisper_module(fake_model)):
            adapter.load()
        result = adapter.transcribe("/fake/audio.wav")

        assert isinstance(result, AdapterResult)
        assert "hello world" in result.text
        assert result.language == "en"
        assert result.processing_time >= 0
        assert result.duration == pytest.approx(5.0)
        assert len(result.segments) == 1

    def test_transcribe_with_language(self):
        adapter = self._make_adapter()
        fake_model = _make_fake_fw_model()
        with mock.patch.dict(sys.modules, _mock_faster_whisper_module(fake_model)):
            adapter.load()
        adapter.transcribe("/fake/audio.wav", language="en")
        _, call_kwargs = fake_model.transcribe.call_args
        assert call_kwargs.get("language") == "en"

    def test_unload_clears_model(self):
        adapter = self._make_adapter()
        with mock.patch.dict(sys.modules, _mock_faster_whisper_module()):
            adapter.load()
        assert adapter._model is not None
        adapter.unload()
        assert adapter._model is None

    def test_unload_before_load_is_safe(self):
        adapter = self._make_adapter()
        adapter.unload()  # must not raise


class TestWhisperCppAdapter:
    """Contract tests for WhisperCppAdapter (issue #7)."""

    def _make_adapter(self):
        from tests.benchmark.adapters.whisper_cpp import WhisperCppAdapter
        return WhisperCppAdapter(model_path="/fake/model.bin")

    def _make_fake_cpp_model(self, text="hello world"):
        fake_seg = mock.MagicMock()
        fake_seg.t0 = 0
        fake_seg.t1 = 500
        fake_seg.text = text
        fake_seg.tokens = None
        fake_seg.words = None
        model = mock.MagicMock()
        model.transcribe.return_value = [fake_seg]
        model.lang_str = "en"
        return model

    def test_name(self):
        assert self._make_adapter().name == "whisper.cpp"

    def test_features(self):
        adapter = self._make_adapter()
        assert adapter.features.language_detection
        assert adapter.features.word_timestamps
        assert adapter.features.translation

    def test_load_returns_float(self):
        adapter = self._make_adapter()
        fake_module = types.ModuleType("pywhispercpp")
        fake_model_cls = mock.MagicMock(return_value=self._make_fake_cpp_model())
        fake_module.model = types.SimpleNamespace(Model=fake_model_cls)
        with mock.patch.dict(sys.modules, {"pywhispercpp": fake_module, "pywhispercpp.model": fake_module.model}):
            load_time = adapter.load()
        assert isinstance(load_time, float)
        assert load_time >= 0

    def test_transcribe_without_load_raises(self):
        adapter = self._make_adapter()
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            adapter.transcribe("/fake/audio.wav")

    def test_transcribe_returns_adapter_result(self):
        from tests.benchmark.adapters.base import AdapterResult

        adapter = self._make_adapter()
        fake_cpp_model = self._make_fake_cpp_model("hello world")
        fake_module = types.ModuleType("pywhispercpp")
        fake_module.model = types.SimpleNamespace(Model=mock.MagicMock(return_value=fake_cpp_model))
        with mock.patch.dict(sys.modules, {"pywhispercpp": fake_module, "pywhispercpp.model": fake_module.model}):
            adapter.load()
        result = adapter.transcribe("/fake/audio.wav")

        assert isinstance(result, AdapterResult)
        assert result.text == "hello world"
        assert result.language == "en"
        assert result.processing_time >= 0

    def test_unload_clears_model(self):
        adapter = self._make_adapter()
        fake_module = types.ModuleType("pywhispercpp")
        fake_module.model = types.SimpleNamespace(Model=mock.MagicMock(return_value=self._make_fake_cpp_model()))
        with mock.patch.dict(sys.modules, {"pywhispercpp": fake_module, "pywhispercpp.model": fake_module.model}):
            adapter.load()
        assert adapter._model is not None
        adapter.unload()
        assert adapter._model is None


class TestOpenAIWhisperAdapter:
    """Contract tests for OpenAIWhisperAdapter (issue #8)."""

    def _make_adapter(self):
        from tests.benchmark.adapters.openai_whisper import OpenAIWhisperAdapter
        return OpenAIWhisperAdapter(model_size="large-v3", device="cpu")

    def _fake_raw_result(self, text="hello world"):
        return {
            "text": f" {text}",
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 3.5,
                    "text": f" {text}",
                    "words": [
                        {"word": w, "start": 0.0, "end": 0.5, "probability": 0.98}
                        for w in text.split()
                    ],
                }
            ],
        }

    def _mock_whisper_module(self, fake_model=None):
        if fake_model is None:
            fake_model = mock.MagicMock()
        fake_whisper = types.ModuleType("whisper")
        fake_whisper.load_model = mock.MagicMock(return_value=fake_model)
        return {"whisper": fake_whisper}

    def test_name(self):
        assert self._make_adapter().name == "openai-whisper"

    def test_features(self):
        adapter = self._make_adapter()
        assert adapter.features.language_detection
        assert adapter.features.word_timestamps
        assert adapter.features.translation
        assert adapter.features.diarization_ready

    def test_load_returns_float(self):
        adapter = self._make_adapter()
        with mock.patch.dict(sys.modules, self._mock_whisper_module()):
            load_time = adapter.load()
        assert isinstance(load_time, float)
        assert load_time >= 0

    def test_transcribe_without_load_raises(self):
        adapter = self._make_adapter()
        fake_whisper = types.ModuleType("whisper")
        with mock.patch.dict(sys.modules, {"whisper": fake_whisper}):
            with pytest.raises(RuntimeError, match="load\\(\\)"):
                adapter.transcribe("/fake/audio.wav")

    def test_transcribe_returns_adapter_result(self):
        from tests.benchmark.adapters.base import AdapterResult

        adapter = self._make_adapter()
        fake_model = mock.MagicMock()
        fake_model.transcribe.return_value = self._fake_raw_result("hello world")

        with mock.patch.dict(sys.modules, self._mock_whisper_module(fake_model)):
            adapter.load()
        result = adapter.transcribe("/fake/audio.wav")

        assert isinstance(result, AdapterResult)
        assert "hello world" in result.text
        assert result.language == "en"
        assert result.processing_time >= 0
        assert len(result.segments) == 1
        assert result.segments[0].words is not None

    def test_unload_clears_model(self):
        adapter = self._make_adapter()
        with mock.patch.dict(sys.modules, self._mock_whisper_module()):
            adapter.load()
        assert adapter._model is not None
        adapter.unload()
        assert adapter._model is None


class TestDistilWhisperAdapter:
    """Contract tests for DistilWhisperAdapter (issue #9)."""

    def _make_adapter(self, backend="faster-whisper"):
        from tests.benchmark.adapters.distil_whisper import DistilWhisperAdapter
        return DistilWhisperAdapter(
            model_id="distil-large-v3",
            backend=backend,
            device="cpu",
        )

    def test_name(self):
        assert self._make_adapter().name == "distil-whisper"

    def test_invalid_backend_raises(self):
        from tests.benchmark.adapters.distil_whisper import DistilWhisperAdapter
        with pytest.raises(ValueError, match="backend"):
            DistilWhisperAdapter(backend="invalid-backend")

    def test_features_diarization_ready(self):
        adapter = self._make_adapter()
        assert adapter.features.diarization_ready

    def test_load_faster_whisper_backend_returns_float(self):
        adapter = self._make_adapter(backend="faster-whisper")
        with mock.patch.dict(sys.modules, _mock_faster_whisper_module()):
            load_time = adapter.load()
        assert isinstance(load_time, float)
        assert load_time >= 0

    def test_transcribe_faster_whisper_backend(self):
        from tests.benchmark.adapters.base import AdapterResult

        adapter = self._make_adapter(backend="faster-whisper")
        fake_model = _make_fake_fw_model(text="test transcription", language="en", duration=3.0)
        with mock.patch.dict(sys.modules, _mock_faster_whisper_module(fake_model)):
            adapter.load()
        result = adapter.transcribe("/fake/audio.wav")

        assert isinstance(result, AdapterResult)
        assert "test transcription" in result.text
        assert result.language == "en"

    def test_transcribe_without_load_raises(self):
        adapter = self._make_adapter(backend="faster-whisper")
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            adapter.transcribe("/fake/audio.wav")

    def test_unload_clears_model(self):
        adapter = self._make_adapter(backend="faster-whisper")
        with mock.patch.dict(sys.modules, _mock_faster_whisper_module()):
            adapter.load()
        assert adapter._model is not None
        adapter.unload()
        assert adapter._model is None

    def test_transcribe_huggingface_backend(self):
        from tests.benchmark.adapters.base import AdapterResult
        from tests.benchmark.adapters.distil_whisper import DistilWhisperAdapter

        adapter = DistilWhisperAdapter(
            model_id="distil-whisper/distil-large-v3",
            backend="huggingface",
            device="cpu",
        )
        fake_pipeline_result = {
            "text": "hello world",
            "chunks": [{"text": "hello world", "timestamp": (0.0, 3.5)}],
        }
        fake_pipe_fn = mock.MagicMock(return_value=fake_pipeline_result)
        adapter._pipeline = fake_pipe_fn
        result = adapter.transcribe("/fake/audio.wav")

        assert isinstance(result, AdapterResult)
        assert result.text == "hello world"
        assert len(result.segments) == 1
        assert result.segments[0].start == pytest.approx(0.0)
        assert result.segments[0].end == pytest.approx(3.5)


class TestWhisperJaxAdapter:
    """Contract tests for WhisperJaxAdapter (issue #10, optional)."""

    def _make_adapter(self):
        from tests.benchmark.adapters.whisper_jax import WhisperJaxAdapter
        return WhisperJaxAdapter(model_size="large-v2")

    def test_name(self):
        assert self._make_adapter().name == "whisper-jax"

    def test_features_word_timestamps_not_supported(self):
        adapter = self._make_adapter()
        assert not adapter.features.word_timestamps

    def test_load_raises_import_error_when_not_installed(self):
        """WhisperJaxAdapter.load() must raise ImportError if whisper_jax is missing."""
        adapter = self._make_adapter()
        with mock.patch.dict(sys.modules, {"whisper_jax": None, "jax": None, "jax.numpy": None}):
            with pytest.raises((ImportError, TypeError)):
                adapter.load()

    def test_transcribe_without_load_raises(self):
        adapter = self._make_adapter()
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            adapter.transcribe("/fake/audio.wav")

    def test_transcribe_returns_adapter_result(self):
        from tests.benchmark.adapters.base import AdapterResult
        from tests.benchmark.adapters.whisper_jax import WhisperJaxAdapter

        adapter = WhisperJaxAdapter(model_size="large-v2")
        fake_result = {
            "text": "hello world",
            "chunks": [{"text": "hello world", "timestamp": (0.0, 4.0)}],
        }
        fake_pipe = mock.MagicMock(return_value=fake_result)
        adapter._pipeline = fake_pipe
        result = adapter.transcribe("/fake/audio.wav")

        assert isinstance(result, AdapterResult)
        assert result.text == "hello world"
        assert len(result.segments) == 1
        assert result.duration == pytest.approx(4.0)

    def test_unload_clears_pipeline(self):
        from tests.benchmark.adapters.whisper_jax import WhisperJaxAdapter

        adapter = WhisperJaxAdapter(model_size="large-v2")
        adapter._pipeline = mock.MagicMock()
        adapter.unload()
        assert adapter._pipeline is None


class TestAdapterRegistration:
    """Verify all adapters are importable from the adapters package."""

    def test_all_adapters_importable(self):
        from tests.benchmark.adapters import (  # noqa: F401
            DistilWhisperAdapter,
            FasterWhisperAdapter,
            OpenAIWhisperAdapter,
            WhisperCppAdapter,
            WhisperJaxAdapter,
        )

    def test_all_adapters_are_base_adapter_subclasses(self):
        from tests.benchmark.adapters import (
            BaseAdapter,
            DistilWhisperAdapter,
            FasterWhisperAdapter,
            OpenAIWhisperAdapter,
            WhisperCppAdapter,
            WhisperJaxAdapter,
        )
        for cls in (
            FasterWhisperAdapter,
            WhisperCppAdapter,
            OpenAIWhisperAdapter,
            DistilWhisperAdapter,
            WhisperJaxAdapter,
        ):
            assert issubclass(cls, BaseAdapter), f"{cls.__name__} must extend BaseAdapter"

    def test_all_adapters_have_name(self):
        from tests.benchmark.adapters import (
            DistilWhisperAdapter,
            FasterWhisperAdapter,
            OpenAIWhisperAdapter,
            WhisperCppAdapter,
            WhisperJaxAdapter,
        )
        for cls in (
            FasterWhisperAdapter,
            WhisperCppAdapter,
            OpenAIWhisperAdapter,
            DistilWhisperAdapter,
            WhisperJaxAdapter,
        ):
            assert isinstance(cls.name, str) and cls.name != "unknown", (
                f"{cls.__name__}.name must be set to a non-default string"
            )

    def test_all_adapters_have_feature_support(self):
        from tests.benchmark.adapters import (
            BaseAdapter,
            DistilWhisperAdapter,
            FeatureSupport,
            FasterWhisperAdapter,
            OpenAIWhisperAdapter,
            WhisperCppAdapter,
            WhisperJaxAdapter,
        )
        for cls in (
            FasterWhisperAdapter,
            WhisperCppAdapter,
            OpenAIWhisperAdapter,
            DistilWhisperAdapter,
            WhisperJaxAdapter,
        ):
            assert isinstance(cls.features, FeatureSupport), (
                f"{cls.__name__}.features must be a FeatureSupport instance"
            )


# ---------------------------------------------------------------------------
# FullBenchmarkSuite tests (issue #12)
# ---------------------------------------------------------------------------

class TestFullBenchmarkSuite:
    """Tests for the full suite orchestrator in tests.benchmark.suite."""

    # ---- StubAdapter that can optionally raise --------------------------

    class _SuccessAdapter(StubAdapter):
        name = "stub-success"

        def __init__(self, **kwargs):
            super().__init__()

        def load(self) -> float:
            return 0.0

    class _FailingAdapter(StubAdapter):
        name = "stub-failing"

        def __init__(self, **kwargs):
            super().__init__()

        def load(self) -> float:
            return 0.0

        def transcribe(self, audio_path: str, language=None):
            raise RuntimeError("deliberate transcribe failure")

    class _LoadFailAdapter(StubAdapter):
        name = "stub-load-fail"

        def __init__(self, **kwargs):
            super().__init__()

        def load(self) -> float:
            raise RuntimeError("deliberate load failure")

    # ---- Helpers --------------------------------------------------------

    def _make_manifest(self, tmp_path: Path) -> dict:
        """Write a minimal manifest.json with two samples."""
        wav1 = tmp_path / "sample_001.wav"
        wav2 = tmp_path / "sample_002.wav"
        ref1 = tmp_path / "sample_001.txt"
        ref2 = tmp_path / "sample_002.txt"
        wav1.write_bytes(b"")
        wav2.write_bytes(b"")
        ref1.write_text("hello world", encoding="utf-8")
        ref2.write_text("foo bar", encoding="utf-8")

        manifest = {
            "version": "1",
            "description": "test",
            "samples": [
                {
                    "id": "s001",
                    "filename": "sample_001.wav",
                    "reference": "sample_001.txt",
                    "category": "clean_single_speaker",
                    "language": "en",
                    "duration_s": 5.0,
                },
                {
                    "id": "s002",
                    "filename": "sample_002.wav",
                    "reference": "sample_002.txt",
                    "category": "noisy",
                    "language": "en",
                    "duration_s": 10.0,
                },
            ],
        }
        (tmp_path / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return manifest

    def _make_config(self, adapter_cls, model_size="stub-model"):
        from tests.benchmark.suite import AdapterConfig

        return AdapterConfig(
            adapter_cls=adapter_cls,
            model_size=model_size,
            model_kwarg="model_size",
            device="cpu",
            extra_kwargs={},
        )

    # ---- AdapterConfig tests -------------------------------------------

    def test_adapter_config_build_success(self):
        from tests.benchmark.suite import AdapterConfig

        cfg = AdapterConfig(
            adapter_cls=self._SuccessAdapter,
            model_size="test",
            model_kwarg="model_size",
            device="cpu",
        )
        adapter = cfg.build()
        assert isinstance(adapter, self._SuccessAdapter)

    # ---- Suite run tests -----------------------------------------------

    def test_suite_all_success(self, tmp_path):
        from tests.benchmark.suite import FullBenchmarkSuite

        self._make_manifest(tmp_path)
        output_dir = tmp_path / "results"
        cfg = self._make_config(self._SuccessAdapter)
        suite = FullBenchmarkSuite(
            configs=[cfg],
            data_dir=tmp_path,
            output_dir=output_dir,
            save_on_adapter_complete=False,
        )
        result = suite.run()

        assert result.total_runs == 2
        assert result.successful_runs == 2
        assert result.failure_count == 0
        assert result.success_rate == 1.0

    def test_suite_handles_transcribe_failure(self, tmp_path):
        from tests.benchmark.suite import FullBenchmarkSuite

        self._make_manifest(tmp_path)
        cfg = self._make_config(self._FailingAdapter)
        suite = FullBenchmarkSuite(
            configs=[cfg],
            data_dir=tmp_path,
            output_dir=tmp_path / "results",
            save_on_adapter_complete=False,
        )
        result = suite.run()

        assert result.total_runs == 2
        assert result.successful_runs == 0
        assert result.failure_count == 2
        failure = result.failures[0]
        assert failure.adapter_name == "stub-failing"
        assert failure.error_type == "RuntimeError"
        assert "transcribe" in failure.error_message

    def test_suite_handles_load_failure(self, tmp_path):
        from tests.benchmark.suite import FullBenchmarkSuite

        self._make_manifest(tmp_path)
        cfg = self._make_config(self._LoadFailAdapter)
        suite = FullBenchmarkSuite(
            configs=[cfg],
            data_dir=tmp_path,
            output_dir=tmp_path / "results",
            save_on_adapter_complete=False,
        )
        result = suite.run()

        # total_runs is 0 because we never reached the per-sample loop
        assert result.total_runs == 0
        assert result.failure_count == 1
        assert result.failures[0].sample_id == "<model_load>"

    def test_suite_multiple_configs(self, tmp_path):
        from tests.benchmark.suite import FullBenchmarkSuite

        self._make_manifest(tmp_path)
        cfgs = [
            self._make_config(self._SuccessAdapter, "model-a"),
            self._make_config(self._SuccessAdapter, "model-b"),
        ]
        suite = FullBenchmarkSuite(
            configs=cfgs,
            data_dir=tmp_path,
            output_dir=tmp_path / "results",
            save_on_adapter_complete=False,
        )
        result = suite.run()

        assert result.total_runs == 4
        assert result.successful_runs == 4
        assert result.failure_count == 0

    def test_suite_saves_json_on_adapter_complete(self, tmp_path):
        from tests.benchmark.suite import FullBenchmarkSuite

        self._make_manifest(tmp_path)
        output_dir = tmp_path / "results"
        cfg = self._make_config(self._SuccessAdapter, "my-model")
        suite = FullBenchmarkSuite(
            configs=[cfg],
            data_dir=tmp_path,
            output_dir=output_dir,
            save_on_adapter_complete=True,
        )
        suite.run()

        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) >= 1, "Expected at least one JSON result file"

    def test_suite_saves_failure_log(self, tmp_path):
        from tests.benchmark.suite import FullBenchmarkSuite

        self._make_manifest(tmp_path)
        cfg = self._make_config(self._FailingAdapter)
        output_dir = tmp_path / "results"
        suite = FullBenchmarkSuite(
            configs=[cfg],
            data_dir=tmp_path,
            output_dir=output_dir,
            save_on_adapter_complete=False,
        )
        suite.run()

        failure_log = output_dir / "failures.json"
        assert failure_log.exists()
        data = json.loads(failure_log.read_text(encoding="utf-8"))
        assert data["failure_count"] == 2
        assert len(data["failures"]) == 2

    def test_suite_no_failure_log_on_clean_run(self, tmp_path):
        from tests.benchmark.suite import FullBenchmarkSuite

        self._make_manifest(tmp_path)
        cfg = self._make_config(self._SuccessAdapter)
        output_dir = tmp_path / "results"
        suite = FullBenchmarkSuite(
            configs=[cfg],
            data_dir=tmp_path,
            output_dir=output_dir,
            save_on_adapter_complete=False,
        )
        suite.run()

        failure_log = output_dir / "failures.json"
        assert not failure_log.exists()

    def test_suite_results_by_adapter_key(self, tmp_path):
        from tests.benchmark.suite import FullBenchmarkSuite

        self._make_manifest(tmp_path)
        cfg = self._make_config(self._SuccessAdapter, "big-model")
        suite = FullBenchmarkSuite(
            configs=[cfg],
            data_dir=tmp_path,
            output_dir=tmp_path / "results",
            save_on_adapter_complete=False,
        )
        result = suite.run()

        expected_key = "stub-success/big-model"
        assert expected_key in result.results_by_adapter
        assert len(result.results_by_adapter[expected_key]) == 2

    def test_suite_mixed_success_and_failure(self, tmp_path):
        """One config succeeds, one fails — suite continues and records both."""
        from tests.benchmark.suite import FullBenchmarkSuite

        self._make_manifest(tmp_path)
        cfgs = [
            self._make_config(self._SuccessAdapter),
            self._make_config(self._FailingAdapter),
        ]
        suite = FullBenchmarkSuite(
            configs=cfgs,
            data_dir=tmp_path,
            output_dir=tmp_path / "results",
            save_on_adapter_complete=False,
        )
        result = suite.run()

        assert result.total_runs == 4
        assert result.successful_runs == 2
        assert result.failure_count == 2

    def test_run_failure_dataclass_fields(self):
        from tests.benchmark.suite import RunFailure

        f = RunFailure(
            adapter_name="eng",
            model_size="lg",
            sample_id="s1",
            audio_path="/f/a.wav",
            error_type="RuntimeError",
            error_message="boom",
            traceback="Traceback …",
        )
        assert f.adapter_name == "eng"
        assert f.error_type == "RuntimeError"

    def test_suite_result_success_rate(self):
        from tests.benchmark.suite import SuiteResult

        sr = SuiteResult(total_runs=10, successful_runs=8)
        assert sr.success_rate == pytest.approx(0.8)

    def test_suite_result_zero_runs(self):
        from tests.benchmark.suite import SuiteResult

        sr = SuiteResult()
        assert sr.success_rate == 0.0


# ---------------------------------------------------------------------------
# Report analysis section tests (issue #13)
# ---------------------------------------------------------------------------

class TestReportAnalysisSection:
    """Tests for the Analysis section added to generate_report (issue #13)."""

    def _m(self, engine="e", model="m", wer=0.1, rtf=5.0, vram=2048,
           lang=True, word_ts=True, translation=False, diar=True):
        return {
            "engine": engine,
            "model": model,
            "wer_avg": wer,
            "wer_min": wer,
            "wer_max": wer,
            "cer_avg": None,
            "wall_time_avg": 1.0,
            "wall_time_min": 0.8,
            "wall_time_max": 1.2,
            "first_token_latency_avg": None,
            "rtf_avg": rtf,
            "vram_peak_mb": vram,
            "model_load_time": None,
            "language_detection": lang,
            "word_timestamps": word_ts,
            "translation": translation,
            "diarization_ready": diar,
            "run_count": 2,
        }

    def test_report_has_analysis_section(self):
        report = generate_report([self._m()])
        assert "## Analysis" in report

    def test_best_wer_identified(self):
        metrics = [
            self._m(engine="a", wer=0.05),
            self._m(engine="b", wer=0.10),
        ]
        report = generate_report(metrics)
        assert "**Best WER:**" in report
        assert "a" in report

    def test_best_rtf_identified(self):
        metrics = [
            self._m(engine="fast", rtf=20.0),
            self._m(engine="slow", rtf=5.0),
        ]
        report = generate_report(metrics)
        assert "**Best Speed (RTF):**" in report
        assert "fast" in report

    def test_best_vram_identified(self):
        metrics = [
            self._m(engine="small", vram=512),
            self._m(engine="large", vram=4096),
        ]
        report = generate_report(metrics)
        assert "**Best VRAM Efficiency:**" in report
        assert "small" in report

    def test_best_diar_pairing_identified(self):
        metrics = [
            self._m(engine="diar-small", vram=512,  word_ts=True, diar=True),
            self._m(engine="diar-large", vram=4096, word_ts=True, diar=True),
        ]
        report = generate_report(metrics)
        assert "**Best Diarization Pairing:**" in report
        assert "diar-small" in report

    def test_diar_pairing_absent_when_no_eligible_engine(self):
        metrics = [self._m(engine="no-diar", word_ts=False, diar=False)]
        report = generate_report(metrics)
        assert "**Best Diarization Pairing:**" in report
        assert "no engine" in report.lower() or "—" in report

    def test_no_analysis_section_when_empty(self):
        report = generate_report([])
        assert "No results" in report

    def test_dominated_engine_detected(self):
        """Engine B is dominated when A beats it on all available axes."""
        metrics = [
            # Engine A: better WER, better RTF, better VRAM
            self._m(engine="A", wer=0.05, rtf=10.0, vram=1000),
            # Engine B: worse on all axes
            self._m(engine="B", wer=0.15, rtf=5.0,  vram=3000),
        ]
        report = generate_report(metrics)
        assert "Strictly Dominated" in report
        assert "B" in report

    def test_no_dominated_engines_when_trade_offs(self):
        """No engine should be flagged as dominated if there are trade-offs."""
        metrics = [
            # A: better WER but worse VRAM
            self._m(engine="A", wer=0.05, vram=4000, rtf=8.0),
            # B: worse WER but better VRAM
            self._m(engine="B", wer=0.15, vram=500,  rtf=8.0),
        ]
        report = generate_report(metrics)
        assert "Strictly Dominated" not in report

    def test_wer_none_handled_gracefully(self):
        metrics = [self._m(wer=None)]
        report = generate_report(metrics)
        assert "## Analysis" in report
        assert "no WER data" in report.lower() or "—" in report

    def test_rtf_none_handled_gracefully(self):
        metrics = [self._m(rtf=None)]
        report = generate_report(metrics)
        assert "## Analysis" in report

    def test_vram_none_handled_gracefully(self):
        metrics = [self._m(vram=None)]
        report = generate_report(metrics)
        assert "## Analysis" in report


# ---------------------------------------------------------------------------
# Co-tenancy benchmark tests (issue #14)
# ---------------------------------------------------------------------------

class TestCoTenancy:
    """Tests for tests.benchmark.cotenancy."""

    def _make_adapter(self):
        return StubAdapter(text="hello world", processing_time=0.01, duration=5.0)

    def _make_sample(self, sample_id="s001", audio_path="/fake/audio.wav"):
        return {
            "sample_id": sample_id,
            "audio_path": audio_path,
            "reference": "hello world",
        }

    # ---- CoTenancyScenario -------------------------------------------

    def test_scenario_defaults(self):
        from tests.benchmark.cotenancy import CoTenancyScenario

        sc = CoTenancyScenario("baseline")
        assert sc.name == "baseline"
        assert sc.tenant_vram_mb == 0
        assert sc.description == ""

    def test_scenario_with_vram(self):
        from tests.benchmark.cotenancy import CoTenancyScenario

        sc = CoTenancyScenario("pyannote", tenant_vram_mb=1500, description="pyannote idle")
        assert sc.tenant_vram_mb == 1500

    # ---- VramTenant context manager (no CUDA in CI) ------------------

    def test_vram_tenant_zero_is_noop(self):
        from tests.benchmark.cotenancy import _VramTenant

        with _VramTenant(0):
            pass  # must not raise

    def test_vram_tenant_nonzero_noop_without_cuda(self):
        from tests.benchmark.cotenancy import _VramTenant

        # In CI without GPU this should silently succeed (no CUDA available)
        with _VramTenant(512):
            pass

    # ---- OOM detection -------------------------------------------

    def test_is_oom_error_runtime(self):
        from tests.benchmark.cotenancy import _is_oom_error

        exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        assert _is_oom_error(exc)

    def test_is_oom_error_false_for_other(self):
        from tests.benchmark.cotenancy import _is_oom_error

        exc = ValueError("some other error")
        assert not _is_oom_error(exc)

    # ---- CoTenancyBenchmark.run() ------------------------------------

    def _make_benchmark(self, adapter_instance, sample=None, extra_scenarios=None):
        from tests.benchmark.cotenancy import CoTenancyBenchmark, CoTenancyScenario

        class _DirectConfig:
            """Helper: adapter_configs entry that wraps a pre-built instance."""

        sample = sample or self._make_sample()
        scenarios = [CoTenancyScenario("baseline", tenant_vram_mb=0)]
        if extra_scenarios:
            scenarios += extra_scenarios

        bench = CoTenancyBenchmark(
            adapter_configs=[],
            scenarios=scenarios,
            audio_samples=[sample],
        )
        # Inject the pre-built adapter directly via monkey-patching _build_adapter
        bench._build_adapter = lambda cfg: adapter_instance
        # Provide a single config dict so the outer loop fires once
        bench.adapter_configs = [{"cls": type(adapter_instance), "model_size": "stub"}]
        return bench

    def test_run_baseline_success(self):
        from tests.benchmark.cotenancy import CoTenancyBenchmark, CoTenancyScenario, OutcomeKind

        adapter = self._make_adapter()
        bench = self._make_benchmark(adapter)
        suite = bench.run()

        assert suite.total == 1
        assert suite.success_count == 1
        assert suite.oom_count == 0
        run = suite.runs[0]
        assert run.outcome == OutcomeKind.SUCCESS
        assert run.scenario_name == "baseline"
        assert run.wall_time is not None and run.wall_time >= 0
        # baseline has no slowdown factor because there's no prior baseline
        assert run.slowdown_factor is None

    def test_run_with_two_scenarios_computes_slowdown(self):
        from tests.benchmark.cotenancy import CoTenancyBenchmark, CoTenancyScenario, OutcomeKind

        adapter = self._make_adapter()
        bench = self._make_benchmark(
            adapter,
            extra_scenarios=[CoTenancyScenario("contention", tenant_vram_mb=0)],
        )
        suite = bench.run()

        # 2 scenarios × 1 sample × 1 adapter = 2 runs
        assert suite.total == 2
        # The second run should have a slowdown_factor set (baseline is from run 1)
        second_run = suite.runs[1]
        assert second_run.slowdown_factor is not None

    def test_run_captures_oom(self):
        """An OOM exception should be recorded with outcome=oom."""
        from tests.benchmark.cotenancy import (
            CoTenancyBenchmark,
            CoTenancyScenario,
            OutcomeKind,
        )

        class OOMAdapter(StubAdapter):
            name = "oom-adapter"

            def load(self):
                return 0.0

            def transcribe(self, audio_path, language=None):
                raise RuntimeError("CUDA out of memory. Tried to allocate 3.00 GiB")

        adapter = OOMAdapter()
        bench = self._make_benchmark(adapter)
        suite = bench.run()

        assert suite.oom_count == 1
        assert suite.runs[0].outcome == OutcomeKind.OOM

    def test_run_captures_generic_error(self):
        from tests.benchmark.cotenancy import CoTenancyBenchmark, CoTenancyScenario, OutcomeKind

        class ErrAdapter(StubAdapter):
            name = "err-adapter"

            def load(self):
                return 0.0

            def transcribe(self, audio_path, language=None):
                raise ValueError("something unrelated to OOM")

        adapter = ErrAdapter()
        bench = self._make_benchmark(adapter)
        suite = bench.run()

        assert suite.error_count == 1
        assert suite.runs[0].outcome == OutcomeKind.ERROR

    # ---- Report generation -------------------------------------------

    def test_generate_report_success_case(self):
        from tests.benchmark.cotenancy import (
            CoTenancyBenchmark,
            CoTenancyRunResult,
            CoTenancySuiteResult,
            OutcomeKind,
        )

        suite = CoTenancySuiteResult(
            runs=[
                CoTenancyRunResult(
                    adapter_name="fw",
                    model_size="large-v3",
                    sample_id="s001",
                    scenario_name="baseline",
                    tenant_vram_mb=0,
                    outcome=OutcomeKind.SUCCESS,
                    wall_time=2.0,
                    slowdown_factor=None,
                ),
                CoTenancyRunResult(
                    adapter_name="fw",
                    model_size="large-v3",
                    sample_id="s001",
                    scenario_name="pyannote",
                    tenant_vram_mb=1500,
                    outcome=OutcomeKind.SUCCESS,
                    wall_time=2.5,
                    baseline_wall_time=2.0,
                    slowdown_factor=1.25,
                ),
            ],
            total=2,
            success_count=2,
        )
        report = CoTenancyBenchmark.generate_report(suite)

        assert "# Co-Tenancy Benchmark Report" in report
        assert "## Per-Scenario Summary" in report
        assert "## Conclusion" in report
        assert "fw" in report
        assert "1.25x" in report or "1.25" in report

    def test_generate_report_with_oom(self):
        from tests.benchmark.cotenancy import (
            CoTenancyBenchmark,
            CoTenancyRunResult,
            CoTenancySuiteResult,
            OutcomeKind,
        )

        suite = CoTenancySuiteResult(
            runs=[
                CoTenancyRunResult(
                    adapter_name="fw",
                    model_size="large-v3",
                    sample_id="s001",
                    scenario_name="llm-idle",
                    tenant_vram_mb=4096,
                    outcome=OutcomeKind.OOM,
                    error_message="CUDA out of memory",
                ),
            ],
            total=1,
            oom_count=1,
        )
        report = CoTenancyBenchmark.generate_report(suite)

        assert "OOM" in report
        assert "## OOM Incidents" in report

    def test_generate_report_conclusion_no_oom(self):
        from tests.benchmark.cotenancy import (
            CoTenancyBenchmark,
            CoTenancyRunResult,
            CoTenancySuiteResult,
            OutcomeKind,
        )

        suite = CoTenancySuiteResult(
            runs=[
                CoTenancyRunResult(
                    adapter_name="fw",
                    model_size="medium",
                    sample_id="s001",
                    scenario_name="baseline",
                    tenant_vram_mb=0,
                    outcome=OutcomeKind.SUCCESS,
                    wall_time=1.0,
                )
            ],
            total=1,
            success_count=1,
        )
        report = CoTenancyBenchmark.generate_report(suite)
        assert "No OOM errors" in report

    def test_save_report_creates_file(self, tmp_path):
        from tests.benchmark.cotenancy import (
            CoTenancyBenchmark,
            CoTenancySuiteResult,
        )

        suite = CoTenancySuiteResult()
        bench = CoTenancyBenchmark([], [], [])
        out = tmp_path / "report.md"
        bench.save_report(suite, out)
        assert out.exists()

    def test_save_json_creates_file(self, tmp_path):
        from tests.benchmark.cotenancy import (
            CoTenancyBenchmark,
            CoTenancyRunResult,
            CoTenancySuiteResult,
            OutcomeKind,
        )

        suite = CoTenancySuiteResult(
            runs=[
                CoTenancyRunResult(
                    adapter_name="fw",
                    model_size="m",
                    sample_id="s1",
                    scenario_name="baseline",
                    tenant_vram_mb=0,
                    outcome=OutcomeKind.SUCCESS,
                    wall_time=1.0,
                )
            ],
            total=1,
            success_count=1,
        )
        bench = CoTenancyBenchmark([], [], [])
        out = tmp_path / "cotenancy.json"
        bench.save_json(suite, out)

        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["total"] == 1
        assert len(data["runs"]) == 1
        assert data["runs"][0]["outcome"] == OutcomeKind.SUCCESS
