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
