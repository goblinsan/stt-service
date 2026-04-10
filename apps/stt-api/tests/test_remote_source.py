from pathlib import Path

import pytest

from src.remote_source import RemoteSourceError, download_remote_audio


class FakeResponse:
    def __init__(self, body: bytes, url: str = "https://example.com/audio.wav"):
        self._body = body
        self.url = url
        self.headers = {"content-length": str(len(body))}

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=1024 * 1024):
        yield self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_download_remote_audio_writes_file(monkeypatch, tmp_path):
    payload = b"audio-bytes"

    def fake_get(*args, **kwargs):
        return FakeResponse(payload)

    monkeypatch.setattr("src.remote_source.requests.get", fake_get)
    path, filename, size = download_remote_audio(
        "https://example.com/audio.wav",
        "clip.wav",
        max_bytes=1024,
        timeout_sec=10,
        allowed_hosts_raw="example.com",
    )

    try:
        assert filename == "clip.wav"
        assert size == len(payload)
        assert Path(path).read_bytes() == payload
    finally:
        Path(path).unlink(missing_ok=True)


def test_download_remote_audio_rejects_disallowed_host():
    with pytest.raises(RemoteSourceError, match="not allowed"):
        download_remote_audio(
            "https://forbidden.example/audio.wav",
            "clip.wav",
            max_bytes=1024,
            timeout_sec=10,
            allowed_hosts_raw="example.com",
        )


def test_download_remote_audio_rejects_oversized_content_length(monkeypatch):
    class LargeResponse(FakeResponse):
        def __init__(self):
            super().__init__(b"small")
            self.headers = {"content-length": str(2 * 1024 * 1024)}

    def fake_get(*args, **kwargs):
        return LargeResponse()

    monkeypatch.setattr("src.remote_source.requests.get", fake_get)
    with pytest.raises(RemoteSourceError, match="exceeds the configured upload limit"):
        download_remote_audio(
            "https://example.com/audio.wav",
            "clip.wav",
            max_bytes=1024,
            timeout_sec=10,
            allowed_hosts_raw="example.com",
        )
