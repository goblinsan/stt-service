import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests


class RemoteSourceError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def _parse_allowed_hosts(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(part.strip().lower() for part in raw.split(",") if part.strip())


def _host_allowed(hostname: str | None, allowed_hosts: tuple[str, ...]) -> bool:
    if not allowed_hosts:
        return True
    if not hostname:
        return False
    hostname = hostname.lower()
    return any(hostname == allowed or hostname.endswith(f".{allowed}") for allowed in allowed_hosts)


def _derive_filename(source_url: str, filename: str | None) -> str:
    if filename and filename.strip():
        return Path(filename.strip()).name
    parsed = urlparse(source_url)
    path_name = Path(parsed.path).name
    return path_name or "remote-audio"


def download_remote_audio(
    source_url: str,
    filename: str | None,
    max_bytes: int,
    timeout_sec: int,
    allowed_hosts_raw: str | None,
) -> tuple[str, str, int]:
    parsed = urlparse(source_url)
    if parsed.scheme not in {"http", "https"}:
        raise RemoteSourceError(422, "source_url must use http or https")

    allowed_hosts = _parse_allowed_hosts(allowed_hosts_raw)
    if not _host_allowed(parsed.hostname, allowed_hosts):
        raise RemoteSourceError(422, "source_url host is not allowed by STT_REMOTE_SOURCE_ALLOWED_HOSTS")

    resolved_filename = _derive_filename(source_url, filename)
    suffix = Path(resolved_filename).suffix or ".bin"
    tmp_path: str | None = None

    try:
        with requests.get(
            source_url,
            stream=True,
            timeout=(10, timeout_sec),
            allow_redirects=True,
        ) as response:
            final_host = urlparse(response.url).hostname
            if not _host_allowed(final_host, allowed_hosts):
                raise RemoteSourceError(422, "redirect target host is not allowed by STT_REMOTE_SOURCE_ALLOWED_HOSTS")
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise RemoteSourceError(502, f"failed to download remote audio: {exc}") from exc

            content_length = response.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > max_bytes:
                        raise RemoteSourceError(413, "Remote audio exceeds the configured upload limit")
                except ValueError:
                    pass

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = tmp.name
                total_bytes = 0
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        raise RemoteSourceError(413, "Remote audio exceeds the configured upload limit")
                    tmp.write(chunk)

            return tmp_path, resolved_filename, total_bytes
    except requests.RequestException as exc:
        raise RemoteSourceError(502, f"failed to download remote audio: {exc}") from exc
    except RemoteSourceError:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
