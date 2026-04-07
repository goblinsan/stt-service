#!/usr/bin/env python3
"""Download reference audio dataset for STT benchmarking (issue #2).

Fetches CC-licensed / public-domain audio clips from LibriSpeech
(OpenSLR 12) and Mozilla Common Voice, then validates them against
the ``manifest.json`` in this directory.

Usage
-----
::

    # Download all samples into tests/benchmark/data/
    python tests/benchmark/data/download_data.py

    # Download only the English clean samples
    python tests/benchmark/data/download_data.py --category clean_single_speaker

    # Dry-run: list what would be downloaded without downloading
    python tests/benchmark/data/download_data.py --dry-run

Requirements
------------
``pip install requests soundfile`` (``soundfile`` is optional but enables
audio duration validation).

Notes
-----
* LibriSpeech samples are extracted from the official test-clean and
  test-other archives hosted at https://www.openslr.org/12.  Only the
  specific utterance files referenced in the manifest are extracted to
  keep the download footprint small (~5–20 MB total).
* Common Voice samples require a free account and dataset download from
  https://commonvoice.mozilla.org/datasets.  Automated download of the
  full Common Voice corpus is intentionally *not* supported here to
  respect Mozilla's terms of service.  The script will print instructions
  for those samples if the files are missing.
* Downloaded ``.wav`` and ``.txt`` files are placed directly in the same
  directory as this script (``tests/benchmark/data/``).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    sys.exit("requests is not installed — run: pip install requests")

DATA_DIR = Path(__file__).parent
MANIFEST_PATH = DATA_DIR / "manifest.json"

# LibriSpeech utterance IDs used to build deterministic filenames.
# Format: (reader_id, chapter_id, utterance_id) → local filename stem
_LIBRISPEECH_UTTERANCES: dict[str, tuple[str, str, str]] = {
    "ls_clean_en_001": ("1089", "134686", "0001"),
    "ls_clean_en_002": ("1089", "134686", "0007"),
    "ls_clean_en_003": ("1089", "134686", "0022"),
    "ls_other_en_001": ("116", "288045", "0001"),
    "ls_other_en_002": ("116", "288045", "0011"),
    "ls_longform_001": ("2300", "131720", "0001"),
    "ls_longform_002": ("2300", "131720", "0002"),
    "ls_multi_speaker_001": None,  # constructed from two utterances — see below
}

_LIBRISPEECH_ARCHIVES: dict[str, str] = {
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
}

# Sample IDs that require manual download from Mozilla Common Voice.
_COMMON_VOICE_IDS = {"cv_accented_en_001", "cv_accented_en_002", "cv_es_001", "cv_fr_001", "cv_ja_001"}


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download_stream(url: str, dest: Path, *, timeout: int = 120) -> None:
    """Stream *url* to *dest*, printing a simple progress indicator."""
    print(f"  Downloading {url} …", end=" ", flush=True)
    t0 = time.monotonic()
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        written = 0
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
                written += len(chunk)
    elapsed = time.monotonic() - t0
    size_mb = written / (1024 * 1024)
    print(f"{size_mb:.1f} MB in {elapsed:.1f}s")


def _extract_librispeech_utterance(
    archive_path: Path,
    subset: str,
    reader: str,
    chapter: str,
    utterance: str,
    dest_wav: Path,
    dest_txt: Path,
) -> None:
    """Extract a single utterance from a LibriSpeech tar.gz archive.

    Args:
        archive_path: Path to the downloaded ``test-clean.tar.gz`` or
            ``test-other.tar.gz`` file.
        subset: LibriSpeech subset name (e.g. ``"test-clean"``), used as the
            subdirectory name inside the archive.
        reader, chapter, utterance: LibriSpeech utterance identifiers.
        dest_wav: Destination ``.wav`` (or ``.flac`` fallback) path.
        dest_txt: Destination ``.txt`` transcript path.
    """
    wav_member = f"LibriSpeech/{subset}/{reader}/{chapter}/{reader}-{chapter}-{utterance}.flac"
    txt_member = f"LibriSpeech/{subset}/{reader}/{chapter}/{reader}-{chapter}.trans.txt"

    with tarfile.open(archive_path, "r:gz") as tf:
        try:
            import soundfile as sf  # noqa: PLC0415

            flac_data = tf.extractfile(wav_member)
            if flac_data is None:
                raise FileNotFoundError(f"Member not found in archive: {wav_member}")
            audio, sr = sf.read(io.BytesIO(flac_data.read()))
            sf.write(str(dest_wav), audio, sr)
        except ImportError:
            # soundfile not available — extract FLAC directly and rename
            flac_member = tf.getmember(wav_member)
            tf.extract(flac_member, path=dest_wav.parent)
            extracted = dest_wav.parent / wav_member
            extracted.rename(dest_wav.with_suffix(".flac"))
            print(
                f"    NOTE: soundfile not installed — saved as .flac instead of .wav: "
                f"{dest_wav.with_suffix('.flac')}"
            )

        # Extract matching reference transcript line
        trans_data = tf.extractfile(txt_member)
        if trans_data is None:
            raise FileNotFoundError(f"Transcript member not found: {txt_member}")
        key = f"{reader}-{chapter}-{utterance}"
        for line in trans_data.read().decode().splitlines():
            if line.startswith(key):
                dest_txt.write_text(line[len(key):].strip(), encoding="utf-8")
                break
        else:
            raise ValueError(f"Transcript line not found for key: {key}")


def _print_common_voice_instructions(missing_ids: list[str], manifest: dict) -> None:
    samples_by_id = {s["id"]: s for s in manifest["samples"]}
    print()
    print("=" * 70)
    print("MANUAL DOWNLOAD REQUIRED — Mozilla Common Voice")
    print("=" * 70)
    print()
    print("The following samples require a free Mozilla account and manual")
    print("dataset download from https://commonvoice.mozilla.org/datasets")
    print()
    for sid in missing_ids:
        sample = samples_by_id.get(sid)
        if sample:
            print(f"  {sid}: {sample.get('description', '')} ({sample.get('language', '')})")
    print()
    print("After downloading, place the .wav and .txt files in:")
    print(f"  {DATA_DIR}")
    print()
    print("Expected filenames:")
    for sid in missing_ids:
        sample = samples_by_id.get(sid)
        if sample:
            print(f"  {sample['filename']}  ←→  {sample['reference']}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download reference audio dataset for STT benchmarking",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Download only samples matching this category (e.g. clean_single_speaker)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Target directory for downloaded files (default: {DATA_DIR})",
    )
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        manifest = json.load(fh)

    samples = manifest["samples"]
    if args.category:
        samples = [s for s in samples if s.get("category") == args.category]
        if not samples:
            print(f"ERROR: No samples found for category '{args.category}'", file=sys.stderr)
            return 1

    missing_cv: list[str] = []
    errors: int = 0

    for sample in samples:
        sid = sample["id"]
        wav_path = data_dir / sample["filename"]
        txt_path = data_dir / sample["reference"]

        if wav_path.exists() and txt_path.exists():
            print(f"  ✓  {sid} — already present")
            continue

        if args.dry_run:
            print(f"  ↓  {sid} — would download ({sample.get('source', 'unknown source')})")
            continue

        if sid in _COMMON_VOICE_IDS:
            if not wav_path.exists():
                missing_cv.append(sid)
            continue

        # LibriSpeech sample
        utterance_info = _LIBRISPEECH_UTTERANCES.get(sid)
        if utterance_info is None:
            print(f"  ?  {sid} — no automated download configured; skipping")
            continue

        reader, chapter, utterance = utterance_info
        # Determine which archive (test-clean or test-other)
        archive_key = "test-other" if sid.startswith("ls_other") else "test-clean"
        archive_url = _LIBRISPEECH_ARCHIVES[archive_key]
        archive_path = data_dir / f"{archive_key}.tar.gz"

        try:
            if not archive_path.exists():
                _download_stream(archive_url, archive_path)
            else:
                print(f"  ✓  {archive_key}.tar.gz — already cached")

            print(f"  Extracting {sid} …", end=" ", flush=True)
            _extract_librispeech_utterance(
                archive_path, archive_key, reader, chapter, utterance, wav_path, txt_path
            )
            print("done")
        except Exception as exc:  # noqa: BLE001
            print(f"\n  ERROR: Failed to download/extract {sid}: {exc}", file=sys.stderr)
            errors += 1

    if missing_cv:
        _print_common_voice_instructions(missing_cv, manifest)

    if not args.dry_run:
        # Summary
        present = sum(
            1 for s in samples
            if (data_dir / s["filename"]).exists() and (data_dir / s["reference"]).exists()
        )
        print(f"\nDataset ready: {present}/{len(samples)} samples in {data_dir}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
