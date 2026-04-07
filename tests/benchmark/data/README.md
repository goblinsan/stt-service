# Reference Audio Dataset

This directory contains the reference audio dataset used by the STT benchmark
harness.  Audio files are **not** committed to the repository (they are
excluded by `.gitignore`).  Use the download script below to fetch them.

## Quick start

```bash
# Install download dependencies
pip install requests soundfile

# Download all samples
python tests/benchmark/data/download_data.py

# Download only English clean samples
python tests/benchmark/data/download_data.py --category clean_single_speaker

# Dry-run: list what would be downloaded
python tests/benchmark/data/download_data.py --dry-run
```

## Dataset composition

The dataset covers the key audio conditions that stress-test STT engines:

| Category | Count | Description |
|---|---|---|
| `clean_single_speaker` | 3 | Baseline clear English narration (LibriSpeech test-clean) |
| `noisy` | 2 | Challenging/noisier English audio (LibriSpeech test-other) |
| `accented_english` | 2 | Non-native English speakers (Mozilla Common Voice) |
| `non_english` | 3 | Spanish, French, and Japanese (Mozilla Common Voice) |
| `multi_speaker` | 1 | Two-speaker conversation constructed from LibriSpeech |
| `long_form` | 2 | 2-minute and 10-minute English narration (LibriSpeech test-clean) |

**Total: 13 samples** covering durations from ~8 s to ~10 min.

## File naming convention

Each sample consists of two files:

```
<id>.wav   ← 16 kHz mono WAV (or FLAC if soundfile is not installed)
<id>.txt   ← Plain-text reference transcript (UTF-8, no punctuation normalisation)
```

Example:

```
ls_clean_en_001.wav
ls_clean_en_001.txt   ← "HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS..."
```

## Sources and licences

| Source | Licence |
|---|---|
| [LibriSpeech](https://www.openslr.org/12) (Panayotov et al., 2015) | CC BY 4.0 |
| [Mozilla Common Voice](https://commonvoice.mozilla.org) | CC-0 / CC BY 4.0 depending on corpus version |

### Common Voice: manual download required

Mozilla Common Voice samples (**`cv_*.wav`**) require a free account and
manual download from <https://commonvoice.mozilla.org/datasets>.  Once
downloaded, place the `.wav` and `.txt` files in this directory with the
filenames listed in `manifest.json`.

Run `download_data.py` for the exact list of files needed and instructions.

## Manifest

`manifest.json` is the authoritative dataset specification.  Each entry
contains:

```jsonc
{
  "id":          "ls_clean_en_001",      // unique sample identifier
  "filename":    "ls_clean_en_001.wav",  // audio file (not in repo)
  "reference":   "ls_clean_en_001.txt", // ground-truth transcript
  "category":    "clean_single_speaker",
  "language":    "en",
  "duration_s":  10.0,
  "source":      "LibriSpeech test-clean",
  "url":         "https://www.openslr.org/resources/12/test-clean.tar.gz",
  "description": "Clean, single-speaker English — baseline"
}
```
