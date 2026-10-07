"""Central configuration: paths, defaults, constants.

Everything that a user might reasonably want to change lives here or in
``app/env.py`` (runtime probing).  No large model blobs, no absolute paths to
other people's machines.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# PROJECT_ROOT = .../drum-practice-generator
PROJECT_ROOT = Path(__file__).resolve().parent.parent

APP_DIR = PROJECT_ROOT / "app"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
MODELS_DIR = PROJECT_ROOT / "models"
TEMP_DIR = PROJECT_ROOT / "temp"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"
TOOLS_DIR = PROJECT_ROOT / "tools"

# Optional user overrides (kept simple on purpose - env vars, not a config file)
if os.environ.get("DPG_OUTPUT_DIR"):
    OUTPUT_DIR = Path(os.environ["DPG_OUTPUT_DIR"]).expanduser().resolve()
if os.environ.get("DPG_TEMP_DIR"):
    TEMP_DIR = Path(os.environ["DPG_TEMP_DIR"]).expanduser().resolve()

TEMP_SEPARATION_DIR = TEMP_DIR / "separation"
TEMP_MIX_DIR = TEMP_DIR / "mix"
TEMP_UPLOAD_DIR = TEMP_DIR / "uploads"

# --------------------------------------------------------------------------
# Web server
# --------------------------------------------------------------------------
HOST = os.environ.get("DPG_HOST", "127.0.0.1")
PORT = int(os.environ.get("DPG_PORT", "8765"))
# Local-only tool: never bind to 0.0.0.0 unless the user explicitly asks.
OPEN_BROWSER = os.environ.get("DPG_NO_BROWSER", "") == ""

# --------------------------------------------------------------------------
# Audio defaults
# --------------------------------------------------------------------------
# Demucs htdemucs operates at 44.1 kHz stereo internally.
MODEL_SAMPLE_RATE = 44100
MODEL_CHANNELS = 2

# Bit depth of the delivered practice track.  16-bit is what every e-drum
# module and phone accepts; 24-bit is offered for further editing.
DEFAULT_OUTPUT_BITS = 16

# Practice defaults (see README)
DEFAULT_DRUM_VOLUME = 0.10      # 10% - "weak drums" practice
DEFAULT_METRONOME_VOLUME = 0.30  # 30%
DEFAULT_TIME_SIGNATURE = "4/4"
DEFAULT_SPEED = 1.0

# Peak normalisation target.  0.97 leaves ~0.27 dB of headroom for the
# inter-sample peaks that lossy encoders like MP3 can otherwise overshoot.
PEAK_TARGET = 0.97

# --------------------------------------------------------------------------
# Separation defaults
# --------------------------------------------------------------------------
# Demucs Hybrid Transformer models were trained on segments up to 7.8 s.
# 7 s is the largest safe value and keeps VRAM well inside 16 GB on a 5070 Ti.
DEFAULT_SEGMENT = 7
DEFAULT_OVERLAP = 0.25
DEFAULT_SHIFTS = 1      # >1 improves SDR slightly but multiplies runtime
DEFAULT_MODEL = "htdemucs"

# Models we expose in the UI.  All are official Demucs releases.
#   signature -> (display name, description, approx download size MB)
AVAILABLE_MODELS: dict[str, dict] = {
    "htdemucs": {
        "name": "Demucs v4 htdemucs",
        "desc": "默认。速度/质量平衡最佳，单模型。",
        "size_mb": 80,
        "multiplier": 1.0,
    },
    "htdemucs_ft": {
        "name": "Demucs v4 htdemucs_ft (fine-tuned)",
        "desc": "4 模型集成，质量略高但耗时约为 4 倍。",
        "size_mb": 320,
        "multiplier": 4.0,
    },
    "hdemucs_mmi": {
        "name": "Demucs v3 hdemucs_mmi",
        "desc": "旧版 HybrID Demucs，作为备用对照。",
        "size_mb": 320,
        "multiplier": 1.5,
    },
}

# --------------------------------------------------------------------------
# Supported input formats
# --------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma", ".opus")

# --------------------------------------------------------------------------
# Limits / guard rails
# --------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 500 * 1024 * 1024   # 500 MB single song
MAX_DURATION_SECONDS = 20 * 60         # 20 min - beyond this we warn
MIN_FREE_DISK_BYTES = 2 * 1024**3      # refuse to start if < 2 GB free

SPEED_MIN = 0.5
SPEED_MAX = 2.0

TIME_SIGNATURES = {
    "4/4": {"beats_per_bar": 4, "beat_unit": 4},
    "3/4": {"beats_per_bar": 3, "beat_unit": 4},
    "6/8": {"beats_per_bar": 6, "beat_unit": 8},
}


def ensure_directories() -> None:
    """Create every directory the app writes to.  Safe to call repeatedly."""
    for path in (
        MODELS_DIR,
        TEMP_DIR,
        TEMP_SEPARATION_DIR,
        TEMP_MIX_DIR,
        TEMP_UPLOAD_DIR,
        OUTPUT_DIR,
        LOGS_DIR,
        TOOLS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
