"""Single source of truth for on-disk locations.

Everything is anchored to the repository root (this file's grandparent), so
paths resolve the same way regardless of the current working directory.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ─── data/raw: dataset as released (audio, annotations, metadata) ─────────
DATA_RAW = ROOT / "data" / "raw"
AUDIO_DIR = DATA_RAW / "KVocSet"
RAW_JSON_DIR = DATA_RAW / "raw_json"
LABEL_DIR = DATA_RAW / "sigimsae_labels"
PRECHECK_LABEL_DIR = DATA_RAW / "precheck_sigimsae_labels"
TICK_DIR = DATA_RAW / "tick_positions"
TOTAL_METADATA = DATA_RAW / "total_metadata.csv"
METADATA = ROOT / "metadata.csv"  # public file, kept at the repo root

# ─── data/derived: regenerable features and caches ────────────────────────
DATA_DERIVED = ROOT / "data" / "derived"
FEATURES_DIR = DATA_DERIVED / "features"
CHROMA_DIR = FEATURES_DIR / "chroma_output"
MEL_DIR = FEATURES_DIR / "mel_output"
PITCH_DIR = FEATURES_DIR / "pitch_output"
RMVPE_DIR = PITCH_DIR / "rmvpe"
BEAT_DIR = FEATURES_DIR / "beat_output"
WAVEFORM_CACHE_DIR = DATA_DERIVED / "waveform_cache"
MERT_CACHE_DIR = DATA_DERIVED / "mert_embedding_cache"
CACHE_DIR = DATA_DERIVED / "cache"

# ─── experiments: outputs of training runs ────────────────────────────────
EXPERIMENTS = ROOT / "experiments"
MODELS_DIR = EXPERIMENTS / "models"
LOGS_DIR = EXPERIMENTS / "logs"
RESULTS_DIR = EXPERIMENTS / "results"
OPTUNA_DB = EXPERIMENTS / "optuna_sweeps.db"
