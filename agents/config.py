"""
agents/config.py

Single source of truth for all shared configurations across the multi-agent pipeline:
- Text column and label definitions in a fixed order
- Dataset and model paths
- Execution thresholds and constants

Every agent, the API, the Streamlit dashboard, and the database logging MUST import from here.
No hardcoded labels allowed.
"""

from pathlib import Path
from typing import Dict, List

# ── Base Paths ───────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "processed"

# Dataset Paths (Single Source of Truth)
TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "val.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

# Model Checkpoint Paths
BERT_MODEL_PATH = ROOT_DIR / "saved_model"
DEBERTA_MODEL_PATH = ROOT_DIR / "models" / "deberta_v3"
ROBERTA_MODEL_PATH = ROOT_DIR / "models" / "roberta"
ROBERTA_MODEL_NAME = "roberta-base"

# SQLite Database Path
DATABASE_PATH = ROOT_DIR / "rio_database.db"

# ── Data Column Schema ───────────────────────────────────────────────────────
TEXT_COLUMN = "cleaned_text"
RAW_TEXT_FALLBACK = "comment_text"
LABEL_ID_COLUMN = "label_id"
LABEL_NAME_COLUMN = "label_name"

# ── Canonical Shared Labels (Fixed Order - Source of Truth) ──────────────────
# 5 mutually exclusive categories aligning with the processed dataset & BERT
SHARED_LABELS: List[str] = [
    "Not Cyberbullying",
    "Insult",
    "Threat",
    "Hate Speech",
    "Harassment",
]

ID2LABEL: Dict[int, str] = {i: name for i, name in enumerate(SHARED_LABELS)}
LABEL2ID: Dict[str, int] = {name: i for i, name in ID2LABEL.items()}

# ── DeBERTa Raw Labels (6 Multi-Label Categories on Jigsaw) ──────────────────
DEBERTA_RAW_LABELS: List[str] = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

# ── Inference Parameters ─────────────────────────────────────────────────────
MAX_LENGTH = 128
DEFAULT_BATCH_SIZE = 32
CYBERBULLYING_THRESHOLD = 0.50

# Severity Mapping for Reporting
SEVERITY_THRESHOLDS = {
    "Threat": "HIGH",
    "Hate Speech": "HIGH",
    "Harassment": "MEDIUM",
    "Insult": "MEDIUM",
    "Not Cyberbullying": "LOW",
}

