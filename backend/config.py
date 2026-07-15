"""
config.py — Central configuration for NeuraScan
=================================================
All constants and settings in one place.
Change settings here — they apply everywhere.
"""

import os
from pathlib import Path

# ── Base paths ─────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODELS_DIR  = BASE_DIR / "models"
HEATMAP_DIR = BASE_DIR / "heatmaps"
SCANS_DIR   = BASE_DIR / "scans"
REPORTS_DIR = BASE_DIR / "reports"

# Auto-create directories
for d in [MODELS_DIR, HEATMAP_DIR, SCANS_DIR, REPORTS_DIR]:
    d.mkdir(exist_ok=True)

# ── Model settings ─────────────────────────────────────────────────────────────
MODEL_WEIGHTS_PATH = MODELS_DIR / "model_weights.npy"
IMG_SIZE           = (224, 224)
GRADCAM_LAYER      = "conv5_block3_out"

# 3-class mapping — must match train_generator.class_indices from notebook
# {'Hemorrhagic': 0, 'Ischemic': 1, 'Normal': 2}
CLASS_NAMES = {
    0: "Hemorrhagic",
    1: "Ischemic",
    2: "Normal",
}
NUM_CLASSES = 3

# ── Upload settings ────────────────────────────────────────────────────────────
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
MAX_FILE_BYTES      = 10 * 1024 * 1024   # 10 MB

# ── Auth settings ──────────────────────────────────────────────────────────────
SECRET_KEY         = os.getenv("SECRET_KEY", "neurascan-change-this-in-production")
ALGORITHM          = "HS256"
TOKEN_EXPIRE_HOURS = 24

# ── Database ───────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/neurascan.db")
# For PostgreSQL: DATABASE_URL = "postgresql://user:pass@host:5432/neurascan"

# ── App metadata ───────────────────────────────────────────────────────────────
APP_TITLE       = "NeuraScan API"
APP_DESCRIPTION = "AI Brain Stroke Detection — 3-Class CNN + Grad-CAM + PDF Reports"
APP_VERSION     = "6.0.0"