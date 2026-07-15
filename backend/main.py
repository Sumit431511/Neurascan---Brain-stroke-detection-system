"""
main.py — NeuraScan API Entry Point
====================================
This file only does 3 things:
  1. Creates the FastAPI app
  2. Registers routers
  3. Handles startup/shutdown (model loading)

Business logic lives in:
  api/      → routes
  core/     → ML, heatmap, PDF, auth logic
  db/       → database models, schemas, CRUD
  config.py → all settings
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import APP_TITLE, APP_DESCRIPTION, APP_VERSION, HEATMAP_DIR, SCANS_DIR, REPORTS_DIR
from db.database import create_tables
from db.models import Doctor  # ensures Doctor table is created
from core.model import load_model, release_model
from api import auth, patients, predictions, dashboard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create DB tables + load ML model. Shutdown: release model."""
    create_tables()
    logger.info("✅ Database tables ready.")
    load_model()
    yield
    release_model()


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static file serving ────────────────────────────────────────────────────────
app.mount("/heatmaps", StaticFiles(directory=str(HEATMAP_DIR)), name="heatmaps")
app.mount("/scans",    StaticFiles(directory=str(SCANS_DIR)),   name="scans")
app.mount("/reports",  StaticFiles(directory=str(REPORTS_DIR)), name="reports")

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(auth.router,        prefix="/auth")
app.include_router(patients.router,    prefix="/patients")
app.include_router(predictions.router, prefix="/predictions")
app.include_router(dashboard.router,   prefix="/dashboard")


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    from core.model import _model
    return {
        "status":       "ok",
        "version":      APP_VERSION,
        "model_loaded": _model is not None,
    }

@app.get("/health", tags=["Health"])
def health():
    from core.model import _model
    return {"status": "healthy", "model_ready": _model is not None}