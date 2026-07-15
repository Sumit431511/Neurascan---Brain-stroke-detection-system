"""
api/dashboard.py — Doctor dashboard routes
/dashboard/stats, /dashboard/monthly
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from db.database import get_db
from db.schemas import DashboardStats
from db.models import Doctor
from core.auth import get_current_doctor
import db.crud as crud

router = APIRouter(tags=["Dashboard"])


@router.get("/stats", response_model=DashboardStats)
def dashboard_stats(
    db:     Session = Depends(get_db),
    doctor: Doctor  = Depends(get_current_doctor),
):
    """Aggregate statistics for dashboard stat cards."""
    return crud.get_dashboard_stats(db)


@router.get("/monthly")
def dashboard_monthly(
    db:     Session = Depends(get_db),
    doctor: Doctor  = Depends(get_current_doctor),
):
    """Monthly prediction counts for bar chart."""
    return crud.get_monthly_counts(db)