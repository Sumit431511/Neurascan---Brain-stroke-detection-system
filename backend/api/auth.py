"""
api/auth.py — Authentication routes
/auth/register, /auth/login, /auth/me
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from db.database import get_db
from db.schemas import RegisterRequest, LoginRequest, TokenResponse, DoctorOut
from db.models import Doctor
from core.auth import create_doctor, authenticate_doctor, create_token, get_current_doctor

router = APIRouter(tags=["Auth"])


@router.post("/register", response_model=TokenResponse)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new doctor account. Returns JWT token immediately."""
    doctor = create_doctor(db, name=body.name, email=body.email, password=body.password)
    return TokenResponse(
        access_token=create_token(doctor.email),
        doctor_name=doctor.name,
        doctor_email=doctor.email,
    )


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    """Login with email and password. Returns JWT token valid for 24 hours."""
    doctor = authenticate_doctor(db, email=body.email, password=body.password)
    return TokenResponse(
        access_token=create_token(doctor.email),
        doctor_name=doctor.name,
        doctor_email=doctor.email,
    )


@router.get("/me", response_model=DoctorOut)
def get_me(doctor: Doctor = Depends(get_current_doctor)):
    """Get current logged-in doctor's profile."""
    return doctor