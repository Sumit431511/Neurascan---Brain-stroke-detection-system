"""
db/schemas.py — Pydantic request/response models
=================================================
Request bodies and response shapes for all API endpoints.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


# ── Auth ───────────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    name:     str = Field(..., min_length=2, max_length=100, example="Dr. Sharma")
    email:    str = Field(..., example="sharma@hospital.com")
    password: str = Field(..., min_length=6, example="securepass123")

class LoginRequest(BaseModel):
    email:    str = Field(..., example="sharma@hospital.com")
    password: str = Field(..., example="securepass123")

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    doctor_name:  str
    doctor_email: str
    expires_in:   str = "24 hours"

class DoctorOut(BaseModel):
    id:    int
    name:  str
    email: str
    class Config:
        from_attributes = True


# ── Patients ───────────────────────────────────────────────────────────────────
class PatientCreate(BaseModel):
    name:    str           = Field(..., min_length=1, max_length=100, example="Rahul Sharma")
    age:     Optional[int] = Field(None, ge=0, le=120, example=45)
    gender:  Optional[str] = Field(None, example="Male")
    contact: Optional[str] = Field(None, example="rahul@email.com")

class PatientOut(BaseModel):
    id:         int
    name:       str
    age:        Optional[int]
    gender:     Optional[str]
    contact:    Optional[str]
    created_at: datetime
    class Config:
        from_attributes = True


# ── Predictions ────────────────────────────────────────────────────────────────
class PredictionOut(BaseModel):
    id:               int
    patient_id:       int
    filename:         Optional[str]
    prediction:       str            # "Hemorrhagic" / "Ischemic" / "Normal"
    stroke_type:      Optional[str]  # "Hemorrhagic" / "Ischemic" / None
    is_stroke:        bool
    confidence:       float
    hemorrhagic_prob: float
    ischemic_prob:    float
    normal_prob:      float
    risk_level:       str
    doctor_notes:     Optional[str]
    reviewed:         bool
    heatmap_path:     Optional[str]
    scan_path:        Optional[str]
    pdf_report_path:  Optional[str]
    created_at:       datetime
    class Config:
        from_attributes = True

class UpdateNotesRequest(BaseModel):
    doctor_notes: str = Field(..., example="Consistent with ischemic stroke.")

class UpdateReviewedRequest(BaseModel):
    reviewed: bool


# ── Dashboard ──────────────────────────────────────────────────────────────────
class RiskBreakdown(BaseModel):
    high:   int
    medium: int
    low:    int

class StrokeBreakdown(BaseModel):
    hemorrhagic: int
    ischemic:    int
    normal:      int

class DashboardStats(BaseModel):
    total_patients:    int
    total_predictions: int
    stroke_cases:      int       # hemorrhagic + ischemic combined
    normal_cases:      int
    hemorrhagic_cases: int
    ischemic_cases:    int
    stroke_rate:       float
    avg_confidence:    float
    risk_breakdown:    RiskBreakdown
    stroke_breakdown:  StrokeBreakdown