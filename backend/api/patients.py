"""
api/patients.py — Patient management routes
/patients CRUD operations
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from db.schemas import PatientCreate, PatientOut, PredictionOut
from db.models import Doctor
from core.auth import get_current_doctor
import db.crud as crud

router = APIRouter(tags=["Patients"])


@router.post("", response_model=PatientOut)
def create_patient(
    body:   PatientCreate,
    db:     Session = Depends(get_db),
    doctor: Doctor  = Depends(get_current_doctor),
):
    """Register a new patient."""
    return crud.create_patient(db, name=body.name, age=body.age,
                               gender=body.gender, contact=body.contact)


@router.get("", response_model=List[PatientOut])
def list_patients(
    skip:   int = 0,
    limit:  int = 100,
    db:     Session = Depends(get_db),
    doctor: Doctor  = Depends(get_current_doctor),
):
    """Get all patients, newest first."""
    return crud.get_all_patients(db, skip=skip, limit=limit)


@router.get("/search", response_model=List[PatientOut])
def search_patients(
    name:   str,
    db:     Session = Depends(get_db),
    doctor: Doctor  = Depends(get_current_doctor),
):
    """Search patients by name."""
    return crud.get_patient_by_name(db, name)


@router.get("/{patient_id}", response_model=PatientOut)
def get_patient(
    patient_id: int,
    db:     Session = Depends(get_db),
    doctor: Doctor  = Depends(get_current_doctor),
):
    """Get a single patient by ID."""
    p = crud.get_patient(db, patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found.")
    return p


@router.get("/{patient_id}/predictions", response_model=List[PredictionOut])
def get_patient_predictions(
    patient_id: int,
    db:     Session = Depends(get_db),
    doctor: Doctor  = Depends(get_current_doctor),
):
    """Get all scan predictions for a patient."""
    if not crud.get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found.")
    return crud.get_predictions_for_patient(db, patient_id)


@router.delete("/{patient_id}")
def delete_patient(
    patient_id: int,
    db:     Session = Depends(get_db),
    doctor: Doctor  = Depends(get_current_doctor),
):
    """Delete a patient and all their records."""
    if not crud.delete_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found.")
    return {"message": f"Patient {patient_id} deleted."}