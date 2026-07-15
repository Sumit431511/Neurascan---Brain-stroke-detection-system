"""
db/crud.py — All database read/write operations
================================================
Routes call these functions. Never touch DB directly from routes.
"""

from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc
from db.models import Patient, Prediction


# ── Patients ───────────────────────────────────────────────────────────────────

def create_patient(db, name, age=None, gender=None, contact=None):
    p = Patient(name=name, age=age, gender=gender, contact=contact)
    db.add(p); db.commit(); db.refresh(p)
    return p

def get_patient(db, patient_id):
    return db.query(Patient).filter(Patient.id == patient_id).first()

def get_patient_by_name(db, name):
    return db.query(Patient).filter(Patient.name.ilike(f"%{name}%")).all()

def get_all_patients(db, skip=0, limit=100):
    return db.query(Patient).order_by(desc(Patient.created_at)).offset(skip).limit(limit).all()

def delete_patient(db, patient_id):
    p = get_patient(db, patient_id)
    if not p: return False
    db.delete(p); db.commit()
    return True


# ── Predictions ────────────────────────────────────────────────────────────────

def create_prediction(db, patient_id, prediction, stroke_type, is_stroke,
                      confidence, hemorrhagic_prob, ischemic_prob, normal_prob,
                      risk_level, filename=None, scan_path=None,
                      heatmap_path=None, doctor_notes=None):
    r = Prediction(
        patient_id=patient_id,
        prediction=prediction,
        stroke_type=stroke_type,
        is_stroke=is_stroke,
        confidence=confidence,
        hemorrhagic_prob=hemorrhagic_prob,
        ischemic_prob=ischemic_prob,
        normal_prob=normal_prob,
        risk_level=risk_level,
        filename=filename,
        scan_path=scan_path,
        heatmap_path=heatmap_path,
        doctor_notes=doctor_notes,
    )
    db.add(r); db.commit(); db.refresh(r)
    return r

def get_prediction(db, prediction_id):
    return db.query(Prediction).filter(Prediction.id == prediction_id).first()

def get_predictions_for_patient(db, patient_id):
    return db.query(Prediction).filter(
        Prediction.patient_id == patient_id
    ).order_by(desc(Prediction.created_at)).all()

def get_all_predictions(db, skip=0, limit=200):
    return db.query(Prediction).order_by(desc(Prediction.created_at)).offset(skip).limit(limit).all()

def get_recent_predictions(db, limit=10):
    return db.query(Prediction).order_by(desc(Prediction.created_at)).limit(limit).all()

def update_notes(db, prediction_id, doctor_notes):
    r = get_prediction(db, prediction_id)
    if r:
        r.doctor_notes = doctor_notes
        db.commit(); db.refresh(r)
    return r

def update_reviewed(db, prediction_id, reviewed: bool):
    r = get_prediction(db, prediction_id)
    if r:
        r.reviewed = reviewed
        db.commit(); db.refresh(r)
    return r

def update_pdf_path(db, prediction_id, pdf_path):
    r = get_prediction(db, prediction_id)
    if r:
        r.pdf_report_path = pdf_path
        db.commit(); db.refresh(r)
    return r

def delete_prediction(db, prediction_id):
    r = get_prediction(db, prediction_id)
    if not r: return False
    db.delete(r); db.commit()
    return True


# ── Dashboard ──────────────────────────────────────────────────────────────────

def get_dashboard_stats(db):
    from sqlalchemy import func
    total_patients    = db.query(Patient).count()
    total_predictions = db.query(Prediction).count()

    hemorrhagic_cases = db.query(Prediction).filter(Prediction.prediction == "Hemorrhagic").count()
    ischemic_cases    = db.query(Prediction).filter(Prediction.prediction == "Ischemic").count()
    normal_cases      = db.query(Prediction).filter(Prediction.prediction == "Normal").count()
    stroke_cases      = hemorrhagic_cases + ischemic_cases

    high_risk   = db.query(Prediction).filter(Prediction.risk_level == "High").count()
    medium_risk = db.query(Prediction).filter(Prediction.risk_level == "Medium").count()
    low_risk    = db.query(Prediction).filter(Prediction.risk_level == "Low").count()

    avg_conf = 0.0
    if total_predictions > 0:
        result   = db.query(func.avg(Prediction.confidence)).scalar()
        avg_conf = round(float(result or 0), 2)

    return {
        "total_patients":    total_patients,
        "total_predictions": total_predictions,
        "stroke_cases":      stroke_cases,
        "normal_cases":      normal_cases,
        "hemorrhagic_cases": hemorrhagic_cases,
        "ischemic_cases":    ischemic_cases,
        "stroke_rate":       round(stroke_cases / total_predictions * 100, 1) if total_predictions > 0 else 0,
        "avg_confidence":    avg_conf,
        "risk_breakdown":    {"high": high_risk, "medium": medium_risk, "low": low_risk},
        "stroke_breakdown":  {"hemorrhagic": hemorrhagic_cases, "ischemic": ischemic_cases, "normal": normal_cases},
    }

def get_monthly_counts(db):
    from sqlalchemy import func, case
    import sqlalchemy as sa
    results = (
        db.query(
            func.strftime("%Y-%m", Prediction.created_at).label("month"),
            func.count(Prediction.id).label("total"),
            func.sum(case((Prediction.prediction == "Hemorrhagic", 1), else_=0)).label("hemorrhagic"),
            func.sum(case((Prediction.prediction == "Ischemic",    1), else_=0)).label("ischemic"),
            func.sum(case((Prediction.prediction == "Normal",      1), else_=0)).label("normal"),
        )
        .group_by("month")
        .order_by("month")
        .limit(6)
        .all()
    )
    return [
        {
            "month":        r.month,
            "total":        r.total,
            "hemorrhagic":  r.hemorrhagic or 0,
            "ischemic":     r.ischemic or 0,
            "normal":       r.normal or 0,
            "strokes":      (r.hemorrhagic or 0) + (r.ischemic or 0),
        }
        for r in results
    ]