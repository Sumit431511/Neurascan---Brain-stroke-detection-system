"""
db/models.py — SQLAlchemy ORM models (database tables)
=======================================================
Each class = one database table.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from db.database import Base


class Doctor(Base):
    """Registered doctor accounts."""
    __tablename__ = "doctors"

    id              = Column(Integer, primary_key=True, index=True)
    name            = Column(String(100), nullable=False)
    email           = Column(String(150), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Doctor id={self.id} email={self.email}>"


class Patient(Base):
    """One record per patient."""
    __tablename__ = "patients"

    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String(100), nullable=False)
    age        = Column(Integer, nullable=True)
    gender     = Column(String(10), nullable=True)
    contact    = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship(
        "Prediction", back_populates="patient", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Patient id={self.id} name={self.name}>"


class Prediction(Base):
    """One record per scan prediction — 3-class model."""
    __tablename__ = "predictions"

    id              = Column(Integer, primary_key=True, index=True)
    patient_id      = Column(Integer, ForeignKey("patients.id"), nullable=False)

    # Files
    filename        = Column(String(255), nullable=True)
    scan_path       = Column(String(500), nullable=True)
    heatmap_path    = Column(String(500), nullable=True)
    pdf_report_path = Column(String(500), nullable=True)

    # Model output — 3-class
    prediction       = Column(String(50),  nullable=False)   # "Hemorrhagic" / "Ischemic" / "Normal"
    stroke_type      = Column(String(50),  nullable=True)    # "Hemorrhagic" / "Ischemic" / None
    is_stroke        = Column(Boolean,     nullable=False)   # True for Hemorrhagic or Ischemic
    confidence       = Column(Float,       nullable=False)   # confidence of predicted class
    hemorrhagic_prob = Column(Float,       nullable=False)   # probability of Hemorrhagic
    ischemic_prob    = Column(Float,       nullable=False)   # probability of Ischemic
    normal_prob      = Column(Float,       nullable=False)   # probability of Normal
    risk_level       = Column(String(20),  nullable=False)   # High / Medium / Low

    # Doctor workflow
    doctor_notes    = Column(Text,    nullable=True)
    reviewed        = Column(Boolean, default=False)

    created_at      = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="predictions")

    def __repr__(self):
        return f"<Prediction id={self.id} result={self.prediction}>"