"""
core/auth.py — JWT authentication utilities
============================================
Password hashing, token creation/verification,
and the FastAPI dependency for protecting routes.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from config import SECRET_KEY, ALGORITHM, TOKEN_EXPIRE_HOURS
from db.database import get_db
from db.models import Doctor

logger     = logging.getLogger(__name__)
pwd_ctx    = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)
bearer     = HTTPBearer()


# ── Password ───────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    truncated = plain.encode("utf-8")[:72].decode("utf-8", errors="ignore")
    return pwd_ctx.hash(truncated)

def verify_password(plain: str, hashed: str) -> bool:
    truncated = plain.encode("utf-8")[:72].decode("utf-8", errors="ignore")
    return pwd_ctx.verify(truncated, hashed)


# ── JWT ────────────────────────────────────────────────────────────────────────

def create_token(email: str) -> str:
    expire  = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    payload = {"sub": email, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── DB operations ──────────────────────────────────────────────────────────────

def get_doctor_by_email(db: Session, email: str) -> Optional[Doctor]:
    return db.query(Doctor).filter(Doctor.email == email.lower()).first()

def create_doctor(db: Session, name: str, email: str, password: str) -> Doctor:
    if get_doctor_by_email(db, email):
        raise HTTPException(status_code=409, detail=f"Email '{email}' is already registered.")
    doc = Doctor(name=name, email=email.lower(), hashed_password=hash_password(password))
    db.add(doc); db.commit(); db.refresh(doc)
    logger.info("New doctor registered: %s", email)
    return doc

def authenticate_doctor(db: Session, email: str, password: str) -> Doctor:
    doc = get_doctor_by_email(db, email)
    if not doc or not verify_password(password, doc.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not doc.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled.")
    return doc


# ── FastAPI dependency ─────────────────────────────────────────────────────────

def get_current_doctor(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    db: Session = Depends(get_db),
) -> Doctor:
    """
    Add to any route to require authentication.
    Usage: doctor: Doctor = Depends(get_current_doctor)
    """
    payload = decode_token(credentials.credentials)
    email   = payload.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token payload.")
    doc = get_doctor_by_email(db, email)
    if not doc or not doc.is_active:
        raise HTTPException(status_code=401, detail="Doctor account not found.")
    return doc