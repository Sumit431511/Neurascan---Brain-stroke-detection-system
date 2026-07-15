"""
api/predictions.py — Prediction routes
POST /predictions, GET /predictions, PDF download
3-class: Hemorrhagic / Ischemic / Normal
"""

import logging
import re
from typing import List
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Depends, Form
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session

from config import ALLOWED_IMAGE_TYPES, MAX_FILE_BYTES
from db.database import get_db
from db.schemas import PredictionOut, UpdateNotesRequest, UpdateReviewedRequest
from db.models import Doctor
from core.auth import get_current_doctor
from core.preprocessing import preprocess_image, save_scan
from core.model import run_inference
import db.crud as crud
from core.ood_detection import is_valid_brain_scan
from core.llm_insights import generate_clinical_insights

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Predictions"])


@router.post("")
async def predict(
    file:             UploadFile = File(...),
    patient_id:       int        = Form(...),
    doctor_notes:     str        = Form(default=""),
    generate_heatmap: bool       = Query(default=True),
    db:               Session    = Depends(get_db),
    doctor:           Doctor     = Depends(get_current_doctor),
):
    """
    Upload a brain scan and get 3-class stroke prediction.
    Returns: Hemorrhagic / Ischemic / Normal
    Optimized: PDF generation is now deferred until the user clicks Download.
    """
    # Validate patient
    patient = crud.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found.")

    # Validate file
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type.")
    
    image_bytes = await file.read()
    
    if len(image_bytes) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Max 10 MB.")
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    # =================================================================
    # Out-of-Distribution (OOD) Safety Check
    # =================================================================
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    if not is_valid_brain_scan(image):
        raise HTTPException(
            status_code=400, 
            detail="AI Safety Alert: Image rejected. The uploaded file does not appear to be a valid CT or MRI brain scan."
        )
    # =================================================================

    # Save original scan
    scan_path = save_scan(image_bytes, file.filename or "scan.jpg")

    # Preprocess + inference
    try:
        img_array = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image processing error: {e}")

    try:
        result = run_inference(img_array)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # Grad-CAM heatmap & LLM Insights
    heatmap_b64  = None
    heatmap_path = None
    llm_insights_html = None  

    if generate_heatmap:
        try:
            from core.gradcam import generate_gradcam, save_heatmap, heatmap_to_base64
            _, overlay   = generate_gradcam(model=_get_model(), img_array=img_array)
            heatmap_b64  = heatmap_to_base64(overlay)
            heatmap_path = str(save_heatmap(overlay))
            
            # Generate LLM Insights
            try:
                llm_insights_html = generate_clinical_insights(
                    stroke_type=result["stroke_type"],
                    confidence=result["confidence"],
                    patient_data={"age": patient.age, "gender": patient.gender},
                    gradcam_image_path=heatmap_path
                )
            except Exception as e:
                logger.warning("LLM generation failed (non-fatal): %s", e)

        except Exception as e:
            logger.warning("Grad-CAM failed (non-fatal): %s", e)

   # =================================================================
    # NEW: Auto-Draft PDF Notes using AI Insights
    # =================================================================
    import html
    import re
    
    final_notes = doctor_notes.strip() if doctor_notes else ""
    
    if llm_insights_html:
        clean_text = llm_insights_html
        
        # 1. Add proper line breaks for block elements
        clean_text = clean_text.replace('<p>', '\n').replace('</p>', '\n')
        clean_text = clean_text.replace('<br>', '\n').replace('<br/>', '\n')
        clean_text = clean_text.replace('<ul>', '\n').replace('</ul>', '\n')
        
        # 2. Format list items as actual indented bullet points
        clean_text = clean_text.replace('<li>', '  • ').replace('</li>', '\n')
        
        # 3. Strip all other HTML tags (like <strong>, <b>, <em>)
        clean_text = re.sub(r'<[^>]+>', '', clean_text)
        
        # 4. Decode HTML entities (e.g., changes &amp; back to &)
        clean_text = html.unescape(clean_text)
        
        # 5. Clean up excessive empty lines
        clean_text = re.sub(r'\n{3,}', '\n\n', clean_text).strip()
        
        ai_header = "\n\n=== AI GENERATED CLINICAL SUMMARY ===\n\n" if final_notes else "=== AI GENERATED CLINICAL SUMMARY ===\n\n"
        final_notes = final_notes + ai_header + clean_text
    # =================================================================

    # Save to database
    record = crud.create_prediction(
        db=db,
        patient_id=patient_id,
        prediction=result["prediction"],
        stroke_type=result["stroke_type"],
        is_stroke=result["is_stroke"],
        confidence=result["confidence"],
        hemorrhagic_prob=result["hemorrhagic_prob"],
        ischemic_prob=result["ischemic_prob"],
        normal_prob=result["normal_prob"],
        risk_level=result["risk_level"],
        filename=file.filename,
        scan_path=scan_path,
        heatmap_path=heatmap_path,
        doctor_notes=final_notes.strip() or None,  # <-- Injects AI summary here!
    )
    logger.info("✅ Prediction saved — ID: %d | Patient: %s | Result: %s",
                record.id, patient.name, result["prediction"])

    return JSONResponse({
        **result,
        "prediction_id":          record.id,
        "patient_id":             patient_id,
        "patient_name":           patient.name,
        "doctor":                 doctor.name,
        "filename":               file.filename,
        "heatmap_base64":         heatmap_b64,
        "heatmap_path":           heatmap_path,
        "clinical_insights_html": llm_insights_html, # Sends HTML to Frontend
        "pdf_url":                f"/predictions/{record.id}/report", 
        "saved_to_db":            True,
    })


def _get_model():
    from core.model import get_model
    return get_model()


@router.get("", response_model=List[PredictionOut])
def list_predictions(
    skip:   int = 0,
    limit:  int = 200,
    db:     Session = Depends(get_db),
    doctor: Doctor  = Depends(get_current_doctor),
):
    return crud.get_all_predictions(db, skip=skip, limit=limit)


@router.get("/recent", response_model=List[PredictionOut])
def recent_predictions(
    limit:  int = 10,
    db:     Session = Depends(get_db),
    doctor: Doctor  = Depends(get_current_doctor),
):
    return crud.get_recent_predictions(db, limit=limit)


@router.get("/{prediction_id}", response_model=PredictionOut)
def get_prediction(
    prediction_id: int,
    db:     Session = Depends(get_db),
    doctor: Doctor  = Depends(get_current_doctor),
):
    r = crud.get_prediction(db, prediction_id)
    if not r:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return r


@router.patch("/{prediction_id}/notes", response_model=PredictionOut)
def update_notes(
    prediction_id: int,
    body:   UpdateNotesRequest,
    db:     Session = Depends(get_db),
    doctor: Doctor  = Depends(get_current_doctor),
):
    r = crud.update_notes(db, prediction_id, body.doctor_notes)
    if not r:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return r


@router.patch("/{prediction_id}/reviewed")
def toggle_reviewed(
    prediction_id: int,
    body:   UpdateReviewedRequest,
    db:     Session = Depends(get_db),
    doctor: Doctor  = Depends(get_current_doctor),
):
    r = crud.update_reviewed(db, prediction_id, body.reviewed)
    if not r:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return {"id": prediction_id, "reviewed": body.reviewed}


@router.delete("/{prediction_id}")
def delete_prediction(
    prediction_id: int,
    db:     Session = Depends(get_db),
    doctor: Doctor  = Depends(get_current_doctor),
):
    if not crud.delete_prediction(db, prediction_id):
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return {"message": f"Prediction {prediction_id} deleted."}


@router.get("/{prediction_id}/report")
def download_report(
    prediction_id: int,
    db:     Session = Depends(get_db),
    doctor: Doctor  = Depends(get_current_doctor),
):
    """Download PDF report. Regenerates on-demand if missing."""
    record = crud.get_prediction(db, prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found.")

    if not record.pdf_report_path or not Path(record.pdf_report_path).exists():
        try:
            from core.pdf_report import generate_pdf_report
            patient  = crud.get_patient(db, record.patient_id)
            pdf_file = generate_pdf_report(
                patient={
                    "id": patient.id, "name": patient.name,
                    "age": patient.age, "gender": patient.gender,
                    "contact": patient.contact,
                },
                prediction={
                    "id":               record.id,
                    "prediction":       record.prediction,
                    "stroke_type":      record.stroke_type,
                    "is_stroke":        record.is_stroke,
                    "confidence":       record.confidence,
                    "hemorrhagic_prob": record.hemorrhagic_prob,
                    "ischemic_prob":    record.ischemic_prob,
                    "normal_prob":      record.normal_prob,
                    "risk_level":       record.risk_level,
                    "doctor_notes":     record.doctor_notes,
                },
                scan_path=record.scan_path,
                heatmap_path=record.heatmap_path,
            )
            crud.update_pdf_path(db, record.id, str(pdf_file))
            record.pdf_report_path = str(pdf_file)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

    patient     = crud.get_patient(db, record.patient_id)
    safe_name   = (patient.name if patient else "patient").replace(" ", "_")
    dl_filename = f"NeuraScan_{safe_name}_Report_{record.id}.pdf"

    return FileResponse(
        path=record.pdf_report_path,
        media_type="application/pdf",
        filename=dl_filename,
    )