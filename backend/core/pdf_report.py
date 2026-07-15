"""
pdf_report.py — Medical PDF Report Generator for NeuraScan
===========================================================
Uses ReportLab to generate professional medical reports.

Install: pip install reportlab

Report contains:
  - Hospital header with logo area
  - Report ID and timestamp
  - Patient details table
  - Original scan + Grad-CAM heatmap side by side
  - Diagnosis result with color coding (3-class updated)
  - Probability bars (Hemorrhagic, Ischemic, Normal)
  - Doctor notes & AI Clinical Summary (Word-wrapped)
  - Medical disclaimer footer
"""

import io
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# ── ReportLab imports ──────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.colors import (
    HexColor, white, black, Color
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF

# ── Colour palette (teal medical theme) ───────────────────────────────────────
TEAL       = HexColor("#0b6b5c")
TEAL_LIGHT = HexColor("#0e8a77")
TEAL_PALE  = HexColor("#e4f4f1")
RED        = HexColor("#c0392b")
RED_PALE   = HexColor("#fdf0ef")
GREEN      = HexColor("#1a7a42")
GREEN_PALE = HexColor("#edf8f2")
AMBER      = HexColor("#b45309")
AMBER_PALE = HexColor("#fffbeb")
GREY_DARK  = HexColor("#1a2e2a")
GREY_MID   = HexColor("#4a6b66")
GREY_LIGHT = HexColor("#e8f2f0")
GREY_BORDER= HexColor("#cce0dc")
PAGE_BG    = HexColor("#f0faf8")

W, H = A4   # 595 x 842 pts


def _styles():
    """Build all paragraph styles."""
    base = getSampleStyleSheet()
    return {
        "hospital":  ParagraphStyle("hospital",  fontSize=22, textColor=TEAL,
                                     fontName="Helvetica-Bold", spaceAfter=2),
        "tagline":   ParagraphStyle("tagline",   fontSize=9,  textColor=GREY_MID,
                                     fontName="Helvetica",     spaceAfter=0),
        "report_id": ParagraphStyle("report_id", fontSize=8,  textColor=GREY_MID,
                                     fontName="Helvetica",     alignment=TA_RIGHT),
        "section":   ParagraphStyle("section",   fontSize=11, textColor=TEAL,
                                     fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6),
        "body":      ParagraphStyle("body",      fontSize=9,  textColor=GREY_DARK,
                                     fontName="Helvetica",     spaceAfter=4, leading=14),
        "small":     ParagraphStyle("small",     fontSize=7.5,textColor=GREY_MID,
                                     fontName="Helvetica",     spaceAfter=2, leading=11),
        "verdict":   ParagraphStyle("verdict",   fontSize=18, fontName="Helvetica-Bold",
                                     alignment=TA_CENTER,      spaceAfter=4),
        "disclaimer":ParagraphStyle("disclaimer",fontSize=7,  textColor=GREY_MID,
                                     fontName="Helvetica",     leading=10, spaceAfter=2),
        "footer":    ParagraphStyle("footer",    fontSize=7,  textColor=GREY_MID,
                                     fontName="Helvetica",     alignment=TA_CENTER),
    }


def _progress_bar_drawing(pct: float, color: HexColor, width: float = 300, height: float = 14) -> Drawing:
    """Draw a filled progress bar."""
    d = Drawing(width, height)
    # Background track
    d.add(Rect(0, 0, width, height, rx=4, ry=4,
               fillColor=GREY_LIGHT, strokeColor=GREY_BORDER, strokeWidth=0.5))
    # Fill
    fill_w = max(4, width * pct / 100)
    d.add(Rect(0, 0, fill_w, height, rx=4, ry=4,
               fillColor=color, strokeColor=None, strokeWidth=0))
    return d


def _header(story, styles, report_id: str, generated_at: str):
    """Hospital header with report metadata."""

    # Top teal bar (simulate with a table)
    header_table = Table(
        [[
            Paragraph("🧠  NeuraScan", styles["hospital"]),
            Paragraph(f"Report ID: <b>{report_id}</b><br/>Generated: {generated_at}",
                      styles["report_id"]),
        ]],
        colWidths=[10 * cm, 8 * cm],
    )
    header_table.setStyle(TableStyle([
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND",  (0, 0), (-1, -1), TEAL_PALE),
        ("ROUNDEDCORNERS", [6]),
        ("TOPPADDING",  (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0),(-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",(0, 0), (-1, -1), 14),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 4))
    story.append(Paragraph("AI-Assisted Brain Stroke Detection System  ·  For Research &amp; Clinical Reference Only",
                            styles["tagline"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=TEAL, spaceAfter=10))


def _patient_section(story, styles, patient: dict):
    """Patient details table."""
    story.append(Paragraph("Patient Information", styles["section"]))

    data = [
        ["Full Name",  patient.get("name", "—"),
         "Patient ID", str(patient.get("id", "—"))],
        ["Age",        str(patient.get("age") or "—"),
         "Gender",     patient.get("gender") or "—"],
        ["Contact",    patient.get("contact") or "—",
         "Date",       patient.get("date", "—")],
    ]

    t = Table(data, colWidths=[3 * cm, 6.5 * cm, 3 * cm, 5 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME",   (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),   # key col
        ("FONTNAME",   (2, 0), (2, -1), "Helvetica-Bold"),   # key col 2
        ("TEXTCOLOR",  (0, 0), (0, -1), TEAL),
        ("TEXTCOLOR",  (2, 0), (2, -1), TEAL),
        ("TEXTCOLOR",  (1, 0), (1, -1), GREY_DARK),
        ("TEXTCOLOR",  (3, 0), (3, -1), GREY_DARK),
        ("BACKGROUND", (0, 0), (-1, -1), GREY_LIGHT),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [white, GREY_LIGHT]),
        ("GRID",       (0, 0), (-1, -1), 0.5, GREY_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1),  6),
        ("LEFTPADDING",(0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))


def _scan_images(story, styles, scan_path: Optional[str], heatmap_path: Optional[str]):
    """Original scan + Grad-CAM heatmap side by side."""
    story.append(Paragraph("Brain Scan Images", styles["section"]))

    img_w  = 8.5 * cm
    img_h  = 8.5 * cm
    cells  = []
    labels = []

    for path, label in [(scan_path, "Original Scan"), (heatmap_path, "Grad-CAM Heatmap")]:
        if path and Path(path).exists():
            try:
                img = RLImage(path, width=img_w, height=img_h)
                cells.append(img)
            except Exception:
                cells.append(Paragraph(f"[Image unavailable]", styles["small"]))
        else:
            cells.append(Paragraph(f"[{label} not available]", styles["small"]))
        labels.append(Paragraph(f"<b>{label}</b>", ParagraphStyle(
            "img_label", fontSize=8, fontName="Helvetica-Bold",
            textColor=TEAL, alignment=TA_CENTER
        )))

    img_table = Table(
        [cells, labels],
        colWidths=[img_w + 0.5 * cm, img_w + 0.5 * cm],
        rowHeights=[img_h + 4, 14],
    )
    img_table.setStyle(TableStyle([
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND",  (0, 0), (-1, 0), GREY_LIGHT),
        ("GRID",        (0, 0), (-1, 0), 0.5, GREY_BORDER),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1),  6),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(img_table)

    # Heatmap legend
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "🔬 <b>Heatmap Guide:</b> Red/yellow = high activation (regions influencing decision most). "
        "Blue = low activation.",
        styles["small"]
    ))
    story.append(Spacer(1, 8))


def _diagnosis_section(story, styles, prediction: dict):
    """Diagnosis result with colour-coded verdict and 3 probability bars."""
    story.append(Paragraph("Diagnosis Result", styles["section"]))

    is_stroke  = prediction.get("is_stroke", False)
    verdict    = prediction.get("prediction", "—")
    confidence = prediction.get("confidence", 0)
    risk       = prediction.get("risk_level", "—")
    
    # 3-Class Metrics
    hem_prob   = prediction.get("hemorrhagic_prob", 0)
    isch_prob  = prediction.get("ischemic_prob", 0)
    norm_prob  = prediction.get("normal_prob", 0)

    verdict_color = RED if is_stroke else GREEN
    bg_color      = RED_PALE if is_stroke else GREEN_PALE
    icon          = "⚠" if is_stroke else "✓"

    # Verdict box
    verdict_para = Paragraph(
        f'<font color="{"#c0392b" if is_stroke else "#1a7a42"}"><b>{icon}  {verdict}</b></font>',
        styles["verdict"]
    )
    risk_para = Paragraph(
        f'Risk Level: <b>{risk}</b>  ·  Confidence: <b>{confidence}%</b>',
        ParagraphStyle("risk_sub", fontSize=9, fontName="Helvetica",
                       textColor=GREY_MID, alignment=TA_CENTER)
    )

    verdict_table = Table([[verdict_para], [risk_para]], colWidths=[17.5 * cm])
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg_color),
        ("GRID",          (0, 0), (-1, -1), 0.5, verdict_color),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("ROUNDEDCORNERS", [6]),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 10))

    # Probability bars (3-Class Layout)
    bar_w = 280

    hem_bar  = _progress_bar_drawing(hem_prob, RED, width=bar_w)
    isch_bar = _progress_bar_drawing(isch_prob, AMBER, width=bar_w)
    norm_bar = _progress_bar_drawing(norm_prob, GREEN, width=bar_w)

    bar_data = [
        [Paragraph("<b>Hemorrhagic</b>", styles["body"]),
         hem_bar,
         Paragraph(f"<b>{hem_prob}%</b>", ParagraphStyle("pct", fontSize=9, fontName="Helvetica-Bold", textColor=RED, alignment=TA_RIGHT))],
        [Paragraph("<b>Ischemic</b>", styles["body"]),
         isch_bar,
         Paragraph(f"<b>{isch_prob}%</b>", ParagraphStyle("pct", fontSize=9, fontName="Helvetica-Bold", textColor=AMBER, alignment=TA_RIGHT))],
        [Paragraph("<b>Normal</b>", styles["body"]),
         norm_bar,
         Paragraph(f"<b>{norm_prob}%</b>", ParagraphStyle("pct2", fontSize=9, fontName="Helvetica-Bold", textColor=GREEN, alignment=TA_RIGHT))],
    ]
    bar_table = Table(bar_data, colWidths=[4.5 * cm, bar_w * 0.75, 2.2 * cm])
    bar_table.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("BACKGROUND",    (0, 0), (-1, -1), white),
        ("GRID",          (0, 0), (-1, -1), 0.3, GREY_BORDER),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(bar_table)
    story.append(Spacer(1, 8))


def _notes_section(story, styles, notes: Optional[str]):
    """Doctor notes & AI Clinical Insights section."""
    story.append(Paragraph("Doctor Notes & Clinical Summary", styles["section"]))
    
    if notes and notes.strip():
        # CRITICAL FIX: ReportLab needs <br/> tags to handle newlines correctly.
        # This prevents text from shooting off the page and correctly renders bullet points.
        formatted_notes = notes.strip().replace('\n', '<br/>')
        story.append(Paragraph(formatted_notes, styles["body"]))
    else:
        story.append(Paragraph("No notes recorded for this scan.", styles["body"]))
        
    story.append(Spacer(1, 8))


def _disclaimer_section(story, styles):
    """Medical disclaimer."""
    story.append(HRFlowable(width="100%", thickness=0.5, color=GREY_BORDER, spaceBefore=8))
    story.append(Spacer(1, 4))
    story.append(Paragraph("⚠  Medical Disclaimer", ParagraphStyle(
        "disc_title", fontSize=8, fontName="Helvetica-Bold",
        textColor=AMBER, spaceAfter=3
    )))
    story.append(Paragraph(
        "This report is generated by an AI-assisted system (NeuraScan) for research and "
        "educational purposes only. It does not constitute a medical diagnosis. Results must "
        "be reviewed and confirmed by a qualified radiologist or neurologist before any "
        "clinical decisions are made. The model's predictions may contain errors. Always "
        "seek professional medical advice for health concerns.",
        styles["disclaimer"]
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "NeuraScan v5.0.0  ·  Powered by ResNet50 + Gemini Multimodal AI  ·  © 2026 NeuraScan",
        styles["footer"]
    ))


def generate_pdf_report(
    patient: dict,
    prediction: dict,
    scan_path: Optional[str] = None,
    heatmap_path: Optional[str] = None,
    report_id: Optional[str] = None,
) -> Path:
    """
    Generate a complete medical PDF report.
    """
    if report_id is None:
        report_id = f"NSR-{str(uuid.uuid4()).upper()[:8]}"

    generated_at = datetime.now().strftime("%d %b %Y, %I:%M %p")
    patient["date"] = datetime.now().strftime("%d %b %Y")

    # Output path
    filename = f"report_{report_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    output_path = REPORTS_DIR / filename

    # Build document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title=f"NeuraScan Report — {patient.get('name', 'Patient')}",
        author="NeuraScan AI",
    )

    styles = _styles()
    story  = []

    _header(story, styles, report_id, generated_at)
    _patient_section(story, styles, patient)
    _scan_images(story, styles, scan_path, heatmap_path)
    _diagnosis_section(story, styles, prediction)
    _notes_section(story, styles, prediction.get("doctor_notes"))
    _disclaimer_section(story, styles)

    doc.build(story)
    logger.info("✅ PDF report generated: %s", output_path)
    return output_path