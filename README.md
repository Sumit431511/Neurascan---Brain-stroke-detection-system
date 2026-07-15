# 🧠 NeuraScan — AI Brain Stroke Detection System

A production-grade medical AI system that detects brain strokes from CT/MRI scans using deep learning, with Grad-CAM explainability, automated PDF reports, JWT authentication, and a doctor dashboard.

---

## 📁 Project Structure

```
NeuraScan/
├── backend/
│   ├── main.py              ← App entry point (50 lines — routers + startup only)
│   ├── config.py            ← All settings and constants in one place
│   │
│   ├── api/                 ← Routes separated by feature
│   │   ├── auth.py          ← POST /auth/login, /auth/register, GET /auth/me
│   │   ├── patients.py      ← CRUD /patients
│   │   ├── predictions.py   ← POST /predict, GET /predictions, PDF download
│   │   └── dashboard.py     ← GET /dashboard/stats, /dashboard/monthly
│   │
│   ├── core/                ← Business logic (no routes here)
│   │   ├── model.py         ← Model loading + inference
│   │   ├── preprocessing.py ← Image preprocessing pipeline
│   │   ├── gradcam.py       ← Grad-CAM heatmap generation
│   │   ├── pdf_report.py    ← ReportLab PDF generator
│   │   └── auth.py          ← JWT + password hashing
│   │
│   ├── db/                  ← Everything database
│   │   ├── database.py      ← SQLAlchemy engine + session
│   │   ├── models.py        ← ORM table definitions
│   │   ├── schemas.py       ← Pydantic request/response shapes
│   │   └── crud.py          ← All DB read/write operations
│   │
│   ├── models/              ← ML model weights (not committed to git)
│   ├── heatmaps/            ← Auto-created: saved heatmap images
│   ├── scans/               ← Auto-created: saved scan images
│   ├── reports/             ← Auto-created: generated PDF reports
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── login.html           ← Doctor login / register
│   ├── index.html           ← Main scan upload page
│   └── dashboard.html       ← Doctor analytics dashboard
│
├── .gitignore
├── docker-compose.yml
└── README.md
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔬 **AI Prediction** | ResNet50 CNN — binary stroke/normal classification |
| 🗺️ **Grad-CAM** | Visual heatmap showing which brain regions influenced the decision |
| 📄 **PDF Reports** | Auto-generated medical reports with scan images |
| 🔐 **JWT Auth** | Doctor login/register with bcrypt passwords and 24hr tokens |
| 🗄️ **Database** | SQLite (local) / PostgreSQL (production) via SQLAlchemy |
| 📊 **Dashboard** | Real-time analytics, charts, patient history, reviewed workflow |
| 🌙 **Dark Mode** | Theme toggle with localStorage persistence |

---

## 🏗️ System Flow

```
Doctor logs in → JWT token issued
        ↓
Upload brain scan + patient details
        ↓
Backend preprocesses image (ResNet50 normalisation)
        ↓
ML model predicts stroke probability
        ↓
Grad-CAM heatmap generated
        ↓
Result + images saved to database
        ↓
PDF report auto-generated
        ↓
Doctor dashboard shows live analytics
        ↓
Doctor downloads PDF / marks as reviewed
```

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/yourusername/neurascan.git
cd neurascan/backend

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add model weights
# Place model_weights.npy in backend/models/

# 5. Start server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 6. Open frontend/login.html in browser
```

**API docs:** `http://localhost:8000/docs`

---

## 📡 API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/` | ❌ | Health check |
| POST | `/auth/register` | ❌ | Register doctor |
| POST | `/auth/login` | ❌ | Login, get token |
| GET | `/auth/me` | ✅ | Current doctor |
| POST | `/patients` | ✅ | Register patient |
| GET | `/patients` | ✅ | List patients |
| GET | `/patients/search` | ✅ | Search by name |
| GET | `/patients/{id}` | ✅ | Get patient |
| DELETE | `/patients/{id}` | ✅ | Delete patient |
| POST | `/predictions` | ✅ | Upload scan + predict |
| GET | `/predictions` | ✅ | List all predictions |
| GET | `/predictions/{id}` | ✅ | Get prediction |
| GET | `/predictions/{id}/report` | ✅ | Download PDF |
| PATCH | `/predictions/{id}/notes` | ✅ | Update doctor notes |
| PATCH | `/predictions/{id}/reviewed` | ✅ | Toggle reviewed |
| GET | `/dashboard/stats` | ✅ | Dashboard stats |
| GET | `/dashboard/monthly` | ✅ | Monthly chart data |

---

## 🔬 Model Details

| Property | Value |
|---|---|
| Architecture | ResNet50 (ImageNet pretrained) |
| Fine-tuned layers | Last 50 layers |
| Input | 224 × 224 × 3 RGB |
| Preprocessing | ResNet50 `preprocess_input` |
| Output | Sigmoid [0–1] |
| Threshold | 0.35 (tuned for recall) |
| Grad-CAM layer | `conv5_block3_out` |

---

## 🛠️ Tech Stack

**Backend:** FastAPI · SQLAlchemy · python-jose · passlib · ReportLab · TensorFlow · tf-keras

**Frontend:** HTML/CSS/JS · Chart.js

**Database:** SQLite → PostgreSQL

**ML:** ResNet50 · Transfer Learning · Grad-CAM

---

## ⚠️ Disclaimer

For research and educational use only. Not for clinical diagnosis.