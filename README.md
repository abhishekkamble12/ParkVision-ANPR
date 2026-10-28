<div align="center">

<h1>🚗 ParkVision — Indian Smart Parking & ANPR System</h1>

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00C851?style=for-the-badge" />
  <img src="https://img.shields.io/badge/EasyOCR-Recognition-FF6F00?style=for-the-badge" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/Docker-Deploy-2496ED?style=for-the-badge&logo=docker" />
</p>

<p><strong>End-to-end Automatic Number Plate Recognition (ANPR) system for Indian vehicles.<br>
Detects plates with YOLOv8, reads characters with EasyOCR, and manages parking sessions through a FastAPI + Streamlit stack.</strong></p>

</div>

---

## 📸 What This System Does

| Stage | Technology | Description |
|-------|-----------|-------------|
| 🔍 **Plate Detection** | YOLOv8 | Locates number plates in images/video frames |
| 🖼️ **Preprocessing** | OpenCV | 7-stage pipeline: CLAHE → Blur → Adaptive Threshold → Deskew |
| 🔤 **OCR** | EasyOCR | Reads plate characters; zone-aware O→0, I→1 correction |
| 🎯 **Tracking** | Centroid Tracker | Deduplicates plates across consecutive video frames |
| 🅿️ **Parking Logic** | SQLite + FastAPI | Entry/exit sessions, duration, dedup cooldown |
| 📊 **Dashboard** | Streamlit | Live detection upload, session monitor, metrics |
| 🐳 **Deployment** | Docker Compose | One-command full-stack deployment |

---

## 🏗️ Project Structure

```
ParkVision/
│
├── core/
│   ├── detector.py         # YOLOv8 dual-strategy plate detector
│   ├── ocr_engine.py       # EasyOCR + Indian plate regex + char correction
│   ├── preprocessor.py     # 7-stage OpenCV preprocessing pipeline
│   └── tracker.py          # Centroid tracker for frame deduplication
│
├── parking/
│   ├── models.py           # SQLAlchemy ORM (ParkingSession, DetectionLog)
│   └── session_manager.py  # Entry/exit logic, dedup cooldown, stats
│
├── api/
│   └── main.py             # FastAPI REST backend (7 endpoints)
│
├── dashboard/
│   └── app.py              # Streamlit 4-page dashboard
│
├── scripts/
│   ├── run_image.py        # ▶ Run image pipeline (start here)
│   └── run_video.py        # ▶ Run video/webcam pipeline
│
├── tests/
│   └── test_pipeline.py    # CER, plate accuracy, speed benchmarks
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── data/                   # Auto-created on first run
│   ├── parkvision.db       # SQLite database
│   ├── crops/              # Saved plate crop images
│   └── output_images/      # Annotated result images
│
└── requirements.txt
```

---

## 🚀 Quick Start

### 1 — Install Dependencies

```bash
git clone https://github.com/YOUR_USERNAME/ParkVision-ANPR.git
cd ParkVision-ANPR
pip install -r requirements.txt
```

> First run downloads YOLOv8 weights (~6 MB) and EasyOCR models (~150 MB) automatically.

### 2 — Run Image Pipeline (Fastest Demo)

```bash
python scripts/run_image.py
# With your own image:
python scripts/run_image.py --image path/to/car.jpg
# With debug preprocessing stages saved:
python scripts/run_image.py --debug
```

### 3 — Run Video / Webcam Pipeline

```bash
# Webcam (live)
python scripts/run_video.py

# Video file
python scripts/run_video.py --video parking_footage.mp4

# Headless (no window, for servers)
python scripts/run_video.py --video footage.mp4 --headless
```

### 4 — Start FastAPI Backend

```bash
uvicorn api.main:app --reload --port 8000
# API docs at: http://localhost:8000/docs
```

### 5 — Launch Streamlit Dashboard

```bash
streamlit run dashboard/app.py
# Dashboard at: http://localhost:8501
```

### 6 — Docker (Full Stack, One Command)

```bash
cd docker
docker-compose up --build
# API  → http://localhost:8000
# Dashboard → http://localhost:8501
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/detect/image` | Upload image → returns plates + OCR text + bbox |
| `POST` | `/parking/entry` | Log vehicle entry |
| `POST` | `/parking/exit` | Log vehicle exit + compute duration |
| `GET` | `/parking/active` | List currently parked vehicles |
| `GET` | `/parking/sessions` | Full session history |
| `GET` | `/parking/stats` | Dashboard statistics |
| `GET` | `/detections/log` | Raw OCR detection events |

**Swagger UI:** `http://localhost:8000/docs`

---

## 🧠 Technical Deep-Dives

### Preprocessing Pipeline (7 Stages)
```
Raw Crop → Resize (64px height) → Grayscale → CLAHE →
Gaussian Blur → Adaptive Threshold → Morphological Close → Deskew
```
- **CLAHE** handles uneven lighting and shadowed plates better than global histogram equalization
- **Adaptive Threshold** computes local cutoffs per 11×11 block — essential for reflections
- **Deskew** via Hough Line Transform corrects tilt up to ±10°

### Dual-Strategy Detection
1. **Primary**: Fine-tuned plate-specific YOLO model (`keremberke/yolov8n-license-plate-detection`)
2. **Fallback**: Generic YOLOv8 → detect vehicle → search bottom 40% ROI → contour aspect ratio filter (2:1 to 6.5:1)

### Indian Plate Format Support
```
Standard:  MH 12 AB 1234   (State · District · Series · Number)
BH Series: 22 BH 1234 AB   (Bharat series, nationwide)
```

### Zone-Aware OCR Correction
Characters are corrected based on their position in the plate:
- **Alpha zones** (positions 0-1, 4-6): digit lookalikes → letters (`0→O`)
- **Numeric zones** (positions 2-3, 7-10): letter lookalikes → digits (`O→0`, `I→1`, `S→5`)

### Frame Deduplication
Centroid tracker assigns persistent IDs across frames. OCR runs **once per track** at peak confidence — not 30× per second per vehicle.

---

## 📊 Metrics & Evaluation

Run the evaluation suite:
```bash
python tests/test_pipeline.py
```

Measures across 4 difficulty conditions (clean / noisy / tilted / hard):
- **Character Error Rate (CER)** = `edit_distance(pred, gt) / len(gt)`
- **Complete Plate Accuracy** (exact match %)
- **Preprocessing throughput** (FPS)
- **OCR character correction** unit tests

---

## 🛠️ Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| Object Detection | Ultralytics YOLOv8 | ≥8.0 |
| OCR Engine | EasyOCR | ≥1.7 |
| Image Processing | OpenCV | ≥4.8 |
| REST Backend | FastAPI + Uvicorn | ≥0.111 |
| Database ORM | SQLAlchemy + SQLite | ≥2.0 |
| Dashboard | Streamlit | ≥1.35 |
| Containerization | Docker + Compose | latest |

---

## 🇮🇳 Ethics & Privacy

- Only process **consented footage** or public-domain images
- Plate numbers are **not retained** beyond session duration in production mode
- Database stores only what is needed for parking management

---

## 👤 Author

**Abhishek Kamble** — [GitHub](https://github.com/YOUR_USERNAME)

> Integrated a pretrained YOLOv8 detector into a full ANPR pipeline; built the preprocessing, OCR correction, centroid tracking, and parking logic from scratch.
