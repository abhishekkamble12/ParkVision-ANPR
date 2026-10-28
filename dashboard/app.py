"""
=============================================================
  ParkVision — Streamlit Dashboard
=============================================================
Real-time parking management dashboard with:
  - Live plate detection from webcam or uploaded image
  - Active session monitor (who is currently parked)
  - Session history table
  - Performance metrics (OCR confidence histogram, FPS)

Run with:
    streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import time
import datetime
import numpy as np
import streamlit as st
import requests

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ParkVision — Smart Parking ANPR",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

# ─── Custom CSS for premium look ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0f1117; }

    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #252a3a 100%);
        border: 1px solid #2d3450;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-val  { font-size: 2.5rem; font-weight: 700; color: #00e5aa; }
    .metric-lbl  { font-size: 0.85rem; color: #8892b0; margin-top: 4px; }

    .plate-badge {
        background: linear-gradient(90deg, #00e5aa22, #0066ff22);
        border: 1px solid #00e5aa55;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 1.3rem;
        font-weight: 700;
        font-family: 'Courier New', monospace;
        color: #00e5aa;
        letter-spacing: 3px;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00e5aa, #0066ff);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
    }
    .stButton > button:hover { opacity: 0.85; transform: scale(1.02); }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def api_get(endpoint: str):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


def api_post(endpoint: str, payload: dict):
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


def api_post_image(endpoint: str, image_bytes: bytes, filename: str):
    try:
        r = requests.post(
            f"{API_URL}{endpoint}",
            files={"file": (filename, image_bytes, "image/jpeg")},
            timeout=30
        )
        return r.json() if r.ok else None
    except Exception:
        return None


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 ParkVision")
    st.markdown("*Indian Smart Parking · ANPR*")
    st.divider()

    page = st.radio("Navigate", [
        "🏠 Live Detection",
        "📋 Active Sessions",
        "📊 Session History",
        "📈 Metrics",
    ])

    st.divider()
    health = api_get("/health")
    if health:
        st.success("🟢 API Online")
    else:
        st.error("🔴 API Offline — start FastAPI:\n`uvicorn api.main:app --reload`")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: LIVE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Live Detection":
    st.title("🏠 Live Number Plate Detection")
    st.markdown("Upload a vehicle image to detect and recognize the number plate.")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### 📤 Upload Image")
        uploaded = st.file_uploader(
            "Drop a vehicle photo here",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        action_col, entry_col, exit_col = st.columns(3)
        detect_btn = action_col.button("🔍 Detect Plate", use_container_width=True)

    with col2:
        st.markdown("#### 🎯 Detection Result")
        result_placeholder = st.empty()

    if uploaded and detect_btn:
        img_bytes = uploaded.read()
        with st.spinner("Running YOLO + EasyOCR pipeline..."):
            results = api_post_image("/detect/image", img_bytes, uploaded.name)

        if results is None:
            st.error("API unreachable. Make sure FastAPI is running.")
        elif len(results) == 0:
            st.warning("No number plates detected in this image.")
        else:
            # Draw boxes on image
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            for det in results:
                x1, y1, x2, y2 = det["bbox"]
                label = f"{det['plate_number']} ({det['confidence']*100:.0f}%)"
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 229, 170), 3)
                cv2.rectangle(img_rgb, (x1, y1 - 30), (x1 + len(label) * 11, y1), (0, 229, 170), -1)
                cv2.putText(img_rgb, label, (x1 + 4, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

            with col1:
                st.image(img_rgb, caption="Annotated Result", use_column_width=True)

            with result_placeholder.container():
                for det in results:
                    st.markdown(f'<div class="plate-badge">{det["plate_number"] or "---"}</div>',
                                unsafe_allow_html=True)
                    validity = "✅ Valid Indian Plate" if det["is_valid_indian_plate"] else "⚠️ Unrecognized Format"
                    st.markdown(f"**{validity}** &nbsp;|&nbsp; Confidence: `{det['confidence']*100:.1f}%`")
                    st.caption(f"Raw OCR: `{det['raw_ocr']}`")
                    st.divider()

                    # Quick entry/exit buttons
                    c1, c2 = st.columns(2)
                    if c1.button(f"🟢 Log Entry — {det['plate_number']}", key=f"entry_{det['plate_number']}"):
                        resp = api_post("/parking/entry", {"plate_number": det["plate_number"]})
                        st.success(f"Entry logged!" if resp else "Error logging entry")
                    if c2.button(f"🔴 Log Exit — {det['plate_number']}", key=f"exit_{det['plate_number']}"):
                        resp = api_post("/parking/exit", {"plate_number": det["plate_number"]})
                        st.success(f"Exit logged!" if resp else "No active session found")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: ACTIVE SESSIONS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋 Active Sessions":
    st.title("📋 Currently Parked Vehicles")

    stats = api_get("/parking/stats")
    active = api_get("/parking/active") or []

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("🚗", stats["active_sessions"] if stats else "—", "Currently Parked"),
        ("✅", stats["completed_sessions"] if stats else "—", "Completed Today"),
        ("⏱️", f"{stats['avg_duration_mins']} min" if stats else "—", "Avg. Duration"),
        ("📊", stats["total_detections"] if stats else "—", "Total Detections"),
    ]
    for col, (icon, val, lbl) in zip([m1, m2, m3, m4], metrics):
        col.markdown(
            f'<div class="metric-card"><div class="metric-val">{icon} {val}</div>'
            f'<div class="metric-lbl">{lbl}</div></div>',
            unsafe_allow_html=True
        )

    st.divider()

    if not active:
        st.info("No vehicles currently parked.")
    else:
        for s in active:
            with st.expander(f"🚗 **{s['plate_number']}** — Slot: {s['slot_id'] or 'Unassigned'}", expanded=True):
                entry_dt = datetime.datetime.fromisoformat(s["entry_time"])
                duration_so_far = (datetime.datetime.utcnow() - entry_dt).seconds // 60
                c1, c2, c3 = st.columns(3)
                c1.metric("Entry Time", entry_dt.strftime("%H:%M:%S"))
                c2.metric("Time Parked", f"{duration_so_far} min")
                c3.metric("OCR Confidence", f"{s['ocr_confidence']*100:.0f}%")

                if st.button(f"🔴 Log Exit", key=f"exit_active_{s['id']}"):
                    resp = api_post("/parking/exit", {"plate_number": s["plate_number"]})
                    if resp:
                        st.success(f"Exit logged. Duration: {resp.get('duration_mins', '?')} min")
                        st.rerun()

    if st.button("🔄 Refresh"):
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: SESSION HISTORY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Session History":
    st.title("📊 Parking Session History")

    sessions = api_get("/parking/sessions") or []
    if not sessions:
        st.info("No sessions recorded yet.")
    else:
        import pandas as pd
        df = pd.DataFrame(sessions)
        df["status"] = df["is_active"].map({True: "🟢 Active", False: "✅ Completed"})
        df["duration_mins"] = df["duration_mins"].fillna("—")
        st.dataframe(
            df[["plate_number", "status", "entry_time", "exit_time", "duration_mins", "slot_id", "ocr_confidence"]],
            use_container_width=True,
            height=500
        )

    # Manual entry / exit
    st.divider()
    st.markdown("#### ✍️ Manual Entry / Exit")
    plate_input = st.text_input("Plate Number", placeholder="MH 12 AB 1234")
    c1, c2 = st.columns(2)
    if c1.button("🟢 Log Entry"):
        if plate_input:
            resp = api_post("/parking/entry", {"plate_number": plate_input})
            st.success("Entry logged!") if resp else st.error("Failed")
    if c2.button("🔴 Log Exit"):
        if plate_input:
            resp = api_post("/parking/exit", {"plate_number": plate_input})
            st.success(f"Exit logged. {resp.get('duration_mins', '?')} min") if resp else st.error("No active session")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: METRICS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Metrics":
    st.title("📈 Pipeline Performance Metrics")
    st.markdown("Based on raw detection events logged to the database.")

    logs = api_get("/detections/log?limit=500") or []
    if not logs:
        st.info("No detection events yet. Run the detection pipeline first.")
    else:
        import pandas as pd
        df = pd.DataFrame(logs)
        df["confidence_pct"] = df["confidence"] * 100

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### OCR Confidence Distribution")
            st.bar_chart(df["confidence_pct"].round(0).value_counts().sort_index())

        with col2:
            st.markdown("#### Valid Indian Plate Rate")
            valid_count = df["is_valid_ind"].sum()
            total = len(df)
            rate = valid_count / total * 100 if total else 0
            st.metric("Valid Plate Recognition Rate", f"{rate:.1f}%")
            st.metric("Total Detections", total)
            st.metric("Avg OCR Confidence", f"{df['confidence_pct'].mean():.1f}%")

        st.divider()
        st.markdown("#### 📋 Recent Detection Log")
        st.dataframe(
            df[["timestamp", "plate_number", "raw_ocr_text", "confidence", "is_valid_ind", "source"]].head(50),
            use_container_width=True
        )
