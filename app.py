import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import io
import os
import re
import hashlib
import psycopg2
import pytesseract

from datetime import datetime
from zoneinfo import ZoneInfo
from psycopg2.extras import RealDictCursor


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="KRCL | Thermal Intelligence",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

IST = ZoneInfo("Asia/Kolkata")
FAULT_THRESHOLD = 5.0


# ============================================================
# OEM REFERENCE DATA
# ============================================================

OEM_DATA = {
    "20250315-133514-016": {"max": 39.5, "min": 37.6},
    "20251111-112915-003": {"max": 31.5, "min": 21.9},
    "20251111-114541-001": {"max": 35.6, "min": 7.2},
    "20251111-114605-002": {"max": 35.4, "min": 27.4},
    "20251113-074547-002": {"max": 17.6, "min": 13.7},
    "20251113-091808-012": {"max": 24.7, "min": 19.3},
    "20251113-092707-005": {"max": 24.4, "min": 20.5},
    "20251113-093327-012": {"max": 24.0, "min": 18.3},
    "20251113-113151-016": {"max": 34.5, "min": 24.7},
    "20251114-101128-007": {"max": 26.2, "min": 15.8},
    "20251114-114841-018": {"max": 35.0, "min": 13.6},
    "20251117-124611-004": {"max": 33.8, "min": 22.8},
}


# ============================================================
# PREMIUM INTERFACE
# ============================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #080f1c;
        color: #e8eef8;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #101c30 0%,
            #0b1424 100%
        );
        border-right: 1px solid #24354d;
    }

    [data-testid="stSidebar"] * {
        color: #e8eef8;
    }

    .hero {
        background: linear-gradient(
            115deg,
            #12243c 0%,
            #123d59 55%,
            #087e8b 100%
        );
        border: 1px solid #24536b;
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 24px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.18);
    }

    .hero h1 {
        font-size: 34px;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 10px 0;
    }

    .hero p {
        color: #c8e5ef;
        font-size: 14px;
        margin: 0;
    }

    .eyebrow {
        color: #56e0e8;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }

    .section-heading {
        color: #f2f7ff;
        font-size: 21px;
        font-weight: 750;
        margin-top: 18px;
        margin-bottom: 12px;
    }

    .section-caption {
        color: #91a5bf;
        font-size: 13px;
        margin-top: -8px;
        margin-bottom: 20px;
    }

    .metric-card {
        background: linear-gradient(145deg, #142238, #101a2b);
        border: 1px solid #253750;
        border-radius: 16px;
        padding: 20px;
        min-height: 125px;
    }

    .metric-label {
        color: #9db0c9;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 28px;
        font-weight: 800;
        line-height: 1.2;
    }

    .metric-subtitle {
        color: #68d9e4;
        font-size: 11px;
        margin-top: 8px;
    }

    .status-good {
        background: #10382e;
        border: 1px solid #247b5e;
        color: #78efbd;
        padding: 15px;
        border-radius: 12px;
        font-weight: 700;
    }

    .status-bad {
        background: #411d29;
        border: 1px solid #9b3c50;
        color: #ff9eaa;
        padding: 15px;
        border-radius: 12px;
        font-weight: 700;
    }

    .status-neutral {
        background: #202d41;
        border: 1px solid #42546c;
        color: #d7e4f5;
        padding: 15px;
        border-radius: 12px;
        font-weight: 700;
    }

    .info-card {
        background: #111d30;
        border: 1px solid #263952;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 14px;
    }

    .info-card h4 {
        color: #f2f7ff;
        margin-top: 0;
        margin-bottom: 8px;
    }

    .info-card p {
        color: #a8bad0;
        font-size: 13px;
    }

    .small-label {
        color: #90a5c0;
        font-size: 12px;
    }

    div[data-testid="stMetric"] {
        background: #142238;
        border: 1px solid #253750;
        padding: 16px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #a8bad0;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }

    .stButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 700;
        border: 1px solid #2e8c9d;
        transition: 0.2s ease;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #087e8b, #11a6b5);
        color: white;
        border: none;
    }

    .stButton > button:hover {
        border-color: #53dce6;
        color: white;
    }

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"],
    .stDateInput input {
        background: #111d30;
        color: white;
        border-radius: 9px;
    }

    [data-testid="stFileUploader"] {
        background: #111d30;
        border: 1px dashed #3a647e;
        border-radius: 14px;
        padding: 12px;
    }

    hr {
        border-color: #27364c;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .footer-note {
        color: #71859f;
        font-size: 11px;
        text-align: center;
        padding: 22px 0 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_database_connection():
    database_url = st.secrets.get("DATABASE_URL", "")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing from Streamlit Secrets."
        )

    return psycopg2.connect(database_url)


def initialize_database():
    conn = get_database_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS thermal_inspections (
                    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    image_filename TEXT NOT NULL,
                    image_id TEXT,
                    image_datetime TIMESTAMPTZ,
                    uploaded_at TIMESTAMPTZ NOT NULL,
                    minimum_temperature DOUBLE PRECISION,
                    maximum_temperature DOUBLE PRECISION,
                    temperature_difference DOUBLE PRECISION,
                    mean_temperature DOUBLE PRECISION,
                    fault_status TEXT NOT NULL,
                    recommendation TEXT,
                    temperature_source TEXT NOT NULL,
                    image_sha256 TEXT NOT NULL,
                    image_data BYTEA NOT NULL,
                    station_name TEXT,
                    ohe_mast_number TEXT
                );
                """
            )

            cur.execute(
                """
                ALTER TABLE thermal_inspections
                ADD COLUMN IF NOT EXISTS station_name TEXT;
                """
            )

            cur.execute(
                """
                ALTER TABLE thermal_inspections
                ADD COLUMN IF NOT EXISTS ohe_mast_number TEXT;
                """
            )

        conn.commit()

    finally:
        conn.close()


# ============================================================
# FILE AND IMAGE UTILITIES
# ============================================================

def get_image_hash(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()


def clean_filename(filename):
    return os.path.basename(filename)


def extract_image_timestamp(filename):
    pattern = r"(?<!\d)(\d{8})-(\d{6})-(\d+)(?!\d)"
    match = re.search(pattern, filename)

    if not match:
        return None, None

    date_part = match.group(1)
    time_part = match.group(2)
    image_id = match.group(3)

    try:
        parsed = datetime.strptime(
            date_part + time_part,
            "%Y%m%d%H%M%S"
        ).replace(tzinfo=IST)

        return parsed, image_id

    except ValueError:
        return None, image_id


def extract_station_and_mast_from_image(image):
    """
    OCR attempts to find station and OHE mast information.

    OCR is not guaranteed to read every image correctly.
    Users can enter missing details manually.
    """

    raw_text = ""

    try:
        rgb = np.array(image.convert("RGB"))

        candidates = [
            rgb,
            cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        ]

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        enlarged = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        _, thresholded = cv2.threshold(
            enlarged,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        candidates.append(thresholded)

        texts = []

        for candidate in candidates:
            try:
                text = pytesseract.image_to_string(
                    candidate,
                    config="--psm 11"
                )

                if text.strip():
                    texts.append(text)

            except Exception:
                continue

        raw_text = "\n".join(texts)

    except Exception:
        pass

    station = None
    mast = None

    station_patterns = [
        r"station\s*(?:name)?\s*[:\-]\s*([A-Za-z0-9 ._-]+)",
        r"stn\s*[:\-]\s*([A-Za-z0-9 ._-]+)"
    ]

    mast_patterns = [
        r"(?:OHE\s*)?mast\s*(?:no\.?|number)?\s*[:\-]?\s*([A-Za-z0-9/-]+)",
        r"mast\s*id\s*[:\-]\s*([A-Za-z0-9/-]+)"
    ]

    for pattern in station_patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)

        if match:
            station = match.group(1).strip()
            break

    for pattern in mast_patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)

        if match:
            mast = match.group(1).strip()
            break

    return station, mast, raw_text


def get_analysis_region(image):
    rgb = np.array(image.convert("RGB"))
    height, width = rgb.shape[:2]

    x1 = int(width * 0.03)
    x2 = int(width * 0.88)
    y1 = int(height * 0.08)
    y2 = int(height * 0.92)

    if x2 <= x1 or y2 <= y1:
        return rgb

    return rgb[y1:y2, x1:x2]


# ============================================================
# TEMPERATURE FEATURE EXTRACTION
# ============================================================

def extract_features(image):
    region = get_analysis_region(image)

    hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(region, cv2.COLOR_RGB2LAB)

    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)

    mask = (v > 25) & (s > 20)

    if np.count_nonzero(mask) < 20:
        mask = np.ones(v.shape, dtype=bool)

    features = [
        np.mean(h[mask]),
        np.std(h[mask]),
        np.mean(s[mask]),
        np.std(s[mask]),
        np.mean(v[mask]),
        np.std(v[mask]),
        np.mean(l[mask]),
        np.std(l[mask]),
        np.mean(a[mask]),
        np.std(a[mask]),
        np.mean(b[mask]),
        np.std(b[mask]),
        np.percentile(v[mask], 10),
        np.percentile(v[mask], 25),
        np.percentile(v[mask], 50),
        np.percentile(v[mask], 75),
        np.percentile(v[mask], 90),
    ]

    return np.array(features, dtype=float)


# ============================================================
# CALIBRATION MODEL
# ============================================================

def train_calibration_model():
    """
    Train only when calibration_images contains images whose
    filenames correspond to the OEM_DATA keys.

    Otherwise, return None and use a clearly labelled fallback.
    """

    calibration_dir = "calibration_images"

    if not os.path.isdir(calibration_dir):
        return None

    feature_rows = []
    targets = []

    for filename in os.listdir(calibration_dir):
        path = os.path.join(calibration_dir, filename)

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            continue

        stem = os.path.splitext(filename)[0]

        if stem not in OEM_DATA:
            continue

        try:
            image = Image.open(path).convert("RGB")
            feature_rows.append(extract_features(image))

            targets.append([
                OEM_DATA[stem]["min"],
                OEM_DATA[stem]["max"]
            ])

        except Exception:
            continue

    if len(feature_rows) < 4:
        return None

    X = np.array(feature_rows, dtype=float)
    Y = np.array(targets, dtype=float)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1

    X_scaled = (X - mean) / std

    X_design = np.column_stack([
        np.ones(X_scaled.shape[0]),
        X_scaled
    ])

    alpha = 10.0

    penalty = np.eye(X_design.shape[1]) * alpha
    penalty[0, 0] = 0

    try:
        coefficients = np.linalg.solve(
            X_design.T @ X_design + penalty,
            X_design.T @ Y
        )

        return {
            "mean": mean,
            "std": std,
            "coefficients": coefficients
        }

    except np.linalg.LinAlgError:
        return None


def predict_temperature(features, model):
    """
    Returns minimum and maximum estimates.

    These are not certified radiometric measurements.
    """

    if model is not None:
        scaled = (
            features - model["mean"]
        ) / model["std"]

        design = np.concatenate([[1.0], scaled])

        predictions = design @ model["coefficients"]

        minimum = float(predictions[0])
        maximum = float(predictions[1])

        if minimum > maximum:
            minimum, maximum = maximum, minimum

        minimum = float(np.clip(minimum, 0, 40))
        maximum = float(np.clip(maximum, 0, 40))

        return minimum, maximum, "Calibration model estimate"

    # Fallback when no valid calibration model exists.
    # These values are approximations, not actual measurements.
    v_mean = features[4]
    v_low = features[12]
    v_high = features[16]

    minimum = 7.2 + (v_low / 255.0) * (39.5 - 7.2)
    maximum = 7.2 + (v_high / 255.0) * (39.5 - 7.2)

    if minimum > maximum:
        minimum, maximum = maximum, minimum

    # Keep ordering and bounds consistent.
    minimum = float(np.clip(minimum, 0, 40))
    maximum = float(np.clip(maximum, minimum, 40))

    return minimum, maximum, "Unvalidated color-based estimate"


# ============================================================
# FAULT CLASSIFICATION
# ============================================================

def get_decision(delta):
    if delta is None:
        return (
            "NOT ANALYZED",
            "Temperature values are unavailable."
        )

    if delta < FAULT_THRESHOLD:
        return (
            "NO FAULT",
            "No threshold exceedance indicated by this analysis."
        )

    return (
        "FAULT",
        "Requires inspection and attention within 2 days "
        "under the configured rule."
    )


# ============================================================
# DATABASE SAVE AND RETRIEVAL
# ============================================================

def save_inspection(
    image_filename,
    image_id,
    image_datetime,
    image_bytes,
    minimum_temperature,
    maximum_temperature,
    mean_temperature,
    fault_status,
    recommendation,
    temperature_source,
    station_name,
    ohe_mast_number
):
    conn = get_database_connection()

    try:
        image_sha256 = get_image_hash(image_bytes)

        if (
            minimum_temperature is not None
            and maximum_temperature is not None
        ):
            temperature_difference = (
                maximum_temperature - minimum_temperature
            )
        else:
            temperature_difference = None

        uploaded_at = datetime.now(IST)

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO thermal_inspections (
                    image_filename,
                    image_id,
                    image_datetime,
                    uploaded_at,
                    minimum_temperature,
                    maximum_temperature,
                    temperature_difference,
                    mean_temperature,
                    fault_status,
                    recommendation,
                    temperature_source,
                    image_sha256,
                    image_data,
                    station_name,
                    ohe_mast_number
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
                RETURNING id;
                """,
                (
                    image_filename,
                    image_id,
                    image_datetime,
                    uploaded_at,
                    minimum_temperature,
                    maximum_temperature,
                    temperature_difference,
                    mean_temperature,
                    fault_status,
                    recommendation,
                    temperature_source,
                    image_sha256,
                    psycopg2.Binary(image_bytes),
                    station_name,
                    ohe_mast_number
                )
            )

            record_id = cur.fetchone()[0]

        conn.commit()
        return record_id

    finally:
        conn.close()


def load_inspections(limit=1000):
    conn = get_database_connection()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    image_filename,
                    image_id,
                    image_datetime,
                    uploaded_at,
                    minimum_temperature,
                    maximum_temperature,
                    temperature_difference,
                    mean_temperature,
                    fault_status,
                    recommendation,
                    temperature_source,
                    station_name,
                    ohe_mast_number,
                    image_sha256
                FROM thermal_inspections
                ORDER BY uploaded_at DESC
                LIMIT %s;
                """,
                (int(limit),)
            )

            rows = cur.fetchall()

        return pd.DataFrame([dict(row) for row in rows])

    finally:
        conn.close()


def load_inspection_image(record_id):
    conn = get_database_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT image_filename, image_data
                FROM thermal_inspections
                WHERE id = %s;
                """,
                (int(record_id),)
            )

            result = cur.fetchone()

        return result

    finally:
        conn.close()


def load_all_export_records():
    conn = get_database_connection()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    image_filename,
                    image_id,
                    image_datetime,
                    uploaded_at,
                    minimum_temperature,
                    maximum_temperature,
                    temperature_difference,
                    mean_temperature,
                    fault_status,
                    recommendation,
                    temperature_source,
                    station_name,
                    ohe_mast_number,
                    image_sha256,
                    OCTET_LENGTH(image_data) AS image_size_bytes
                FROM thermal_inspections
                ORDER BY uploaded_at DESC;
                """
            )

            rows = cur.fetchall()

        return pd.DataFrame([dict(row) for row in rows])

    finally:
        conn.close()


# ============================================================
# EXPORT UTILITIES
# ============================================================

def dataframe_to_excel(dataframe, sheet_name="Inspections"):
    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:
        dataframe.to_excel(
            writer,
            index=False,
            sheet_name=sheet_name[:31]
        )

    output.seek(0)
    return output.getvalue()


# ============================================================
# SHARED UI COMPONENTS
# ============================================================

def render_hero(title, subtitle, eyebrow="KRCL • THERMAL INTELLIGENCE"):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_metric(label, value, subtitle=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_heading(title, caption=None):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True
    )

    if caption:
        st.markdown(
            f'<div class="section-caption">{caption}</div>',
            unsafe_allow_html=True
        )


def get_temperature_source_for_known_image(image_id):
    if image_id and image_id in OEM_DATA:
        return "OEM reference"

    return None


# ============================================================
# APPLICATION STARTUP
# ============================================================

st.sidebar.markdown(
    """
    <div style="padding: 12px 4px 22px 4px;">
        <div style="font-size: 12px; color: #56e0e8;
                    font-weight: 800; letter-spacing: 2px;">
            KONKAN RAILWAY
        </div>
        <div style="font-size: 24px; font-weight: 800;
                    color: #ffffff; margin-top: 6px;">
            THERMAL<span style="color:#56e0e8;">.AI</span>
        </div>
        <div style="font-size: 11px; color: #91a5bf;
                    margin-top: 6px;">
            IT Department · Inspection Platform
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Thermal Analysis Studio",
        "Fault Monitor",
        "Inspection History",
        "Analytics",
        "Export Center",
        "OEM Validation",
        "System Status"
    ],
    index=0
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div style="background:#111d30; border:1px solid #263952;
                padding:14px; border-radius:12px;">
        <div style="font-size:11px; color:#91a5bf;">
            FAULT THRESHOLD
        </div>
        <div style="font-size:25px; color:#56e0e8;
                    font-weight:800; margin-top:4px;">
            5.0 °C
        </div>
        <div style="font-size:11px; color:#91a5bf;">
            Configured temperature difference
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div style="font-size:11px; color:#71859f;
                margin-top:22px; line-height:1.8;">
        KRCL Thermal Track Analyzer<br>
        Internal inspection support tool<br>
        All times shown in IST
    </div>
    """,
    unsafe_allow_html=True
)

db_ready = False
db_error = None

try:
    initialize_database()
    db_ready = True

except Exception as e:
    db_error = str(e)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    render_hero(
        "Thermal Intelligence Dashboard",
        "A centralized workspace for thermal inspection records, "
        "fault monitoring and infrastructure review."
    )

    if not db_ready:
        st.error(
            "Database connection unavailable. Check DATABASE_URL "
            "in Streamlit Secrets."
        )

        with st.expander("Technical details"):
            st.code(db_error or "Unknown database error")

        st.stop()

    try:
        history = load_inspections(limit=5000)

    except Exception as e:
        st.error("Could not load dashboard records.")
        st.code(str(e))
        st.stop()

    if history.empty:
        total = 0
        faults = 0
        no_faults = 0
        pending = 0
        recent = history

    else:
        total = len(history)

        faults = int(
            (history["fault_status"] == "FAULT").sum()
        )

        no_faults = int(
            (history["fault_status"] == "NO FAULT").sum()
        )

        pending = int(
            (history["fault_status"] == "NOT ANALYZED").sum()
        )

        recent = history.head(8)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_metric(
            "TOTAL INSPECTIONS",
            f"{total:,}",
            "Saved database records"
        )

    with c2:
        render_metric(
            "FAULT RECORDS",
            f"{faults:,}",
            "Require engineering review"
        )

    with c3:
        render_metric(
            "NO-FAULT RECORDS",
            f"{no_faults:,}",
            "Below configured threshold"
        )

    with c4:
        render_metric(
            "NOT ANALYZED",
            f"{pending:,}",
            "Missing temperature result"
        )

    section_heading(
        "Inspection Overview",
        "A quick view of saved inspection activity."
    )

    left, right = st.columns([1.15, 0.85])

    with left:
        if not history.empty:
            chart_data = pd.DataFrame({
                "Status": [
                    "FAULT",
                    "NO FAULT",
                    "NOT ANALYZED"
                ],
                "Records": [
                    faults,
                    no_faults,
                    pending
                ]
            })

            st.bar_chart(
                chart_data.set_index("Status"),
                height=280
            )

        else:
            st.info(
                "No inspection records yet. Start by uploading "
                "a thermal image in Thermal Analysis Studio."
            )

    with right:
        st.markdown(
            """
            <div class="info-card">
                <h4>⚡ Quick Actions</h4>
                <p>Upload and process a thermal image.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "Open Thermal Analysis Studio",
            type="primary",
            use_container_width=True
        ):
            st.session_state["requested_page"] = (
                "Thermal Analysis Studio"
            )
            st.rerun()

        if st.button(
            "View Inspection History",
            use_container_width=True
        ):
            st.session_state["requested_page"] = (
                "Inspection History"
            )
            st.rerun()

    section_heading(
        "Latest Inspections",
        "The eight most recently saved records."
    )

    if not recent.empty:
        display_columns = [
            "id",
            "image_filename",
            "station_name",
            "ohe_mast_number",
            "minimum_temperature",
            "maximum_temperature",
            "temperature_difference",
            "fault_status",
            "uploaded_at"
        ]

        available = [
            col for col in display_columns
            if col in recent.columns
        ]

        st.dataframe(
            recent[available],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No inspections have been saved.")


# ============================================================
# THERMAL ANALYSIS STUDIO
# ============================================================

elif page == "Thermal Analysis Studio":

    render_hero(
        "Thermal Analysis Studio",
        "Upload inspection images, capture OHE location details "
        "and generate an inspection record."
    )

    st.markdown(
        """
        <div class="info-card">
            <h4>Before you analyse</h4>
            <p>
                Temperature estimates derived from ordinary image
                colours are not equivalent to radiometric thermal
                measurements. Validate results against approved
                equipment and engineering procedures before using
                them for operational decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Upload thermal inspection images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        help="You can upload multiple inspection images."
    )

    if uploaded_files:
        section_heading(
            "Inspection Location",
            "Enter the location information for this batch."
        )

        col1, col2 = st.columns(2)

        with col1:
            station_name_input = st.text_input(
                "Station name",
                placeholder="e.g. Ratnagiri"
            )

        with col2:
            mast_number_input = st.text_input(
                "OHE mast number",
                placeholder="e.g. 124/7"
            )

        analyze_button = st.button(
            "Analyse Images & Save Records",
            type="primary",
            use_container_width=True
        )

        if analyze_button:

            if not db_ready:
                st.error(
                    "Database unavailable. Configure DATABASE_URL "
                    "before analysing images."
                )

            else:
                model = train_calibration_model()

                progress = st.progress(0)
                status_text = st.empty()

                batch_results = []
                total_files = len(uploaded_files)

                for index, uploaded_file in enumerate(
                    uploaded_files
                ):

                    status_text.write(
                        f"Processing {index + 1} of "
                        f"{total_files}: {uploaded_file.name}"
                    )

                    image_bytes = uploaded_file.getvalue()
                    filename = clean_filename(
                        uploaded_file.name
                    )

                    try:
                        image = Image.open(
                            io.BytesIO(image_bytes)
                        ).convert("RGB")

                        image_datetime, image_id = (
                            extract_image_timestamp(filename)
                        )

                        ocr_station, ocr_mast, ocr_text = (
                            extract_station_and_mast_from_image(
                                image
                            )
                        )

                        station_name = (
                            station_name_input.strip()
                            or ocr_station
                            or "No record found"
                        )

                        mast_number = (
                            mast_number_input.strip()
                            or ocr_mast
                            or "No record found"
                        )

                        oem = (
                            OEM_DATA.get(image_id)
                            if image_id else None
                        )

                        if oem:
                            minimum_temperature = float(
                                oem["min"]
                            )

                            maximum_temperature = float(
                                oem["max"]
                            )

                            temperature_source = (
                                "OEM reference data"
                            )

                        else:
                            features = extract_features(image)

                            (
                                minimum_temperature,
                                maximum_temperature,
                                temperature_source
                            ) = predict_temperature(
                                features,
                                model
                            )

                        difference = (
                            maximum_temperature
                            - minimum_temperature
                        )

                        mean_temperature = (
                            maximum_temperature
                            + minimum_temperature
                        ) / 2.0

                        fault_status, recommendation = (
                            get_decision(difference)
                        )

                        record_id = save_inspection(
                            image_filename=filename,
                            image_id=image_id,
                            image_datetime=image_datetime,
                            image_bytes=image_bytes,
                            minimum_temperature=minimum_temperature,
                            maximum_temperature=maximum_temperature,
                            mean_temperature=mean_temperature,
                            fault_status=fault_status,
                            recommendation=recommendation,
                            temperature_source=temperature_source,
                            station_name=station_name,
                            ohe_mast_number=mast_number
                        )

                        batch_results.append({
                            "Record ID": record_id,
                            "Filename": filename,
                            "Image ID": image_id,
                            "Station": station_name,
                            "OHE Mast": mast_number,
                            "Minimum °C": minimum_temperature,
                            "Maximum °C": maximum_temperature,
                            "Difference °C": difference,
                            "Status": fault_status,
                            "Temperature Source": temperature_source,
                            "Recommendation": recommendation
                        })

                    except Exception as e:
                        batch_results.append({
                            "Record ID": None,
                            "Filename": filename,
                            "Image ID": None,
                            "Station": station_name_input,
                            "OHE Mast": mast_number_input,
                            "Minimum °C": None,
                            "Maximum °C": None,
                            "Difference °C": None,
                            "Status": "NOT ANALYZED",
                            "Temperature Source": "Error",
                            "Recommendation": str(e)
                        })

                    progress.progress(
                        (index + 1) / total_files
                    )

                status_text.empty()

                st.session_state["batch_results"] = batch_results

                st.success(
                    "Batch processing completed. Successfully "
                    "saved records are available in Inspection History."
                )

                st.rerun()

    if "batch_results" in st.session_state:
        section_heading(
            "Batch Results",
            "Results from the most recent processing run."
        )

        results_df = pd.DataFrame(
            st.session_state["batch_results"]
        )

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        excel_data = dataframe_to_excel(
            results_df,
            "Batch Results"
        )

        st.download_button(
            "Download Batch Results (Excel)",
            data=excel_data,
            file_name="KRCL_Thermal_Batch_Results.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            type="primary"
        )


# ============================================================
# FAULT MONITOR
# ============================================================

elif page == "Fault Monitor":

    render_hero(
        "Fault Monitor",
        "Review saved records classified by the configured "
        "temperature-difference threshold."
    )

    if not db_ready:
        st.error("Database unavailable.")
        st.stop()

    try:
        history = load_inspections(limit=5000)

    except Exception as e:
        st.error("Could not load fault records.")
        st.code(str(e))
        st.stop()

    if history.empty:
        st.info("No inspection records are available.")
        st.stop()

    faults = history[
        history["fault_status"] == "FAULT"
    ].copy()

    no_faults = history[
        history["fault_status"] == "NO FAULT"
    ].copy()

    pending = history[
        history["fault_status"] == "NOT ANALYZED"
    ].copy()

    a, b, c = st.columns(3)

    with a:
        render_metric(
            "FAULT RECORDS",
            len(faults),
            "Configured threshold exceeded"
        )

    with b:
        render_metric(
            "NO FAULT",
            len(no_faults),
            "Below configured threshold"
        )

    with c:
        render_metric(
            "NOT ANALYZED",
            len(pending),
            "Needs a usable analysis result"
        )

    section_heading("Fault Records")

    if faults.empty:
        st.success("No saved records currently carry FAULT status.")
    else:
        st.dataframe(
            faults,
            use_container_width=True,
            hide_index=True
        )

        fault_export = dataframe_to_excel(
            faults,
            "Fault Records"
        )

        st.download_button(
            "Download Fault Records",
            data=fault_export,
            file_name="KRCL_Fault_Records.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            )
        )

    with st.expander("View NO FAULT records"):
        st.dataframe(
            no_faults,
            use_container_width=True,
            hide_index=True
        )

    with st.expander("View NOT ANALYZED records"):
        st.dataframe(
            pending,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# INSPECTION HISTORY
# ============================================================

elif page == "Inspection History":

    render_hero(
        "Inspection History",
        "Search saved inspection records and retrieve original images."
    )

    if not db_ready:
        st.error("Database unavailable.")
        st.stop()

    try:
        history = load_inspections(limit=5000)

    except Exception as e:
        st.error("Could not load inspection history.")
        st.code(str(e))
        st.stop()

    if history.empty:
        st.info("No inspection history is available yet.")
        st.stop()

    section_heading("Search & Filter")

    f1, f2, f3 = st.columns(3)

    with f1:
        search_text = st.text_input(
            "Search filename, station or mast"
        )

    with f2:
        statuses = sorted(
            history["fault_status"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_statuses = st.multiselect(
            "Filter by status",
            statuses,
            default=statuses
        )

    with f3:
        station_values = sorted(
            history["station_name"]
            .fillna("No record found")
            .astype(str)
            .unique()
            .tolist()
        )

        selected_stations = st.multiselect(
            "Filter by station",
            station_values,
            default=station_values
        )

    filtered = history[
        history["fault_status"].isin(selected_statuses)
        & history["station_name"]
        .fillna("No record found")
        .astype(str)
        .isin(selected_stations)
    ].copy()

    if search_text.strip():
        searchable = (
            filtered["image_filename"].fillna("").astype(str)
            + " "
            + filtered["station_name"].fillna("").astype(str)
            + " "
            + filtered["ohe_mast_number"].fillna("").astype(str)
            + " "
            + filtered["image_id"].fillna("").astype(str)
        )

        filtered = filtered[
            searchable.str.contains(
                search_text.strip(),
                case=False,
                regex=False
            )
        ]

    st.markdown(
        f"**{len(filtered):,}** matching inspection records"
    )

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    section_heading(
        "Retrieve Saved Image",
        "Enter a database record ID to display its original image."
    )

    record_id = st.number_input(
        "Inspection record ID",
        min_value=1,
        step=1
    )

    if st.button("Retrieve Image", type="primary"):

        try:
            result = load_inspection_image(record_id)

            if result:
                image_filename, image_bytes = result

                st.success(
                    f"Retrieved image: {image_filename}"
                )

                retrieved_image = Image.open(
                    io.BytesIO(bytes(image_bytes))
                )

                st.image(
                    retrieved_image,
                    caption=image_filename,
                    use_container_width=True
                )

                st.download_button(
                    "Download Original Image",
                    data=bytes(image_bytes),
                    file_name=image_filename,
                    mime="application/octet-stream"
                )

            else:
                st.warning(
                    "No image was found for that record ID."
                )

        except Exception as e:
            st.error("Could not retrieve the image.")
            st.code(str(e))


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    render_hero(
        "Inspection Analytics",
        "Explore recorded temperature differences, fault counts "
        "and inspection trends."
    )

    if not db_ready:
        st.error("Database unavailable.")
        st.stop()

    try:
        history = load_inspections(limit=5000)

    except Exception as e:
        st.error("Could not load analytics.")
        st.code(str(e))
        st.stop()

    if history.empty:
        st.info("Save inspection records to start seeing analytics.")
        st.stop()

    valid = history[
        history["temperature_difference"].notna()
    ].copy()

    if not valid.empty:
        avg_delta = valid["temperature_difference"].mean()
        max_delta = valid["temperature_difference"].max()
        avg_max = valid["maximum_temperature"].mean()

    else:
        avg_delta = None
        max_delta = None
        avg_max = None

    c1, c2, c3 = st.columns(3)

    with c1:
        render_metric(
            "AVERAGE DIFFERENCE",
            f"{avg_delta:.2f} °C" if avg_delta is not None else "N/A",
            "Across records with temperature values"
        )

    with c2:
        render_metric(
            "HIGHEST DIFFERENCE",
            f"{max_delta:.2f} °C" if max_delta is not None else "N/A",
            "Largest saved difference"
        )

    with c3:
        render_metric(
            "AVERAGE MAX TEMP",
            f"{avg_max:.2f} °C" if avg_max is not None else "N/A",
            "Across records with temperature values"
        )

    section_heading("Fault Status Distribution")

    status_counts = (
        history["fault_status"]
        .value_counts()
        .rename_axis("Status")
        .to_frame("Records")
    )

    st.bar_chart(status_counts)

    section_heading("Temperature Difference Distribution")

    if not valid.empty:
        chart_df = valid[
            ["id", "temperature_difference"]
        ].copy()

        chart_df["id"] = chart_df["id"].astype(str)

        st.line_chart(
            chart_df.set_index("id"),
            height=300
        )

        st.caption(
            "The chart shows stored values, not independent "
            "validation of their accuracy."
        )

    else:
        st.info("No usable temperature differences are available.")

    section_heading("Station Summary")

    if "station_name" in history.columns:
        station_summary = (
            history.groupby("station_name", dropna=False)
            .agg(
                Inspections=("id", "count"),
                Faults=(
                    "fault_status",
                    lambda s: int((s == "FAULT").sum())
                ),
                Average_Difference=(
                    "temperature_difference",
                    "mean"
                )
            )
            .reset_index()
        )

        st.dataframe(
            station_summary,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# EXPORT CENTER
# ============================================================

elif page == "Export Center":

    render_hero(
        "Export Center",
        "Download inspection registers and reporting datasets."
    )

    if not db_ready:
        st.error("Database unavailable.")
        st.stop()

    try:
        export_df = load_all_export_records()

    except Exception as e:
        st.error("Could not load export records.")
        st.code(str(e))
        st.stop()

    if export_df.empty:
        st.info("There are no records available to export.")
        st.stop()

    st.markdown(
        f"### {len(export_df):,} saved records available"
    )

    section_heading("Excel Export")

    excel_df = export_df.copy()

    excel_bytes = dataframe_to_excel(
        excel_df,
        "Inspection Register"
    )

    st.download_button(
        "Download Complete Inspection Register (Excel)",
        data=excel_bytes,
        file_name="KRCL_Inspection_Register.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        type="primary",
        use_container_width=True
    )

    section_heading("CSV Export")

    csv_bytes = export_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "Download Inspection Register (CSV)",
        data=csv_bytes,
        file_name="KRCL_Inspection_Register.csv",
        mime="text/csv",
        use_container_width=True
    )

    section_heading("Fault-Only Export")

    fault_df = export_df[
        export_df["fault_status"] == "FAULT"
    ].copy()

    if fault_df.empty:
        st.info("No FAULT records are currently saved.")

    else:
        st.download_button(
            "Download Fault Register (Excel)",
            data=dataframe_to_excel(
                fault_df,
                "Fault Register"
            ),
            file_name="KRCL_Fault_Register.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True
        )

    with st.expander("Preview export dataset"):
        st.dataframe(
            export_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# OEM VALIDATION
# ============================================================

elif page == "OEM Validation":

    render_hero(
        "OEM Reference Validation",
        "Review the reference dataset currently configured "
        "inside the application."
    )

    oem_df = pd.DataFrame([
        {
            "Image ID": image_id,
            "OEM Minimum °C": values["min"],
            "OEM Maximum °C": values["max"],
            "OEM Difference °C": (
                values["max"] - values["min"]
            )
        }
        for image_id, values in OEM_DATA.items()
    ])

    c1, c2, c3 = st.columns(3)

    with c1:
        render_metric(
            "REFERENCE IMAGES",
            len(oem_df),
            "Configured OEM reference entries"
        )

    with c2:
        render_metric(
            "MINIMUM REFERENCE",
            f"{oem_df['OEM Minimum °C'].min():.1f} °C",
            "Lowest configured OEM minimum"
        )

    with c3:
        render_metric(
            "MAXIMUM REFERENCE",
            f"{oem_df['OEM Maximum °C'].max():.1f} °C",
            "Highest configured OEM maximum"
        )

    section_heading("OEM Reference Dataset")

    st.dataframe(
        oem_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download OEM Reference Dataset",
        data=dataframe_to_excel(
            oem_df,
            "OEM References"
        ),
        file_name="KRCL_OEM_Reference_Data.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )

    section_heading("Calibration Image Directory")

    if os.path.isdir("calibration_images"):
        calibration_files = [
            f for f in os.listdir("calibration_images")
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            )
        ]

        st.write(
            f"Calibration image files found: "
            f"**{len(calibration_files)}**"
        )

        if calibration_files:
            st.dataframe(
                pd.DataFrame({
                    "Calibration Image": calibration_files
                }),
                use_container_width=True,
                hide_index=True
            )

    else:
        st.warning(
            "The calibration_images directory was not found. "
            "The calibrated model will not be trained from "
            "local calibration images."
        )

    st.markdown(
        """
        <div class="info-card">
            <h4>Validation note</h4>
            <p>
                An OEM reference table is not, by itself, a complete
                calibration dataset. Reliable validation requires
                representative images, trusted temperature labels,
                appropriate calibration and independent testing.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SYSTEM STATUS
# ============================================================

elif page == "System Status":

    render_hero(
        "System Status",
        "Review the runtime environment and connectivity checks."
    )

    section_heading("Application Environment")

    runtime = [
        {
            "Component": "Python",
            "Status": "Available",
            "Details": "Application runtime"
        },
        {
            "Component": "OpenCV",
            "Status": "Available",
            "Details": cv2.__version__
        },
        {
            "Component": "NumPy",
            "Status": "Available",
            "Details": np.__version__
        },
        {
            "Component": "Pandas",
            "Status": "Available",
            "Details": pd.__version__
        },
        {
            "Component": "Tesseract OCR",
            "Status": (
                "Available"
                if os.path.exists("/usr/bin/tesseract")
                else "Check configuration"
            ),
            "Details": "Used for station/mast text extraction"
        },
        {
            "Component": "Database",
            "Status": (
                "Connected" if db_ready else "Unavailable"
            ),
            "Details": (
                "PostgreSQL/Supabase"
                if db_ready
                else "Check DATABASE_URL in Streamlit Secrets"
            )
        }
    ]

    st.dataframe(
        pd.DataFrame(runtime),
        use_container_width=True,
        hide_index=True
    )

    if db_ready:
        st.success("Database connection successful.")

        if st.button("Test Database Read", type="primary"):
            try:
                history = load_inspections(limit=1)

                st.success(
                    "Database read successful. "
                    f"Records returned: {len(history)}"
                )

            except Exception as e:
                st.error("Database read failed.")
                st.code(str(e))

    else:
        st.error("Database connection failed.")

        with st.expander("Database error details"):
            st.code(db_error or "Unknown error")

    section_heading("Current Configuration")

    config_df = pd.DataFrame([
        {
            "Setting": "Fault threshold",
            "Value": f"{FAULT_THRESHOLD:.1f} °C"
        },
        {
            "Setting": "Timezone",
            "Value": "Asia/Kolkata"
        },
        {
            "Setting": "OEM reference entries",
            "Value": str(len(OEM_DATA))
        },
        {
            "Setting": "Calibration directory",
            "Value": "calibration_images/"
        }
    ])

    st.dataframe(
        config_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-note">
        KRCL THERMAL TRACK ANALYZER · IT DEPARTMENT<br>
        Inspection support platform · Times displayed in IST<br>
        Temperature estimates require appropriate technical validation.
    </div>
    """,
    unsafe_allow_html=True
)
