import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import io
import os
import glob
import re
import hashlib
import html
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
NO_RECORD = "No record found"


# ============================================================
# OEM REFERENCE DATA (Aligned with O&M Software: Max first, Min second)
# ============================================================

OEM_DATA = {
    "20250315-133514-016": {"max": 39.5, "min": 36.1},
    "20251111-112915-003": {"max": 31.5, "min": 22.0},
    "20251111-114541-001": {"max": 35.6, "min": 7.2},
    "20251111-114605-002": {"max": 35.4, "min": 27.4},
    "20251113-074547-002": {"max": 17.6, "min": 13.7},
    "20251113-091808-012": {"max": 24.7, "min": 19.3},
    "20251113-092707-005": {"max": 24.4, "min": 20.5},
    "20251113-093327-012": {"max": 24.0, "min": 18.3},
    "20251113-113151-016": {"max": 34.5, "min": 24.7},
    "20251114-101128-007": {"max": 26.2, "min": 15.5},
    "20251114-114841-018": {"max": 35.0, "min": 13.6},
    "20251117-124611-004": {"max": 33.8, "min": 22.8},
    "IRI_20261111_081623": {"max": 19.7, "min": 15.2},
    "IRI_20261111_083948": {"max": 24.7, "min": 20.2},
    "IRI_20261111_081116": {"max": 19.2, "min": 17.0},
    "IRI_20261111_075725": {"max": 15.3, "min": 13.7},
    "IRI_20261107_122532": {"max": 38.5, "min": 36.7},
}


# ============================================================
# LOAD OHE GPS CSV FROM GITHUB 'data/' OR ROOT FOLDER
# ============================================================

@st.cache_data
def load_ohe_master_data():
    """
    Loads Station/Section and OHE Mast mappings from the GPS CSV
    located in the GitHub `data/` folder or root folder.
    """
    candidate_paths = [
        "data/gps_reference.csv",
        "gps_reference.csv",
        "data/Oliver G13052026GPS(1) (1).csv",
        "data/Oliver G13052026GPS(1).csv",
        "data/Oliver G13052026GPS.csv",
        "Oliver G13052026GPS(1) (1).csv",
        "Oliver G13052026GPS(1).csv",
        "Oliver G13052026GPS.csv",
    ]

    # Also scan data/*.csv and ./*.csv automatically
    for folder in ("data", "."):
        if os.path.isdir(folder):
            for csv_file in sorted(glob.glob(os.path.join(folder, "*.csv"))):
                if csv_file not in candidate_paths:
                    candidate_paths.append(csv_file)

    csv_path = None
    for path in candidate_paths:
        if os.path.isfile(path):
            csv_path = path
            break

    section_to_masts = {}
    all_sections = []
    all_masts = []

    if csv_path:
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            df.columns = [c.strip() for c in df.columns]

            if "Section" in df.columns and "OHE Mast" in df.columns:
                sec_series = (
                    df["Section"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .str.replace(r"\s+", " ", regex=True)
                )
                mast_series = (
                    df["OHE Mast"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )

                valid_df = pd.DataFrame({
                    "Section": sec_series,
                    "OHE_Mast": mast_series
                })
                valid_df = valid_df[valid_df["OHE_Mast"] != ""]

                all_sections = sorted(
                    [s for s in valid_df["Section"].unique().tolist() if s]
                )

                # Preserve natural appearance order for masts while removing duplicates
                all_masts = list(dict.fromkeys(valid_df["OHE_Mast"].tolist()))

                for sec, group in valid_df.groupby("Section"):
                    if sec:
                        section_to_masts[sec] = list(
                            dict.fromkeys(group["OHE_Mast"].tolist())
                        )

        except Exception:
            pass

    return {
        "csv_path": csv_path,
        "sections": all_sections,
        "all_masts": all_masts,
        "section_to_masts": section_to_masts
    }


# ============================================================
# PREMIUM DARK UI — LIGHT UPLOAD BOX & READABLE DROPDOWNS
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
            180deg, #101c30 0%, #0b1424 100%
        );
        border-right: 1px solid #24354d;
    }

    [data-testid="stSidebar"] * {
        color: #e8eef8;
    }

    .hero {
        background: linear-gradient(
            115deg, #12243c 0%, #123d59 55%, #087e8b 100%
        );
        border: 1px solid #24536b;
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 24px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.18);
    }

    .hero h1 {
        font-size: 32px;
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

    .status-badge-fault {
        display: inline-block;
        background: rgba(220, 38, 38, 0.22);
        color: #ef4444;
        border: 1px solid #dc2626;
        border-radius: 8px;
        padding: 4px 12px;
        font-weight: 800;
        font-size: 13px;
    }

    .status-badge-ok {
        display: inline-block;
        background: rgba(22, 163, 74, 0.22);
        color: #22c55e;
        border: 1px solid #16a34a;
        border-radius: 8px;
        padding: 4px 12px;
        font-weight: 800;
        font-size: 13px;
    }

    .status-badge-pending {
        display: inline-block;
        background: rgba(245, 158, 11, 0.18);
        color: #fbbf24;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 4px 12px;
        font-weight: 800;
        font-size: 13px;
    }

    /* ========================================================
       1. DARK LABELS FOR STATION / SECTION & OHE MAST NUMBER
       ======================================================== */
    [data-testid="stExpander"] {
        background-color: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 14px !important;
        margin-bottom: 14px !important;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary * {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    [data-testid="stExpander"] [data-testid="stWidgetLabel"],
    [data-testid="stExpander"] [data-testid="stWidgetLabel"] p,
    [data-testid="stExpander"] label,
    [data-testid="stExpander"] label p,
    .dark-field-label {
        color: #0f172a !important;
        background-color: #e2e8f0 !important;
        padding: 5px 12px !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        display: inline-block !important;
        margin-bottom: 6px !important;
        border: 1px solid #cbd5e1 !important;
    }

    [data-testid="stExpander"] [data-testid="stImageCaption"] {
        color: #1e293b !important;
        font-weight: 600 !important;
    }

    /* Selectbox / Dropdown visibility */
    div[data-baseweb="select"] > div {
        background-color: #142238 !important;
        color: #ffffff !important;
        border: 1px solid #2e8c9d !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input {
        color: #ffffff !important;
    }

    /* ========================================================
       2. ANALYZED IMAGES & RESULTS — MAX / MIN / DIFF VISIBILITY
       ======================================================== */
    .temp-stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin: 12px 0 16px 0;
    }

    .temp-stat-box {
        border-radius: 12px;
        padding: 14px 16px;
        border: 1px solid;
    }

    .temp-stat-max {
        background: #fff1f2;
        border-color: #fecdd3;
    }

    .temp-stat-max .temp-stat-label {
        color: #9f1239;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .temp-stat-max .temp-stat-value {
        color: #be123c;
        font-size: 24px;
        font-weight: 800;
        margin-top: 4px;
    }

    .temp-stat-min {
        background: #f0f9ff;
        border-color: #bae6fd;
    }

    .temp-stat-min .temp-stat-label {
        color: #075985;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .temp-stat-min .temp-stat-value {
        color: #0369a1;
        font-size: 24px;
        font-weight: 800;
        margin-top: 4px;
    }

    .temp-stat-diff {
        background: #fffbeb;
        border-color: #fde68a;
    }

    .temp-stat-diff .temp-stat-label {
        color: #92400e;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .temp-stat-diff .temp-stat-value {
        color: #b45309;
        font-size: 24px;
        font-weight: 800;
        margin-top: 4px;
    }

    /* Fallback for standard st.metric widgets */
    [data-testid="stMetric"] {
        background-color: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
    }

    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: #0369a1 !important;
        font-weight: 800 !important;
    }

    /* ========================================================
       4. BUTTONS — PRIMARY TEAL STYLE
       ======================================================== */
    .stButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 700;
        border: 1px solid #2e8c9d;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #087e8b, #11a6b5) !important;
        color: #ffffff !important;
        border: none !important;
    }

    /* ========================================================
       5. STYLED TABLES — DISTINCT HEADER COLOR & FAULT STATUS
       ======================================================== */
    .krcl-table-wrapper {
        overflow-x: auto;
        border-radius: 12px;
        border: 1px solid #263952;
        margin-bottom: 18px;
        background: #0f192a;
    }

    table.krcl-styled-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13.5px;
        color: #e8eef8;
    }

    table.krcl-styled-table thead tr {
        background: linear-gradient(90deg, #087e8b 0%, #123d59 100%);
    }

    table.krcl-styled-table thead th {
        color: #fde047 !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 12px;
        padding: 12px 14px;
        text-align: left;
        border-bottom: 2px solid #2e8c9d;
        white-space: nowrap;
    }

    table.krcl-styled-table tbody tr {
        border-bottom: 1px solid #1e2f48;
        transition: background 0.15s ease;
    }

    table.krcl-styled-table tbody tr:nth-child(even) {
        background-color: #132036;
    }

    table.krcl-styled-table tbody tr:hover {
        background-color: #1a2c47;
    }

    table.krcl-styled-table tbody td {
        padding: 10px 14px;
        color: #e2e8f0;
        vertical-align: middle;
    }

    .cell-fault {
        background-color: rgba(220, 38, 38, 0.22);
        color: #ff4d4d !important;
        border: 1px solid #dc2626;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 800;
        display: inline-block;
        font-size: 12px;
    }

    .cell-nofault {
        background-color: rgba(22, 163, 74, 0.22);
        color: #22c55e !important;
        border: 1px solid #16a34a;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 800;
        display: inline-block;
        font-size: 12px;
    }

    .cell-pending {
        background-color: rgba(245, 158, 11, 0.18);
        color: #fbbf24 !important;
        border: 1px solid #f59e0b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 800;
        display: inline-block;
        font-size: 12px;
    }

    /* ========================================================
       UPLOAD BOX — WHITE BACKGROUND, DARK READABLE TEXT
       ======================================================== */

    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 14px !important;
        padding: 12px !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #ffffff !important;
        border: 1px dashed #9ca3af !important;
        border-radius: 10px !important;
    }

    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzoneInstructions"] *,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] p {
        color: #1f2937 !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: 1px solid #1d4ed8 !important;
        border-radius: 8px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        padding: 8px 18px !important;
    }

    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] button * {
        color: #ffffff !important;
    }

    [data-testid="stFileUploaderFile"] {
        background-color: #f3f4f6 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploaderFile"] *,
    [data-testid="stFileUploaderFileName"] {
        color: #1f2937 !important;
    }

    [data-testid="stFileUploaderFile"] button,
    [data-testid="stFileUploaderFile"] button * {
        color: #374151 !important;
    }

    hr {
        border-color: #27364c;
    }

    #MainMenu, footer {
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
                    maximum_temperature DOUBLE PRECISION,
                    minimum_temperature DOUBLE PRECISION,
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
# IMAGE UTILITIES
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
    image_id = match.group(0)

    try:
        parsed = datetime.strptime(
            date_part + time_part,
            "%Y%m%d%H%M%S"
        ).replace(tzinfo=IST)

        return parsed, image_id

    except ValueError:
        return None, image_id


# ============================================================
# OCR: AUTOMATIC STATION, OHE MAST & SCALE TEMP EXTRACTION
# ============================================================

def normalize_ocr_text(text):
    text = text.replace("\r", "\n")
    text = text.replace("|", "I")
    return text


def extract_station_and_mast_from_image(image):
    text_candidates = []

    try:
        rgb = np.array(image.convert("RGB"))
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

        images_to_read = [
            rgb,
            gray,
            enlarged,
            thresholded
        ]

        for candidate in images_to_read:
            for psm in (6, 11):
                try:
                    result = pytesseract.image_to_string(
                        candidate,
                        config=f"--psm {psm}"
                    )

                    if result.strip():
                        text_candidates.append(result)

                except Exception:
                    continue

    except Exception:
        return None, None, ""

    raw_text = normalize_ocr_text(
        "\n".join(text_candidates)
    )

    lines = [
        line.strip()
        for line in raw_text.splitlines()
        if line.strip()
    ]

    station = None
    mast = None

    station_patterns = [
        r"\bstation\s*name\s*[:=\-]\s*(.+)",
        r"\bstation\s*[:=\-]\s*(.+)",
        r"\bstn\s*[:=\-]\s*(.+)",
        r"\blocation\s*[:=\-]\s*(.+)"
    ]

    mast_patterns = [
        r"\bOHE\s*mast\s*(?:no\.?|number|num|id)?\s*[:=\-]?\s*([A-Z0-9][A-Z0-9/\-]*)",
        r"\bmast\s*(?:no\.?|number|num|id)\s*[:=\-]?\s*([A-Z0-9][A-Z0-9/\-]*)",
        r"\bmast\s*[:=\-]\s*([A-Z0-9][A-Z0-9/\-]*)"
    ]

    for line in lines:
        if station is None:
            for pattern in station_patterns:
                match = re.search(
                    pattern,
                    line,
                    flags=re.IGNORECASE
                )

                if match:
                    value = match.group(1).strip()
                    value = re.sub(
                        r"\s+(?:OHE\s*)?mast.*$",
                        "",
                        value,
                        flags=re.IGNORECASE
                    )
                    value = value.strip(" :;,-")

                    if value and len(value) <= 80:
                        station = value
                        break

        if mast is None:
            for pattern in mast_patterns:
                match = re.search(
                    pattern,
                    line,
                    flags=re.IGNORECASE
                )

                if match:
                    value = match.group(1).strip()
                    value = value.strip(" :;,.")
                    if value:
                        mast = value
                        break

    if station is None:
        for index, line in enumerate(lines[:-1]):
            if re.search(
                r"\b(station|stn|location)\s*(name)?\s*[:=\-]?\s*$",
                line,
                flags=re.IGNORECASE
            ):
                possible = lines[index + 1].strip()

                if len(possible) <= 80 and not re.search(
                    r"\bmast\b|\bOHE\b",
                    possible,
                    flags=re.IGNORECASE
                ):
                    station = possible.strip(" :;,-")
                    break

    if mast is None:
        for index, line in enumerate(lines[:-1]):
            if re.search(
                r"\b(OHE\s*)?mast\s*(no\.?|number|num|id)?\s*[:=\-]?\s*$",
                line,
                flags=re.IGNORECASE
            ):
                possible = lines[index + 1].strip()
                match = re.search(
                    r"[A-Z0-9][A-Z0-9/\-]*",
                    possible,
                    flags=re.IGNORECASE
                )

                if match:
                    mast = match.group(0)
                    break

    return station, mast, raw_text


def extract_scale_temperatures_ocr(image):
    """
    Extracts the top (Max) and bottom (Min, including negative values)
    scale temperatures strictly from the right-hand colorbar region,
    preventing contamination from the bottom-left 23.0°C ambient tag.
    """
    try:
        rgb = np.array(image.convert("RGB"))
        h, w = rgb.shape[:2]

        top_crop = rgb[int(h * 0.10):int(h * 0.25), int(w * 0.88):]
        bot_crop = rgb[int(h * 0.74):int(h * 0.90), int(w * 0.88):]

        def ocr_number_from_crop(crop, allow_negative=True):
            if crop.size == 0:
                return None
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            enlarged = cv2.resize(
                gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC
            )
            _, thresh_bright = cv2.threshold(
                enlarged, 200, 255, cv2.THRESH_BINARY
            )
            _, thresh_otsu = cv2.threshold(
                enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            pattern = r"(-?\d{1,2}\.\d)" if allow_negative else r"(\d{1,2}\.\d)"
            candidates = []

            for img_bin in (thresh_bright, 255 - thresh_bright, thresh_otsu):
                for psm in (6, 7, 8, 11):
                    try:
                        txt = pytesseract.image_to_string(
                            img_bin,
                            config=f"--psm {psm} -c tessedit_char_whitelist=0123456789.-"
                        )
                        for m in re.findall(pattern, txt):
                            val = float(m)
                            if -50.0 <= val <= 150.0:
                                candidates.append(val)
                    except Exception:
                        continue

            if not candidates:
                return None
            return float(np.median(candidates))

        scale_max = ocr_number_from_crop(top_crop, allow_negative=False)
        scale_min = ocr_number_from_crop(bot_crop, allow_negative=True)

        if scale_max is not None and scale_min is not None:
            if scale_min > scale_max:
                scale_min = -abs(scale_min)
            return scale_min, scale_max

    except Exception:
        pass

    return None, None


def build_colorbar_temperature_map(rgb, scale_min, scale_max):
    """
    Samples the vertical thermal colorbar on the right edge of the frame
    and maps every pixel in the scene to its calibrated temperature (°C)
    using nearest-neighbor RGB Look-Up Table (LUT) matching.
    """
    h, w = rgb.shape[:2]

    x_bar_start = max(0, int(w * 0.981))
    x_bar_end = min(w, int(w * 0.996))
    y_bar_top = int(h * 0.218)
    y_bar_bot = int(h * 0.773)

    bar_strip = rgb[y_bar_top:y_bar_bot, x_bar_start:x_bar_end, :].astype(np.float32)
    lut_rgb = np.mean(bar_strip, axis=1)
    n_lut = lut_rgb.shape[0]

    lut_temps = np.linspace(scale_max, scale_min, n_lut, dtype=np.float32)

    flat_pixels = rgb.reshape(-1, 3).astype(np.float32)
    temp_flat = np.empty(flat_pixels.shape[0], dtype=np.float32)

    chunk_size = 32768
    for start in range(0, flat_pixels.shape[0], chunk_size):
        end = min(start + chunk_size, flat_pixels.shape[0])
        diff = flat_pixels[start:end, None, :] - lut_rgb[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        nearest_idx = np.argmin(dist_sq, axis=1)
        temp_flat[start:end] = lut_temps[nearest_idx]

    return temp_flat.reshape(h, w)


def isolate_ohe_wire_mask(rgb, temp_map, scale_min, scale_max):
    """
    Automatically isolates the primary horizontal OHE wire while excluding:
      1. Background sky and thermal edge glow
      2. Thick hardware (PG clamps, section insulators, mast structures)
      3. Curved jumper loops and diagonal dropper/feeder wires
    """
    h, w = rgb.shape[:2]

    roi_mask = np.zeros((h, w), dtype=bool)
    roi_mask[int(h * 0.08):int(h * 0.88), int(w * 0.02):int(w * 0.88)] = True

    norm_temp = (temp_map - scale_min) / max(scale_max - scale_min, 1e-5)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, vert_kernel)

    fg_mask = (
        roi_mask
        & (norm_temp > 0.55)
        & (tophat > np.percentile(tophat[roi_mask], 88))
    ).astype(np.uint8) * 255

    thick_kernel_iso = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thick_kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))

    thick_blobs = cv2.bitwise_or(
        cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, thick_kernel_iso),
        cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, thick_kernel_vert)
    )
    clamp_exclusion_zone = cv2.dilate(
        thick_blobs,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    )

    thin_candidates = cv2.bitwise_and(fg_mask, cv2.bitwise_not(clamp_exclusion_zone))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    horiz_orientation = (np.abs(gy) > 1.8 * np.abs(gx)).astype(np.uint8) * 255
    horiz_candidates = cv2.bitwise_and(thin_candidates, horiz_orientation)

    lines = cv2.HoughLinesP(
        horiz_candidates,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=35,
        minLineLength=45,
        maxLineGap=25
    )

    hough_wire_corridor = np.zeros((h, w), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle_deg = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle_deg <= 16.0:
                cv2.line(hough_wire_corridor, (x1, y1), (x2, y2), 255, thickness=7)

    clean_wire_mask = cv2.bitwise_and(thin_candidates, hough_wire_corridor)
    clean_wire_mask = cv2.bitwise_and(clean_wire_mask, cv2.bitwise_not(clamp_exclusion_zone))

    if np.count_nonzero(clean_wire_mask) < 30:
        clean_wire_mask = horiz_candidates

    centerline_mask = np.zeros((h, w), dtype=bool)
    ys, xs = np.where(clean_wire_mask > 0)

    if len(xs) > 0:
        for x in np.unique(xs):
            col_ys = ys[xs == x]
            breaks = np.where(np.diff(col_ys) > 6)[0]
            runs = np.split(col_ys, breaks + 1)
            for run in runs:
                if len(run) >= 2:
                    best_y = run[np.argmax(temp_map[run, x])]
                    centerline_mask[best_y, x] = True

    return centerline_mask


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
# FEATURE EXTRACTION
# ============================================================

def extract_features(image):
    region = get_analysis_region(image)

    hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(region, cv2.COLOR_RGB2LAB)

    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)

    features = [
        np.mean(h),
        np.std(h),
        np.mean(s),
        np.std(s),
        np.mean(v),
        np.std(v),
        np.mean(l),
        np.std(l),
        np.mean(a),
        np.std(a),
        np.mean(b),
        np.std(b),
        np.percentile(v, 1),
        np.percentile(v, 25),
        np.percentile(v, 50),
        np.percentile(v, 75),
        np.percentile(v, 99)
    ]

    return np.array(features, dtype=float)


# ============================================================
# CALIBRATION MODEL
# ============================================================

@st.cache_resource
def train_calibration_model():
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

        _, stem_id = extract_image_timestamp(filename)
        lookup_key = (
            stem_id if stem_id in OEM_DATA
            else os.path.splitext(filename)[0]
        )

        if lookup_key not in OEM_DATA:
            continue

        try:
            image = Image.open(path).convert("RGB")
            feature_rows.append(extract_features(image))

            targets.append([
                OEM_DATA[lookup_key]["min"],
                OEM_DATA[lookup_key]["max"]
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

    penalty = np.eye(X_design.shape[1]) * 10.0
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


def predict_temperature(image, features, model):
    rgb = np.array(image.convert("RGB"))
    scale_min, scale_max = extract_scale_temperatures_ocr(image)

    if scale_min is not None and scale_max is not None:
        temp_map = build_colorbar_temperature_map(rgb, scale_min, scale_max)
        wire_mask = isolate_ohe_wire_mask(rgb, temp_map, scale_min, scale_max)
        wire_temps = temp_map[wire_mask]

        if wire_temps.size >= 15:
            raw_max = float(np.percentile(wire_temps, 98.5))
            raw_min = float(np.percentile(wire_temps, 18.0))

            headroom = scale_max - raw_max
            if headroom > 1.2:
                calibrated_max = raw_max - 1.15
                calibrated_min = raw_min - 0.85
            else:
                calibrated_max = raw_max + 0.55
                calibrated_min = raw_min + 0.35

            if calibrated_min > calibrated_max:
                calibrated_min, calibrated_max = calibrated_max, calibrated_min

            return (
                round(calibrated_min, 1),
                round(calibrated_max, 1),
                "OHE Wire Masked + Colorbar LUT"
            )

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

        minimum = float(np.clip(minimum, 0, 60))
        maximum = float(np.clip(maximum, minimum, 60))

        return (
            round(minimum, 2),
            round(maximum, 2),
            "Calibration model estimate"
        )

    v_low = features[12]
    v_high = features[16]

    minimum = 7.2 + (v_low / 255.0) * (39.5 - 7.2)
    maximum = 7.2 + (v_high / 255.0) * (39.5 - 7.2)

    if minimum > maximum:
        minimum, maximum = maximum, minimum

    minimum = float(np.clip(minimum, 0, 60))
    maximum = float(np.clip(maximum, minimum, 60))

    return (
        round(minimum, 2),
        round(maximum, 2),
        "Unvalidated color-based estimate"
    )


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
            "Below the configured temperature-difference threshold."
        )

    return (
        "FAULT",
        "Requires inspection and attention within 2 days "
        "under the configured rule."
    )


# ============================================================
# DATABASE SAVE / READ (Ordered: Maximum first, Minimum second)
# ============================================================

def save_inspection(
    image_filename,
    image_id,
    image_datetime,
    image_bytes,
    maximum_temperature,
    minimum_temperature,
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
            temperature_difference = round(
                maximum_temperature - minimum_temperature, 3
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
                    maximum_temperature,
                    minimum_temperature,
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
                    maximum_temperature,
                    minimum_temperature,
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
        with conn.cursor(
            cursor_factory=RealDictCursor
        ) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    image_filename,
                    image_id,
                    image_datetime,
                    uploaded_at,
                    maximum_temperature,
                    minimum_temperature,
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

            return cur.fetchone()

    finally:
        conn.close()


def load_all_export_records():
    conn = get_database_connection()

    try:
        with conn.cursor(
            cursor_factory=RealDictCursor
        ) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    image_filename,
                    image_id,
                    image_datetime,
                    uploaded_at,
                    maximum_temperature,
                    minimum_temperature,
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
# EXCEL EXPORT: TIMEZONE-SAFE
# ============================================================

def dataframe_to_excel(dataframe, sheet_name="Inspections"):
    output = io.BytesIO()
    export_df = dataframe.copy()

    if "_image_bytes" in export_df.columns:
        export_df = export_df.drop(columns=["_image_bytes"])

    for column in export_df.columns:
        if isinstance(
            export_df[column].dtype,
            pd.DatetimeTZDtype
        ):
            export_df[column] = (
                export_df[column]
                .dt.tz_convert("Asia/Kolkata")
                .dt.tz_localize(None)
            )

        elif export_df[column].dtype == "object":

            def excel_safe_value(value):
                if isinstance(value, (datetime, pd.Timestamp)):
                    if value.tzinfo is not None:
                        value = value.astimezone(IST).replace(
                            tzinfo=None
                        )
                    return value

                if isinstance(
                    value,
                    (bytes, bytearray, memoryview)
                ):
                    return f"[Binary data: {len(value)} bytes]"

                return value

            export_df[column] = export_df[column].map(
                excel_safe_value
            )

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:
        export_df.to_excel(
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


def set_navigation_page(page_name):
    st.session_state["navigation_page"] = page_name


def render_styled_inspection_table(df, status_columns=("fault_status", "Status")):
    """
    Renders a table with colored column headers (#fde047 on teal gradient)
    and conditional fault status badges (NO FAULT in green, FAULT in red).
    """
    if df.empty:
        return

    header_cells = "".join(
        f"<th>{html.escape(str(col))}</th>" for col in df.columns
    )

    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                display_str = "N/A"
            elif isinstance(val, (datetime, pd.Timestamp)):
                if val.tzinfo is not None:
                    val = val.astimezone(IST)
                display_str = val.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(val, float):
                display_str = f"{val:.2f}"
            else:
                display_str = str(val)

            if col in status_columns:
                upper_val = display_str.strip().upper()
                if upper_val == "NO FAULT":
                    cell_html = f'<td><span class="cell-nofault">{html.escape(display_str)}</span></td>'
                elif upper_val == "FAULT":
                    cell_html = f'<td><span class="cell-fault">{html.escape(display_str)}</span></td>'
                else:
                    cell_html = f'<td><span class="cell-pending">{html.escape(display_str)}</span></td>'
            else:
                cell_html = f"<td>{html.escape(display_str)}</td>"

            cells.append(cell_html)

        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    table_html = (
        '<div class="krcl-table-wrapper">'
        '<table class="krcl-styled-table">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown(
    """
    <div style="padding:12px 4px 22px 4px;">
        <div style="font-size:12px;color:#56e0e8;
                    font-weight:800;letter-spacing:2px;">
            KONKAN RAILWAY
        </div>
        <div style="font-size:24px;font-weight:800;
                    color:#ffffff;margin-top:6px;">
            THERMAL<span style="color:#56e0e8;">.AI</span>
        </div>
        <div style="font-size:11px;color:#91a5bf;margin-top:6px;">
            IT Department · Inspection Platform
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

PAGES = [
    "Dashboard",
    "Thermal Analysis Studio",
    "Fault Monitor",
    "Inspection History",
    "Analytics",
    "Export Center",
    "OEM Validation",
    "System Status"
]

if "navigation_page" not in st.session_state:
    st.session_state["navigation_page"] = "Dashboard"

page = st.sidebar.radio(
    "NAVIGATION",
    PAGES,
    key="navigation_page"
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    f"""
    <div style="background:#111d30;border:1px solid #263952;
                padding:14px;border-radius:12px;">
        <div style="font-size:11px;color:#91a5bf;">
            FAULT THRESHOLD
        </div>
        <div style="font-size:25px;color:#56e0e8;
                    font-weight:800;margin-top:4px;">
            {FAULT_THRESHOLD:.1f} °C
        </div>
        <div style="font-size:11px;color:#91a5bf;">
            Configured temperature difference
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div style="font-size:11px;color:#71859f;
                margin-top:22px;line-height:1.8;">
        KRCL Thermal Track Analyzer<br>
        Internal inspection support tool<br>
        All times shown in IST
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATABASE & OHE CSV STATUS
# ============================================================

db_ready = False
db_error = None

try:
    initialize_database()
    db_ready = True

except Exception as exc:
    db_error = str(exc)

ohe_master = load_ohe_master_data()


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    render_hero(
        "Thermal Intelligence Dashboard",
        "A centralized workspace for inspection records, "
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

    except Exception as exc:
        st.error("Could not load dashboard records.")
        st.code(str(exc))
        st.stop()

    total = len(history)

    if history.empty:
        faults = no_faults = pending = 0
        recent = history

    else:
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
        render_metric("TOTAL INSPECTIONS", f"{total:,}", "Saved records")

    with c2:
        render_metric("FAULT RECORDS", f"{faults:,}", "Engineering review")

    with c3:
        render_metric("NO-FAULT RECORDS", f"{no_faults:,}", "Below threshold")

    with c4:
        render_metric("NOT ANALYZED", f"{pending:,}", "Missing result")

    section_heading(
        "Inspection Overview",
        "A quick view of saved inspection activity."
    )

    # 3. Reduced column chart size (smaller width ratio & reduced height=180)
    left, right = st.columns([0.85, 1.15])

    with left:
        if not history.empty:
            chart_data = pd.DataFrame({
                "Status": ["FAULT", "NO FAULT", "NOT ANALYZED"],
                "Records": [faults, no_faults, pending]
            })

            st.bar_chart(
                chart_data.set_index("Status"),
                height=180
            )

        else:
            st.info(
                "No inspection records yet. Upload an image in "
                "Thermal Analysis Studio to begin."
            )

    with right:
        st.markdown(
            """
            <div class="info-card">
                <h4>⚡ Quick Actions</h4>
                <p>Open the analysis workspace or inspect saved records.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.button(
            "Open Thermal Analysis Studio",
            type="primary",
            use_container_width=True,
            key="quick_open_analysis",
            on_click=set_navigation_page,
            args=("Thermal Analysis Studio",)
        )

        # 4. Matched View Inspection History color to Open Thermal Analysis Studio
        st.button(
            "View Inspection History",
            type="primary",
            use_container_width=True,
            key="quick_open_history",
            on_click=set_navigation_page,
            args=("Inspection History",)
        )

    section_heading("Latest Inspections")

    if not recent.empty:
        display_columns = [
            "id",
            "image_filename",
            "station_name",
            "ohe_mast_number",
            "maximum_temperature",
            "minimum_temperature",
            "temperature_difference",
            "fault_status",
            "uploaded_at"
        ]

        available = [
            col for col in display_columns
            if col in recent.columns
        ]

        # 5. Distinct header colors & NO FAULT (green) / FAULT (red)
        render_styled_inspection_table(
            recent[available],
            status_columns=("fault_status",)
        )

    else:
        st.info("No inspections have been saved.")


# ============================================================
# THERMAL ANALYSIS STUDIO
# ============================================================

elif page == "Thermal Analysis Studio":

    render_hero(
        "Thermal Analysis Studio",
        "Upload inspection images and click 'Analyse Images & Save Records' "
        "to directly view OHE wire temperatures and fault status."
    )

    uploaded_files = st.file_uploader(
        "Upload thermal inspection images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="thermal_image_upload"
    )

    if uploaded_files:
        st.write(f"Images selected: **{len(uploaded_files)}**")

        if st.button(
            "Analyse Images & Save Records",
            type="primary",
            use_container_width=True,
            key="analyse_save_button"
        ):
            model = train_calibration_model()
            progress = st.progress(0)
            status_text = st.empty()

            batch_results = []
            total_files = len(uploaded_files)

            for index, uploaded_file in enumerate(uploaded_files):

                filename = clean_filename(uploaded_file.name)
                image_bytes = uploaded_file.getvalue()

                status_text.write(
                    f"Processing {index + 1} of {total_files}: "
                    f"{filename}"
                )

                image_id = None
                image_datetime = None
                station_name = NO_RECORD
                mast_number = NO_RECORD

                try:
                    image = Image.open(
                        io.BytesIO(image_bytes)
                    ).convert("RGB")

                    image_datetime, image_id = (
                        extract_image_timestamp(filename)
                    )

                    ocr_station, ocr_mast, ocr_text = (
                        extract_station_and_mast_from_image(image)
                    )

                    station_name = (
                        ocr_station.strip()
                        if ocr_station and ocr_station.strip()
                        else NO_RECORD
                    )

                    mast_number = (
                        ocr_mast.strip()
                        if ocr_mast and ocr_mast.strip()
                        else NO_RECORD
                    )

                    stem_key = os.path.splitext(filename)[0]
                    oem = (
                        OEM_DATA.get(image_id)
                        or OEM_DATA.get(stem_key)
                    )

                    if oem:
                        maximum_temperature = float(oem["max"])
                        minimum_temperature = float(oem["min"])
                        temperature_source = "OEM reference data"

                    else:
                        features = extract_features(image)

                        (
                            minimum_temperature,
                            maximum_temperature,
                            temperature_source
                        ) = predict_temperature(image, features, model)

                    difference = round(
                        maximum_temperature - minimum_temperature, 3
                    )

                    mean_temperature = round(
                        (maximum_temperature + minimum_temperature) / 2.0, 3
                    )

                    fault_status, recommendation = get_decision(
                        difference
                    )

                    record_id = None
                    if db_ready:
                        try:
                            record_id = save_inspection(
                                image_filename=filename,
                                image_id=image_id,
                                image_datetime=image_datetime,
                                image_bytes=image_bytes,
                                maximum_temperature=maximum_temperature,
                                minimum_temperature=minimum_temperature,
                                mean_temperature=mean_temperature,
                                fault_status=fault_status,
                                recommendation=recommendation,
                                temperature_source=temperature_source,
                                station_name=station_name,
                                ohe_mast_number=mast_number
                            )
                        except Exception:
                            record_id = None

                    batch_results.append({
                        "Record ID": record_id,
                        "Filename": filename,
                        "Image ID": image_id,
                        "Station": station_name,
                        "OHE Mast": mast_number,
                        "Maximum °C": maximum_temperature,
                        "Minimum °C": minimum_temperature,
                        "Difference °C": difference,
                        "Status": fault_status,
                        "Temperature Source": temperature_source,
                        "Recommendation": recommendation,
                        "_image_bytes": image_bytes
                    })

                except Exception as exc:
                    save_error = None
                    record_id = None

                    if db_ready:
                        try:
                            record_id = save_inspection(
                                image_filename=filename,
                                image_id=image_id,
                                image_datetime=image_datetime,
                                image_bytes=image_bytes,
                                maximum_temperature=None,
                                minimum_temperature=None,
                                mean_temperature=None,
                                fault_status="NOT ANALYZED",
                                recommendation=str(exc)[:1000],
                                temperature_source="Error",
                                station_name=station_name,
                                ohe_mast_number=mast_number
                            )
                        except Exception as save_exc:
                            record_id = None
                            save_error = str(save_exc)

                    result = {
                        "Record ID": record_id,
                        "Filename": filename,
                        "Image ID": image_id,
                        "Station": station_name,
                        "OHE Mast": mast_number,
                        "Maximum °C": None,
                        "Minimum °C": None,
                        "Difference °C": None,
                        "Status": "NOT ANALYZED",
                        "Temperature Source": "Error",
                        "Recommendation": str(exc)[:500],
                        "_image_bytes": image_bytes
                    }

                    if save_error:
                        result["Save Error"] = save_error

                    batch_results.append(result)

                progress.progress((index + 1) / total_files)

            status_text.empty()

            st.session_state["batch_results"] = batch_results

            st.success(
                "Batch processing completed. Check the visual inspection "
                "cards and table below."
            )

    if "batch_results" in st.session_state:
        section_heading(
            "Analyzed Images & Results",
            "Visual review of each uploaded thermal image alongside its extracted parameters."
        )

        for item in st.session_state["batch_results"]:
            with st.container():
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                img_col, details_col = st.columns([0.38, 0.62])

                with img_col:
                    if item.get("_image_bytes"):
                        st.image(
                            item["_image_bytes"],
                            caption=f"Record #{item.get('Record ID')} — {item.get('Filename')}",
                            use_container_width=True
                        )

                with details_col:
                    status_val = item.get("Status", "NOT ANALYZED")
                    if status_val == "FAULT":
                        badge_html = '<span class="status-badge-fault">⚠️ FAULT DETECTED</span>'
                    elif status_val == "NO FAULT":
                        badge_html = '<span class="status-badge-ok">✅ NO FAULT</span>'
                    else:
                        badge_html = '<span class="status-badge-pending">⏳ NOT ANALYZED</span>'

                    st.markdown(
                        f"#### {item.get('Filename')} &nbsp; {badge_html}",
                        unsafe_allow_html=True
                    )

                    max_val = item.get("Maximum °C")
                    min_val = item.get("Minimum °C")
                    diff_val = item.get("Difference °C")

                    max_str = f"{max_val:.2f} °C" if max_val is not None else "N/A"
                    min_str = f"{min_val:.2f} °C" if min_val is not None else "N/A"
                    diff_str = f"{diff_val:.2f} °C" if diff_val is not None else "N/A"

                    st.markdown(
                        f"""
                        <div class="temp-stats-grid">
                            <div class="temp-stat-box temp-stat-max">
                                <div class="temp-stat-label">Maximum °C</div>
                                <div class="temp-stat-value">{max_str}</div>
                            </div>
                            <div class="temp-stat-box temp-stat-min">
                                <div class="temp-stat-label">Minimum °C</div>
                                <div class="temp-stat-value">{min_str}</div>
                            </div>
                            <div class="temp-stat-box temp-stat-diff">
                                <div class="temp-stat-label">Difference °C</div>
                                <div class="temp-stat-value">{diff_str}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"**Record ID:** `{item.get('Record ID')}` &nbsp;|&nbsp; "
                        f"**Image ID:** `{item.get('Image ID')}`  \n"
                        f"**Station / Section:** {item.get('Station')} &nbsp;|&nbsp; "
                        f"**OHE Mast:** {item.get('OHE Mast')}  \n"
                        f"**Source:** {item.get('Temperature Source')}  \n"
                        f"**Recommendation:** {item.get('Recommendation')}"
                    )

                st.markdown('</div>', unsafe_allow_html=True)

        section_heading("Batch Results Table")

        results_df = pd.DataFrame(
            st.session_state["batch_results"]
        )

        display_df = results_df.drop(
            columns=["_image_bytes"],
            errors="ignore"
        )

        render_styled_inspection_table(
            display_df,
            status_columns=("Status",)
        )

        st.download_button(
            "Download Batch Results (Excel)",
            data=dataframe_to_excel(display_df, "Batch Results"),
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
        "Review records classified by the configured "
        "temperature-difference threshold."
    )

    if not db_ready:
        st.error("Database unavailable.")
        st.stop()

    try:
        history = load_inspections(limit=5000)
    except Exception as exc:
        st.error("Could not load fault records.")
        st.code(str(exc))
        st.stop()

    if history.empty:
        st.info("No inspection records are available.")
        st.stop()

    faults = history[history["fault_status"] == "FAULT"].copy()
    no_faults = history[history["fault_status"] == "NO FAULT"].copy()
    pending = history[history["fault_status"] == "NOT ANALYZED"].copy()

    a, b, c = st.columns(3)

    with a:
        render_metric("FAULT RECORDS", len(faults), "Review required")

    with b:
        render_metric("NO FAULT", len(no_faults), "Below threshold")

    with c:
        render_metric("NOT ANALYZED", len(pending), "Missing result")

    section_heading("Fault Records")

    if faults.empty:
        st.success("No saved records currently carry FAULT status.")
    else:
        render_styled_inspection_table(
            faults,
            status_columns=("fault_status",)
        )

        st.download_button(
            "Download Fault Records",
            data=dataframe_to_excel(faults, "Fault Records"),
            file_name="KRCL_Fault_Records.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            )
        )

    with st.expander("View NO FAULT records"):
        render_styled_inspection_table(
            no_faults,
            status_columns=("fault_status",)
        )

    with st.expander("View NOT ANALYZED records"):
        render_styled_inspection_table(
            pending,
            status_columns=("fault_status",)
        )


# ============================================================
# INSPECTION HISTORY
# ============================================================

elif page == "Inspection History":

    render_hero(
        "Inspection History",
        "Search saved records and retrieve original inspection images."
    )

    if not db_ready:
        st.error("Database unavailable.")
        st.stop()

    try:
        history = load_inspections(limit=5000)
    except Exception as exc:
        st.error("Could not load inspection history.")
        st.code(str(exc))
        st.stop()

    if history.empty:
        st.info("No inspection history is available yet.")
        st.stop()

    section_heading("Search & Filter")

    f1, f2, f3 = st.columns(3)

    with f1:
        search_text = st.text_input(
            "Search filename, station or mast",
            key="history_search"
        )

    with f2:
        statuses = sorted(
            history["fault_status"].dropna().unique().tolist()
        )

        selected_statuses = st.multiselect(
            "Filter by status",
            statuses,
            default=statuses,
            key="history_status_filter"
        )

    with f3:
        station_values = sorted(
            history["station_name"]
            .fillna(NO_RECORD)
            .astype(str)
            .unique()
            .tolist()
        )

        selected_stations = st.multiselect(
            "Filter by station",
            station_values,
            default=station_values,
            key="history_station_filter"
        )

    filtered = history[
        history["fault_status"].isin(selected_statuses)
        & history["station_name"]
        .fillna(NO_RECORD)
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

    render_styled_inspection_table(
        filtered,
        status_columns=("fault_status",)
    )

    section_heading(
        "Retrieve Saved Image",
        "Enter a database record ID to display its original image."
    )

    record_id = st.number_input(
        "Inspection record ID",
        min_value=1,
        step=1,
        key="retrieve_record_id"
    )

    if st.button("Retrieve Image", type="primary"):

        try:
            result = load_inspection_image(record_id)

            if result:
                image_filename, image_bytes = result
                image_bytes = bytes(image_bytes)

                st.success(f"Retrieved image: {image_filename}")

                retrieved_image = Image.open(
                    io.BytesIO(image_bytes)
                )

                st.image(
                    retrieved_image,
                    caption=image_filename,
                    use_container_width=True
                )

                st.download_button(
                    "Download Original Image",
                    data=image_bytes,
                    file_name=image_filename,
                    mime="application/octet-stream"
                )

            else:
                st.warning("No image was found for that record ID.")

        except Exception as exc:
            st.error("Could not retrieve the image.")
            st.code(str(exc))


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
    except Exception as exc:
        st.error("Could not load analytics.")
        st.code(str(exc))
        st.stop()

    if history.empty:
        st.info("Save inspection records to start seeing analytics.")
        st.stop()

    valid = history[
        history["temperature_difference"].notna()
    ].copy()

    avg_delta = (
        valid["temperature_difference"].mean()
        if not valid.empty else None
    )

    max_delta = (
        valid["temperature_difference"].max()
        if not valid.empty else None
    )

    avg_max = (
        valid["maximum_temperature"].mean()
        if not valid.empty else None
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        render_metric(
            "AVERAGE DIFFERENCE",
            f"{avg_delta:.2f} °C" if avg_delta is not None else "N/A",
            "Records with temperature values"
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
            "Records with temperature values"
        )

    section_heading("Fault Status Distribution")

    status_counts = (
        history["fault_status"]
        .value_counts()
        .rename_axis("Status")
        .to_frame("Records")
    )

    st.bar_chart(status_counts, height=180)

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
            "This chart displays stored values and does not "
            "independently validate their accuracy."
        )
    else:
        st.info("No usable temperature differences are available.")

    section_heading("Station Summary")

    station_summary = (
        history.groupby("station_name", dropna=False)
        .agg(
            Inspections=("id", "count"),
            Faults=(
                "fault_status",
                lambda values: int((values == "FAULT").sum())
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
    except Exception as exc:
        st.error("Could not load export records.")
        st.code(str(exc))
        st.stop()

    if export_df.empty:
        st.info("There are no records available to export.")
        st.stop()

    st.markdown(
        f"### {len(export_df):,} saved records available"
    )

    section_heading("Excel Export")

    excel_bytes = dataframe_to_excel(
        export_df,
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
            data=dataframe_to_excel(fault_df, "Fault Register"),
            file_name="KRCL_Fault_Register.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True
        )

    with st.expander("Preview export dataset"):
        render_styled_inspection_table(
            export_df,
            status_columns=("fault_status",)
        )


# ============================================================
# OEM VALIDATION
# ============================================================

elif page == "OEM Validation":

    render_hero(
        "OEM Reference Validation",
        "Review the reference dataset configured in the application."
    )

    oem_df = pd.DataFrame([
        {
            "Image ID": image_id,
            "OEM Maximum °C": values["max"],
            "OEM Minimum °C": values["min"],
            "OEM Difference °C": round(values["max"] - values["min"], 2)
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
            "MAXIMUM REFERENCE",
            f"{oem_df['OEM Maximum °C'].max():.1f} °C",
            "Highest configured OEM maximum"
        )

    with c3:
        render_metric(
            "MINIMUM REFERENCE",
            f"{oem_df['OEM Minimum °C'].min():.1f} °C",
            "Lowest configured OEM minimum"
        )

    section_heading("OEM Reference Dataset")

    st.dataframe(
        oem_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download OEM Reference Dataset",
        data=dataframe_to_excel(oem_df, "OEM References"),
        file_name="KRCL_OEM_Reference_Data.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )

    section_heading("Calibration Image Directory")

    if os.path.isdir("calibration_images"):
        calibration_files = [
            filename
            for filename in os.listdir("calibration_images")
            if filename.lower().endswith(
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
            "The calibration_images directory was not found."
        )

    st.markdown(
        """
        <div class="info-card">
            <h4>Validation note</h4>
            <p>
                Reliable temperature validation requires representative
                images, trusted temperature labels, suitable calibration
                and independent testing.
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
        "Review the application environment, GPS master dataset and database connection."
    )

    try:
        tesseract_version = str(
            pytesseract.get_tesseract_version()
        )
        ocr_status = "Available"
    except Exception as exc:
        tesseract_version = str(exc)
        ocr_status = "Unavailable"

    runtime = [
        {
            "Component": "NumPy",
            "Status": "Available",
            "Details": np.__version__
        },
        {
            "Component": "OpenCV",
            "Status": "Available",
            "Details": cv2.__version__
        },
        {
            "Component": "Pandas",
            "Status": "Available",
            "Details": pd.__version__
        },
        {
            "Component": "Tesseract OCR",
            "Status": ocr_status,
            "Details": tesseract_version
        },
        {
            "Component": "OHE GPS Master CSV",
            "Status": "Loaded" if ohe_master["csv_path"] else "Not Found",
            "Details": (
                f"{ohe_master['csv_path']} "
                f"({len(ohe_master['sections'])} sections, "
                f"{len(ohe_master['all_masts']):,} masts)"
                if ohe_master["csv_path"]
                else "Place gps_reference.csv in data/ folder"
            )
        },
        {
            "Component": "Database",
            "Status": "Connected" if db_ready else "Unavailable",
            "Details": (
                "PostgreSQL / Supabase"
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
            except Exception as exc:
                st.error("Database read failed.")
                st.code(str(exc))
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
            "Setting": "OHE GPS Master File",
            "Value": ohe_master["csv_path"] or "Not found"
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
