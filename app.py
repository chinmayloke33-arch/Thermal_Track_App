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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="KRCL Thermal Track Analyzer",
    page_icon="🌡️",
    layout="wide"
)

IST = ZoneInfo("Asia/Kolkata")
FAULT_THRESHOLD = 5.0

st.title("🌡️ KRCL Thermal Track Analyzer")
st.caption(
    "IT Department | Thermal Image Analysis, "
    "OEM Validation and Inspection Records"
)


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
    "20251117-124611-004": {"max": 33.8, "min": 22.8}
}


# ============================================================
# DATABASE CONNECTION AND INITIALIZATION
# ============================================================

def get_database_connection():
    """Connect to PostgreSQL using Streamlit Secrets DATABASE_URL."""
    database_url = st.secrets.get("DATABASE_URL", "")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is missing. Configure it in Streamlit Cloud Secrets."
        )
    return psycopg2.connect(database_url)


def initialize_database():
    """
    Create the inspection table if missing and add location columns.
    Existing records are preserved.
    """
    connection = get_database_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
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
                    image_data BYTEA NOT NULL
                );
                """
            )

            # Safe schema migration: add columns only if they do not exist.
            cursor.execute(
                """
                ALTER TABLE thermal_inspections
                ADD COLUMN IF NOT EXISTS station_name TEXT;
                """
            )
            cursor.execute(
                """
                ALTER TABLE thermal_inspections
                ADD COLUMN IF NOT EXISTS ohe_mast_number TEXT;
                """
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


# ============================================================
# TIMESTAMP EXTRACTION
# ============================================================

def extract_station_and_mast_from_image(image):
    """Use OCR to detect station name and OHE mast number printed in the image.

    Only clearly labelled text is accepted to avoid guessing. If OCR cannot
    find a value, the caller receives "No record found" for that field.
    """
    try:
        rgb = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # Upscale and threshold to improve OCR on small printed annotations.
        enlarged = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        processed = cv2.threshold(
            enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        texts = []
        for candidate in (rgb, processed):
            try:
                texts.append(pytesseract.image_to_string(candidate, config="--psm 11"))
            except Exception:
                continue
        text = "\n".join(texts)
        lines = [re.sub(r"\s+", " ", line).strip(" :|-\t") for line in text.splitlines()]
        lines = [line for line in lines if line]

        station_name = None
        mast_number = None

        for i, line in enumerate(lines):
            # Accept labelled fields such as "Station Name: Ratnagiri".
            station_match = re.search(
                r"(?i)\b(?:station\s*name|station)\s*[:#=-]?\s*(.+)$", line
            )
            if station_match:
                value = station_match.group(1).strip(" :#=-")
                if value and not re.fullmatch(r"(?i)(name|no\.?|number)", value):
                    station_name = value[:100]
                elif i + 1 < len(lines):
                    station_name = lines[i + 1][:100]

            mast_match = re.search(
                r"(?i)\b(?:OHE\s*)?(?:mast|m\s*/\s*t)\s*(?:no\.?|number|#)?\s*[:#=-]?\s*([A-Z0-9]+(?:[/.-][A-Z0-9]+)*)\b",
                line
            )
            if mast_match:
                mast_number = mast_match.group(1).strip(" .,:;")[:50]
            elif re.search(r"(?i)\b(?:OHE\s*)?mast\s*(?:no\.?|number)?\s*[:#=-]?\s*$", line) and i + 1 < len(lines):
                possible = re.search(r"[A-Z0-9]+(?:[/.-][A-Z0-9]+)*", lines[i + 1], re.I)
                if possible:
                    mast_number = possible.group(0)[:50]

        return station_name or "No record found", mast_number or "No record found", text.strip()
    except Exception:
        return "No record found", "No record found", ""


def extract_image_timestamp(filename):
    """
    Recognize filenames such as 20250315-133514-016.jpg.
    Interprets timestamp as India Standard Time.
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    match = re.search(r"(?<!\d)(\d{8})-(\d{6})-(\d+)(?!\d)", name)

    if not match:
        return None, None

    date_part, time_part, image_id = match.groups()
    try:
        timestamp = datetime.strptime(
            date_part + time_part, "%Y%m%d%H%M%S"
        ).replace(tzinfo=IST)
        return timestamp, image_id
    except ValueError:
        return None, image_id


# ============================================================
# FILENAME / OEM MATCHING
# ============================================================

def clean_filename(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r"\s+", "", name)


def find_oem_reference(filename):
    filename = clean_filename(filename)
    if filename in OEM_DATA:
        return OEM_DATA[filename]
    for image_id in OEM_DATA:
        if image_id in filename:
            return OEM_DATA[image_id]
    return None


# ============================================================
# IMAGE PREPROCESSING AND FEATURES
# ============================================================

def get_analysis_region(image):
    image_array = np.array(image.convert("RGB"))
    height, width = image_array.shape[:2]
    left = int(width * 0.03)
    right = int(width * 0.88)
    top = int(height * 0.08)
    bottom = int(height * 0.92)
    return image_array[top:bottom, left:right]


def extract_features(image):
    region = get_analysis_region(image)

    hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(region, cv2.COLOR_RGB2LAB)

    H = hsv[:, :, 0].astype(float)
    S = hsv[:, :, 1].astype(float)
    V = hsv[:, :, 2].astype(float)
    L = lab[:, :, 0].astype(float)
    A = lab[:, :, 1].astype(float)
    B = lab[:, :, 2].astype(float)

    mask = (V > 25) & (S > 20)
    if np.sum(mask) < 100:
        mask = np.ones(V.shape, dtype=bool)

    def percentile(array, value):
        return np.percentile(array[mask], value)

    features = [
        np.mean(H[mask]), np.std(H[mask]),
        np.mean(S[mask]), np.std(S[mask]),
        np.mean(V[mask]), np.std(V[mask]),
        percentile(V, 5), percentile(V, 10),
        percentile(V, 25), percentile(V, 50),
        percentile(V, 75), percentile(V, 90),
        percentile(V, 95), np.mean(L[mask]),
        np.mean(A[mask]), np.mean(B[mask])
    ]
    return np.array(features, dtype=float), region


# ============================================================
# RIDGE CALIBRATION MODEL
# ============================================================

def train_calibration_model():
    """Train from matching OEM images; retain a nearest-reference fallback."""
    folder = "calibration_images"
    X, Y_MAX, Y_MIN = [], [], []

    if os.path.isdir(folder):
        for filename in os.listdir(folder):
            reference = find_oem_reference(filename)
            if reference is None:
                continue
            path = os.path.join(folder, filename)
            try:
                with Image.open(path) as original:
                    features, _ = extract_features(original.convert("RGB"))
                X.append(features)
                Y_MAX.append(reference["max"])
                Y_MIN.append(reference["min"])
            except Exception:
                continue

    # If at least one reference image is available, preserve it for nearest-neighbour fallback.
    if len(X) > 0:
        X = np.asarray(X, dtype=float)
        Y_MAX = np.asarray(Y_MAX, dtype=float)
        Y_MIN = np.asarray(Y_MIN, dtype=float)
    else:
        X = np.empty((0, 16), dtype=float)
        Y_MAX = np.array([], dtype=float)
        Y_MIN = np.array([], dtype=float)

    if len(X) < 4:
        return {
            "fallback": True,
            "fallback_X": X,
            "fallback_max": Y_MAX,
            "fallback_min": Y_MIN,
            "samples": len(X)
        }

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-9] = 1.0
    X_scaled = (X - mean) / std

    alpha = 10.0
    identity = np.eye(X_scaled.shape[1])
    coef_max = np.linalg.solve(
        X_scaled.T @ X_scaled + alpha * identity,
        X_scaled.T @ Y_MAX
    )
    coef_min = np.linalg.solve(
        X_scaled.T @ X_scaled + alpha * identity,
        X_scaled.T @ Y_MIN
    )

    return {
        "fallback": False,
        "mean": mean,
        "std": std,
        "coef_max": coef_max,
        "coef_min": coef_min,
        "intercept_max": float(np.mean(Y_MAX - X_scaled @ coef_max)),
        "intercept_min": float(np.mean(Y_MIN - X_scaled @ coef_min)),
        "fallback_X": X,
        "fallback_max": Y_MAX,
        "fallback_min": Y_MIN,
        "samples": len(X)
    }


# ============================================================
# TEMPERATURE PREDICTION
# ============================================================

def predict_temperature(features, model):
    """Return estimated min/max for any image, including unknown filenames."""
    observed_min = min(value["min"] for value in OEM_DATA.values())
    observed_max = max(value["max"] for value in OEM_DATA.values())

    if model is None:
        # Last-resort estimate based on the centre of the available OEM range.
        predicted_min = float(np.mean([v["min"] for v in OEM_DATA.values()]))
        predicted_max = float(np.mean([v["max"] for v in OEM_DATA.values()]))
    elif model.get("fallback", False):
        reference_features = model.get("fallback_X", np.empty((0, len(features))))
        if len(reference_features) > 0:
            # Standardize each feature before selecting the most similar reference image.
            scale = np.std(reference_features, axis=0)
            scale[scale < 1e-9] = 1.0
            standardized_refs = (reference_features - np.mean(reference_features, axis=0)) / scale
            standardized_new = (features - np.mean(reference_features, axis=0)) / scale
            distances = np.linalg.norm(standardized_refs - standardized_new, axis=1)
            nearest = int(np.argmin(distances))
            predicted_max = float(model["fallback_max"][nearest])
            predicted_min = float(model["fallback_min"][nearest])
        else:
            # No usable calibration files: provide a clearly labelled rough baseline estimate.
            predicted_min = float(np.mean([v["min"] for v in OEM_DATA.values()]))
            predicted_max = float(np.mean([v["max"] for v in OEM_DATA.values()]))
    else:
        X = (features - model["mean"]) / model["std"]
        predicted_max = float(model["intercept_max"] + X @ model["coef_max"])
        predicted_min = float(model["intercept_min"] + X @ model["coef_min"])

    if predicted_min > predicted_max:
        predicted_min, predicted_max = predicted_max, predicted_min

    predicted_min = float(np.clip(predicted_min, observed_min, observed_max))
    predicted_max = float(np.clip(predicted_max, observed_min, observed_max))
    return predicted_min, predicted_max


# ============================================================
# FAULT DECISION AND RESULT DISPLAY
# ============================================================

def get_decision(delta):
    if delta < FAULT_THRESHOLD:
        return "NO FAULT", "Temperature difference is below 5°C."
    return "FAULT", "Attention required within 2 days."


def display_result(
    minimum,
    maximum,
    source,
    image_datetime=None,
    image_id=None,
    reference=None
):
    difference = maximum - minimum
    mean_temperature = (minimum + maximum) / 2
    status, recommendation = get_decision(difference)

    st.subheader("Analysis Results")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Minimum Temperature", f"{minimum:.1f} °C")
    col2.metric("Maximum Temperature", f"{maximum:.1f} °C")
    col3.metric("Temperature Difference", f"{difference:.1f} °C")
    col4.metric("Mean of Min/Max", f"{mean_temperature:.1f} °C")

    if image_datetime is not None:
        date_col, time_col = st.columns(2)
        date_col.write(
            "**Image Date:** " + image_datetime.strftime("%d %B %Y")
        )
        time_col.write(
            "**Image Time:** " + image_datetime.strftime("%H:%M:%S IST")
        )
    else:
        st.warning(
            "Timestamp unavailable in filename. Printed-image OCR is not enabled."
        )

    if image_id is not None:
        st.write(f"**Image ID:** {image_id}")

    if status == "FAULT":
        st.error("🔴 FAULT — ATTENTION REQUIRED WITHIN 2 DAYS")
    else:
        st.success("🟢 NO FAULT")

    st.write(f"**Recommendation:** {recommendation}")
    st.caption(f"Temperature source: {source}")

    if reference is not None:
        st.info(
            "OEM reference: "
            f"Maximum {reference['max']:.1f} °C | "
            f"Minimum {reference['min']:.1f} °C"
        )


# ============================================================
# SAVE INSPECTION
# ============================================================

def save_inspection(
    filename,
    image_bytes,
    image_datetime,
    image_id,
    minimum,
    maximum,
    source,
    station_name,
    ohe_mast_number
):
    """Save each uploaded image, location, and its analysis result."""
    uploaded_at = datetime.now(IST)
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    if minimum is not None and maximum is not None:
        minimum = float(minimum)
        maximum = float(maximum)
        difference = maximum - minimum
        mean_temperature = (minimum + maximum) / 2
        status, recommendation = get_decision(difference)
    else:
        difference = None
        mean_temperature = None
        status = "NOT ANALYZED"
        recommendation = (
            "Temperature values unavailable. Review the image or provide "
            "a validated temperature source."
        )

    connection = get_database_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
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
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING id;
                """,
                (
                    filename,
                    image_id,
                    image_datetime,
                    uploaded_at,
                    minimum,
                    maximum,
                    difference,
                    mean_temperature,
                    status,
                    recommendation,
                    source,
                    image_hash,
                    psycopg2.Binary(image_bytes),
                    station_name,
                    ohe_mast_number
                )
            )
            record_id = cursor.fetchone()[0]

        connection.commit()
        return record_id
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Analysis Settings")
    st.metric("Fault Threshold", "5.0 °C")
    st.write("ΔT < 5°C → NO FAULT")
    st.write("ΔT ≥ 5°C → FAULT")
    st.write("FAULT → Attention required within 2 days")
    st.divider()
    st.subheader("Calibration")

    calibration_folder = "calibration_images"
    if os.path.isdir(calibration_folder):
        count = len([
            filename for filename in os.listdir(calibration_folder)
            if filename.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        st.write(f"Calibration images: {count}")
    else:
        st.warning("calibration_images folder not found.")


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

try:
    initialize_database()
    database_ready = True
except Exception as error:
    database_ready = False
    st.error(
        "Database connection failed. Check DATABASE_URL in Streamlit Cloud Secrets."
    )
    st.caption(str(error))


# ============================================================
# ANALYSIS FORM
# ============================================================

st.header("📤 Upload Thermal Images")
st.info(
    "Station Name and OHE Mast Number will be searched in the image using OCR. "
    "If either value is not clearly detected, the app will show 'No record found'."
)

with st.form("thermal_analysis_form"):
    uploaded_files = st.file_uploader(
        "Choose thermal image(s)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    analyze_button = st.form_submit_button(
        "Analyze and Save Results",
        type="primary"
    )


# ============================================================
# PROCESS UPLOADS
# ============================================================

if analyze_button:
    if not database_ready:
        st.error(
            "Cannot save results because the database is unavailable. "
            "No analysis was submitted."
        )
    elif not uploaded_files:
        st.warning("Please upload at least one image.")
    else:
        model = train_calibration_model()
        results = []

        for uploaded_file in uploaded_files:
            image_bytes = uploaded_file.getvalue()

            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception as error:
                st.error(f"Could not read {uploaded_file.name}: {error}")
                continue

            st.divider()
            st.subheader(f"📷 {uploaded_file.name}")

            image_datetime, image_id = extract_image_timestamp(
                uploaded_file.name
            )
            station_name, ohe_mast_number, ocr_text = extract_station_and_mast_from_image(image)

            image_col, result_col = st.columns([1.1, 1.5])

            with image_col:
                st.image(
                    image,
                    caption="Uploaded Thermal Image",
                    use_container_width=True
                )

            reference = find_oem_reference(uploaded_file.name)
            minimum = None
            maximum = None
            source = "Temperature unavailable"

            with result_col:
                st.write(f"**Station Name:** {station_name}")
                st.write(f"**OHE Mast Number:** {ohe_mast_number}")
                if station_name == "No record found" or ohe_mast_number == "No record found":
                    st.warning("Station or mast details were not detected clearly in this image.")
                with st.expander("View OCR text detected from image"):
                    st.text(ocr_text if ocr_text else "No readable text found in image.")

                if reference is not None:
                    minimum = reference["min"]
                    maximum = reference["max"]
                    source = "OEM Reference"

                    display_result(
                        minimum,
                        maximum,
                        source,
                        image_datetime,
                        image_id,
                        reference
                    )

                    st.caption(
                        "These values are the stored OEM reference readings "
                        "for this filename."
                    )
                else:
                    features, region = extract_features(image)

                    minimum, maximum = predict_temperature(features, model)
                    sample_count = model["samples"] if model is not None else 0
                    if model is not None and not model.get("fallback", False):
                        source = f"Empirical calibrated estimate using {sample_count} OEM images"
                    elif sample_count > 0:
                        source = f"Nearest-reference estimate using {sample_count} OEM images"
                    else:
                        source = "Rough OEM-range baseline estimate (no usable calibration images)"

                    display_result(
                        minimum,
                        maximum,
                        source,
                        image_datetime,
                        image_id
                    )

                    st.warning(
                        "Unknown image: temperatures are estimated from image-colour features, "
                        "not directly measured radiometric values. Estimates may be inaccurate; "
                        "validate against the camera/OEM readings before operational use."
                    )

                    with st.expander("View region used for image analysis"):
                        st.image(
                            region,
                            caption="Image feature extraction region",
                            use_container_width=True
                        )

            # Save each image and its location details.
            record_id = None
            try:
                record_id = save_inspection(
                    filename=uploaded_file.name,
                    image_bytes=image_bytes,
                    image_datetime=image_datetime,
                    image_id=image_id,
                    minimum=minimum,
                    maximum=maximum,
                    source=source,
                    station_name=station_
