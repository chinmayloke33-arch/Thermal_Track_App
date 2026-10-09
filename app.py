
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
    "20250315-133514-016": {
        "max": 39.5,
        "min": 37.6
    },
    "20251111-112915-003": {
        "max": 31.5,
        "min": 21.9
    },
    "20251111-114541-001": {
        "max": 35.6,
        "min": 7.2
    },
    "20251111-114605-002": {
        "max": 35.4,
        "min": 27.4
    },
    "20251113-074547-002": {
        "max": 17.6,
        "min": 13.7
    },
    "20251113-091808-012": {
        "max": 24.7,
        "min": 19.3
    },
    "20251113-092707-005": {
        "max": 24.4,
        "min": 20.5
    },
    "20251113-093327-012": {
        "max": 24.0,
        "min": 18.3
    },
    "20251113-113151-016": {
        "max": 34.5,
        "min": 24.7
    },
    "20251114-101128-007": {
        "max": 26.2,
        "min": 15.8
    },
    "20251114-114841-018": {
        "max": 35.0,
        "min": 13.6
    },
    "20251117-124611-004": {
        "max": 33.8,
        "min": 22.8
    }
}


# ============================================================
# DATABASE SETUP
# ============================================================

def get_database_connection():
    """
    Connect to the hosted PostgreSQL database.

    DATABASE_URL must be configured in Streamlit Secrets.
    """

    database_url = st.secrets.get(
        "DATABASE_URL",
        ""
    )

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is missing. "
            "Configure it in Streamlit Cloud Secrets."
        )

    return psycopg2.connect(database_url)


def initialize_database():
    """
    Create the inspection table if it does not exist.
    """

    connection = get_database_connection()

    try:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS thermal_inspections (
                    id BIGINT GENERATED ALWAYS AS IDENTITY
                        PRIMARY KEY,

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

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def save_inspection(
    filename,
    image_bytes,
    image_datetime,
    image_id,
    minimum,
    maximum,
    source
):
    """
    Save the uploaded image and analysis result.

    If temperatures cannot be reliably calculated,
    the image is still stored with NOT ANALYZED status.
    """

    uploaded_at = datetime.now(IST)

    image_hash = hashlib.sha256(
        image_bytes
    ).hexdigest()

    if minimum is not None and maximum is not None:

        minimum = float(minimum)
        maximum = float(maximum)

        difference = maximum - minimum

        mean_temperature = (
            minimum + maximum
        ) / 2

        status, recommendation = get_decision(
            difference
        )

    else:

        difference = None
        mean_temperature = None

        status = "NOT ANALYZED"

        recommendation = (
            "Temperature values unavailable. "
            "Review the image or provide a validated "
            "temperature source."
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
                    image_data
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s
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
                    psycopg2.Binary(image_bytes)
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
# TIMESTAMP EXTRACTION
# ============================================================

def extract_image_timestamp(filename):
    """
    Recognize filenames such as:

    20250315-133514-016.jpg

    Interpreted as:
    15 March 2025, 13:35:14 IST.
    """

    name = os.path.splitext(
        os.path.basename(filename)
    )[0]

    match = re.search(
        r"(?<!\d)(\d{8})-(\d{6})-(\d+)(?!\d)",
        name
    )

    if not match:
        return None, None

    date_part = match.group(1)
    time_part = match.group(2)
    image_id = match.group(3)

    try:
        timestamp = datetime.strptime(
            date_part + time_part,
            "%Y%m%d%H%M%S"
        )

        timestamp = timestamp.replace(
            tzinfo=IST
        )

        return timestamp, image_id

    except ValueError:
        return None, image_id


# ============================================================
# FILENAME MATCHING
# ============================================================

def clean_filename(filename):

    name = os.path.splitext(
        os.path.basename(filename)
    )[0]

    return re.sub(
        r"\s+",
        "",
        name
    )


def find_oem_reference(filename):

    filename = clean_filename(filename)

    if filename in OEM_DATA:
        return OEM_DATA[filename]

    for image_id in OEM_DATA:

        if image_id in filename:
            return OEM_DATA[image_id]

    return None


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def get_analysis_region(image):

    image_array = np.array(
        image.convert("RGB")
    )

    height, width = image_array.shape[:2]

    left = int(width * 0.03)
    right = int(width * 0.88)

    top = int(height * 0.08)
    bottom = int(height * 0.92)

    return image_array[
        top:bottom,
        left:right
    ]


# ============================================================
# IMAGE FEATURE EXTRACTION
# ============================================================

def extract_features(image):

    region = get_analysis_region(image)

    hsv = cv2.cvtColor(
        region,
        cv2.COLOR_RGB2HSV
    )

    lab = cv2.cvtColor(
        region,
        cv2.COLOR_RGB2LAB
    )

    H = hsv[:, :, 0].astype(float)
    S = hsv[:, :, 1].astype(float)
    V = hsv[:, :, 2].astype(float)

    L = lab[:, :, 0].astype(float)
    A = lab[:, :, 1].astype(float)
    B = lab[:, :, 2].astype(float)

    mask = (
        (V > 25) &
        (S > 20)
    )

    if np.sum(mask) < 100:
        mask = np.ones(
            V.shape,
            dtype=bool
        )

    def percentile(array, value):

        return np.percentile(
            array[mask],
            value
        )

    features = [
        np.mean(H[mask]),
        np.std(H[mask]),

        np.mean(S[mask]),
        np.std(S[mask]),

        np.mean(V[mask]),
        np.std(V[mask]),

        percentile(V, 5),
        percentile(V, 10),
        percentile(V, 25),
        percentile(V, 50),
        percentile(V, 75),
        percentile(V, 90),
        percentile(V, 95),

        np.mean(L[mask]),
        np.mean(A[mask]),
        np.mean(B[mask])
    ]

    return np.array(
        features,
        dtype=float
    ), region


# ============================================================
# RIDGE CALIBRATION MODEL
# ============================================================

def train_calibration_model():

    folder = "calibration_images"

    if not os.path.isdir(folder):
        return None

    X = []
    Y_MAX = []
    Y_MIN = []

    for filename in os.listdir(folder):

        reference = find_oem_reference(
            filename
        )

        if reference is None:
            continue

        path = os.path.join(
            folder,
            filename
        )

        try:

            with Image.open(path) as original:

                image = original.convert("RGB")

                features, _ = extract_features(
                    image
                )

            X.append(features)

            Y_MAX.append(
                reference["max"]
            )

            Y_MIN.append(
                reference["min"]
            )

        except Exception:
            continue

    if len(X) < 4:
        return None

    X = np.asarray(
        X,
        dtype=float
    )

    Y_MAX = np.asarray(
        Y_MAX,
        dtype=float
    )

    Y_MIN = np.asarray(
        Y_MIN,
        dtype=float
    )

    mean = X.mean(axis=0)

    std = X.std(axis=0)

    std[std < 1e-9] = 1.0

    X_scaled = (
        X - mean
    ) / std

    alpha = 10.0

    identity = np.eye(
        X_scaled.shape[1]
    )

    coef_max = np.linalg.solve(
        X_scaled.T @ X_scaled
        + alpha * identity,

        X_scaled.T @ Y_MAX
    )

    coef_min = np.linalg.solve(
        X_scaled.T @ X_scaled
        + alpha * identity,

        X_scaled.T @ Y_MIN
    )

    intercept_max = np.mean(
        Y_MAX - X_scaled @ coef_max
    )

    intercept_min = np.mean(
        Y_MIN - X_scaled @ coef_min
    )

    return {
        "mean": mean,
        "std": std,
        "coef_max": coef_max,
        "coef_min": coef_min,
        "intercept_max": intercept_max,
        "intercept_min": intercept_min,
        "samples": len(X)
    }


# ============================================================
# TEMPERATURE PREDICTION
# ============================================================

def predict_temperature(features, model):

    X = (
        features - model["mean"]
    ) / model["std"]

    predicted_max = (
        model["intercept_max"]
        + X @ model["coef_max"]
    )

    predicted_min = (
        model["intercept_min"]
        + X @ model["coef_min"]
    )

    predicted_max = float(
        predicted_max
    )

    predicted_min = float(
        predicted_min
    )

    if predicted_min > predicted_max:

        predicted_min, predicted_max = (
            predicted_max,
            predicted_min
        )

    observed_min = min(
        value["min"]
        for value in OEM_DATA.values()
    )

    observed_max = max(
        value["max"]
        for value in OEM_DATA.values()
    )

    predicted_min = float(
        np.clip(
            predicted_min,
            observed_min,
            observed_max
        )
    )

    predicted_max = float(
        np.clip(
            predicted_max,
            observed_min,
            observed_max
        )
    )

    return predicted_min, predicted_max


# ============================================================
# FAULT DECISION
# ============================================================

def get_decision(delta):

    if delta < FAULT_THRESHOLD:

        return (
            "NO FAULT",
            "Temperature difference is below 5°C."
        )

    return (
        "FAULT",
        "Attention required within 2 days."
    )


# ============================================================
# RESULT DISPLAY
# ============================================================

def display_result(
    minimum,
    maximum,
    source,
    image_datetime=None,
    image_id=None,
    reference=None
):

    difference = maximum - minimum

    mean_temperature = (
        minimum + maximum
    ) / 2

    status, recommendation = get_decision(
        difference
    )

    st.subheader("Analysis Results")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Minimum Temperature",
        f"{minimum:.1f} °C"
    )

    col2.metric(
        "Maximum Temperature",
        f"{maximum:.1f} °C"
    )

    col3.metric(
        "Temperature Difference",
        f"{difference:.1f} °C"
    )

    col4.metric(
        "Mean of Min/Max",
        f"{mean_temperature:.1f} °C"
    )

    if image_datetime is not None:

        date_col, time_col = st.columns(2)

        date_col.write(
            "**Image Date:** "
            + image_datetime.strftime("%d %B %Y")
        )

        time_col.write(
            "**Image Time:** "
            + image_datetime.strftime("%H:%M:%S IST")
        )

    else:

        st.warning(
            "Timestamp unavailable in filename. "
            "Printed-image OCR is not enabled."
        )

    if image_id is not None:

        st.write(
            f"**Image ID:** {image_id}"
        )

    if status == "FAULT":

        st.error(
            "🔴 FAULT — ATTENTION REQUIRED WITHIN 2 DAYS"
        )

    else:

        st.success("🟢 NO FAULT")

    st.write(
        f"**Recommendation:** {recommendation}"
    )

    st.caption(
        f"Temperature source: {source}"
    )

    if reference is not None:

        st.info(
            "OEM reference: "
            f"Maximum {reference['max']:.1f} °C | "
            f"Minimum {reference['min']:.1f} °C"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Analysis Settings")

    st.metric(
        "Fault Threshold",
        "5.0 °C"
    )

    st.write("ΔT < 5°C → NO FAULT")

    st.write("ΔT ≥ 5°C → FAULT")

    st.write(
        "FAULT → Attention required within 2 days"
    )

    st.divider()

    st.subheader("Calibration")

    calibration_folder = "calibration_images"

    if os.path.isdir(calibration_folder):

        count = len([
            filename
            for filename in os.listdir(
                calibration_folder
            )
            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])

        st.write(
            f"Calibration images: {count}"
        )

    else:

        st.warning(
            "calibration_images folder not found."
        )


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

try:

    initialize_database()

    database_ready = True

except Exception as error:

    database_ready = False

    st.error(
        "Database connection failed. "
        "Check your DATABASE_URL in Streamlit Secrets."
    )

    st.caption(str(error))


# ============================================================
# ANALYSIS FORM
# ============================================================

st.header("📤 Upload Thermal Images")

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
# PROCESS UPLOADS ONLY AFTER FORM SUBMISSION
# ============================================================

if analyze_button:

    if not database_ready:

        st.error(
            "Cannot save results because the database "
            "is unavailable. No analysis was submitted."
        )

    elif not uploaded_files:

        st.warning(
            "Please upload at least one image."
        )

    else:

        model = train_calibration_model()

        results = []

        for uploaded_file in uploaded_files:

            image_bytes = uploaded_file.getvalue()

            try:

                image = Image.open(
                    io.BytesIO(image_bytes)
                ).convert("RGB")

            except Exception as error:

                st.error(
                    f"Could not read "
                    f"{uploaded_file.name}: {error}"
                )

                continue

            st.divider()

            st.subheader(
                f"📷 {uploaded_file.name}"
            )

            image_datetime, image_id = (
                extract_image_timestamp(
                    uploaded_file.name
                )
            )

            image_col, result_col = st.columns(
                [1.1, 1.5]
            )

            with image_col:

                st.image(
                    image,
                    caption="Uploaded Thermal Image",
                    use_container_width=True
                )

            reference = find_oem_reference(
                uploaded_file.name
            )

            minimum = None
            maximum = None
            source = "Temperature unavailable"

            with result_col:

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
                        "These values are the stored OEM "
                        "reference readings for this filename."
                    )

                else:

                    features, region = extract_features(
                        image
                    )

                    if model is None:

                        st.warning(
                            "No calibration model is available. "
                            "The image will be stored, but "
                            "temperatures cannot be estimated."
                        )

                    else:

                        minimum, maximum = (
                            predict_temperature(
                                features,
                                model
                            )
                        )

                        source = (
                            "Empirical estimate using "
                            f"{model['samples']} OEM images"
                        )

                        display_result(
                            minimum,
                            maximum,
                            source,
                            image_datetime,
                            image_id
                        )

                        st.warning(
                            "These temperatures are empirical "
                            "estimates from image colours, not "
                            "direct radiometric measurements. "
                            "Validate them before operational use."
                        )

                    with st.expander(
                        "View region used for image analysis"
                    ):

                        st.image(
                            region,
                            caption="Image feature extraction region",
                            use_container_width=True
                        )

            # ------------------------------------------------
            # SAVE EVERY SUBMITTED IMAGE
            # ------------------------------------------------

            try:

                record_id = save_inspection(
                    filename=uploaded_file.name,
                    image_bytes=image_bytes,
                    image_datetime=image_datetime,
                    image_id=image_id,
                    minimum=minimum,
                    maximum=maximum,
                    source=source
                )

                st.success(
                    f"Saved to database. Record ID: {record_id}"
                )

            except Exception as error:

                st.error(
                    "Could not save this inspection to "
                    f"the database: {error}"
                )

            # ------------------------------------------------
            # SUMMARY DATA
            # ------------------------------------------------

            if minimum is not None and maximum is not None:

                difference = maximum - minimum

                status, recommendation = get_decision(
                    difference
                )

            else:

                difference = None
                status = "NOT ANALYZED"

                recommendation = (
                    "Temperature information unavailable."
                )

            results.append({
                "Record Image": uploaded_file.name,
                "Image ID": image_id,
                "Image Date/Time": (
                    image_datetime.isoformat()
                    if image_datetime is not None
                    else None
                ),
                "Temperature Source": source,
                "Minimum (°C)": minimum,
                "Maximum (°C)": maximum,
                "Difference (°C)": difference,
                "Decision": status,
                "Recommendation": recommendation
            })

        # ----------------------------------------------------
        # CURRENT BATCH SUMMARY
        # ----------------------------------------------------

        if results:

            st.divider()

            st.header("📊 Current Batch Summary")

            result_df = pd.DataFrame(results)

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download Analysis Report",
                data=result_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="KRCL_Thermal_Analysis_Report.csv",
                mime="text/csv"
            )


# ============================================================
# INSPECTION HISTORY
# ============================================================

st.divider()

st.header("📚 Inspection History")

st.write(
    "View previously saved images and their inspection records."
)

if st.button("Refresh Inspection History"):

    if not database_ready:

        st.error(
            "Database is unavailable."
        )

    else:

        try:

            connection = get_database_connection()

            try:

                with connection.cursor() as cursor:

                    cursor.execute(
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
                            temperature_source
                        FROM thermal_inspections
                        ORDER BY uploaded_at DESC;
                        """
                    )

                    rows = cursor.fetchall()

                    columns = [
                        "Record ID",
                        "Image",
                        "Image ID",
                        "Image Date/Time",
                        "Uploaded At",
                        "Minimum (°C)",
                        "Maximum (°C)",
                        "Difference (°C)",
                        "Mean (°C)",
                        "Decision",
                        "Recommendation",
                        "Temperature Source"
                    ]

                    history_df = pd.DataFrame(
                        rows,
                        columns=columns
                    )

            finally:

                connection.close()

            st.session_state["inspection_history"] = (
                history_df
            )

        except Exception as error:

            st.error(
                f"Could not load inspection history: {error}"
            )


if "inspection_history" in st.session_state:

    history_df = st.session_state[
        "inspection_history"
    ]

    if history_df.empty:

        st.info(
            "No inspection records found."
        )

    else:

        filter_options = [
            "All",
            "FAULT",
            "NO FAULT",
            "NOT ANALYZED"
        ]

        selected_status = st.selectbox(
            "Filter by decision",
            filter_options
        )

        if selected_status != "All":

            filtered_df = history_df[
                history_df["Decision"] == selected_status
            ]

        else:

            filtered_df = history_df

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Inspection History",
            data=filtered_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="KRCL_Inspection_History.csv",
            mime="text/csv"
        )

        st.subheader("Retrieve Stored Image")

        selected_record = st.number_input(
            "Enter Record ID",
            min_value=1,
            step=1
        )

        if st.button("View Stored Image"):

            try:

                connection = get_database_connection()

                try:

                    with connection.cursor() as cursor:

                        cursor.execute(
                            """
                            SELECT image_filename, image_data
                            FROM thermal_inspections
                            WHERE id = %s;
                            """,
                            (int(selected_record),)
                        )

                        stored_row = cursor.fetchone()

                finally:

                    connection.close()

                if stored_row is None:

                    st.warning(
                        "No image exists for that record ID."
                    )

                else:

                    stored_filename = stored_row[0]

                    stored_bytes = bytes(
                        stored_row[1]
                    )

                    st.write(
                        f"**Original filename:** {stored_filename}"
                    )

                    stored_image = Image.open(
                        io.BytesIO(stored_bytes)
                    )

                    st.image(
                        stored_image,
                        caption=stored_filename,
                        use_container_width=True
                    )

            except Exception as error:

                st.error(
                    f"Could not retrieve stored image: {error}"
                )


# ============================================================
# OEM VALIDATION TABLE
# ============================================================

st.divider()

st.header("🔬 OEM Reference Dataset")

validation = []

for image_id, values in OEM_DATA.items():

    difference = (
        values["max"] - values["min"]
    )

    status, _ = get_decision(
        difference
    )

    validation.append({
        "Image ID": image_id,
        "OEM Maximum (°C)": values["max"],
        "OEM Minimum (°C)": values["min"],
        "OEM Difference (°C)": round(
            difference,
            1
        ),
        "Decision": status
    })

validation_df = pd.DataFrame(validation)

st.dataframe(
    validation_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ENGINEERING NOTE
# ============================================================

st.divider()

st.subheader("⚠️ Engineering Note")

st.write(
    "The application uses OEM reference values when the "
    "uploaded filename matches a known reference. For other "
    "images, it can produce empirical estimates if a calibration "
    "model is available."
)

st.write(
    "A normal colourized thermal JPEG does not necessarily "
    "contain the original radiometric temperature matrix. "
    "Actual temperature extraction requires suitable camera "
    "data or a validated measurement method."
)

st.write(
    "The 5°C decision threshold is a configured project rule. "
    "Operational use must follow the applicable approved "
    "railway inspection procedure."
)

