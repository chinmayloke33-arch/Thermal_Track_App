
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import io
import os
import re


# ============================================================
# KRCL THERMAL TRACK ANALYZER
# IT DEPARTMENT - APPRENTICESHIP PROJECT
# ============================================================

st.set_page_config(
    page_title="KRCL Thermal Track Analyzer",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ KRCL Thermal Track Analyzer")
st.caption("IT Department • Thermal Image Analysis & OEM Validation")


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


FAULT_THRESHOLD = 5.0


# ============================================================
# FILENAME MATCHING
# ============================================================

def clean_filename(filename):

    name = os.path.splitext(
        os.path.basename(filename)
    )[0]

    name = re.sub(r"\s+", "", name)

    return name


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

    image = np.array(
        image.convert("RGB")
    )

    height, width = image.shape[:2]

    # Remove common borders / scale region.
    left = int(width * 0.03)
    right = int(width * 0.88)

    top = int(height * 0.08)
    bottom = int(height * 0.92)

    region = image[
        top:bottom,
        left:right
    ]

    return region


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

    # Remove very dark / nearly grey pixels.
    mask = (
        (V > 25) &
        (S > 20)
    )

    if np.sum(mask) < 100:
        mask = np.ones(V.shape, dtype=bool)

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

    return np.array(features), region


# ============================================================
# RIDGE CALIBRATION
# ============================================================

def train_calibration_model():

    folder = "calibration_images"

    if not os.path.exists(folder):
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

            image = Image.open(
                path
            ).convert("RGB")

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

    X = np.array(X, dtype=float)

    Y_MAX = np.array(
        Y_MAX,
        dtype=float
    )

    Y_MIN = np.array(
        Y_MIN,
        dtype=float
    )

    # Standardization
    mean = X.mean(axis=0)

    std = X.std(axis=0)

    std[std < 1e-9] = 1

    X_scaled = (
        X - mean
    ) / std

    # Ridge regression
    alpha = 10.0

    identity = np.eye(
        X_scaled.shape[1]
    )

    coef_max = np.linalg.solve(
        X_scaled.T @ X_scaled +
        alpha * identity,

        X_scaled.T @ Y_MAX
    )

    coef_min = np.linalg.solve(
        X_scaled.T @ X_scaled +
        alpha * identity,

        X_scaled.T @ Y_MIN
    )

    intercept_max = (
        np.mean(Y_MAX)
        -
        np.mean(X_scaled, axis=0)
        @ coef_max
    )

    intercept_min = (
        np.mean(Y_MIN)
        -
        np.mean(X_scaled, axis=0)
        @ coef_min
    )

    return {

        "mean": mean,

        "std": std,

        "coef_max": coef_max,

        "coef_min": coef_min,

        "intercept_max":
            intercept_max,

        "intercept_min":
            intercept_min,

        "samples":
            len(X)
    }


# ============================================================
# PREDICTION
# ============================================================

def predict_temperature(
    features,
    model
):

    X = (
        features -
        model["mean"]
    ) / model["std"]

    predicted_max = (
        model["intercept_max"]
        +
        X @ model["coef_max"]
    )

    predicted_min = (
        model["intercept_min"]
        +
        X @ model["coef_min"]
    )

    predicted_max = float(
        predicted_max
    )

    predicted_min = float(
        predicted_min
    )

    # Make sure minimum <= maximum.
    if predicted_min > predicted_max:

        predicted_min, predicted_max = (
            predicted_max,
            predicted_min
        )

    return (
        predicted_min,
        predicted_max
    )


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
# DISPLAY RESULT
# ============================================================

def display_result(
    minimum,
    maximum,
    source,
    reference=None
):

    difference = (
        maximum -
        minimum
    )

    mean_temperature = (
        minimum +
        maximum
    ) / 2

    status, message = get_decision(
        difference
    )

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

    if status == "FAULT":

        st.error(
            "🔴 FAULT — "
            "ATTENTION REQUIRED WITHIN 2 DAYS"
        )

    else:

        st.success(
            "🟢 NO FAULT"
        )

    st.caption(
        f"Temperature source: {source}"
    )

    if reference is not None:

        reference_difference = (
            reference["max"]
            -
            reference["min"]
        )

        st.info(
            "OEM Reference: "
            f"Maximum = {reference['max']:.1f} °C | "
            f"Minimum = {reference['min']:.1f} °C | "
            f"Difference = {reference_difference:.1f} °C"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Analysis Settings"
    )

    st.metric(
        "Fault Threshold",
        "5.0 °C"
    )

    st.divider()

    st.subheader(
        "Decision Rule"
    )

    st.write(
        "ΔT < 5°C → NO FAULT"
    )

    st.write(
        "ΔT ≥ 5°C → FAULT"
    )

    st.write(
        "Fault → Attention required within 2 days"
    )

    st.divider()

    st.subheader(
        "Calibration"
    )

    if os.path.exists(
        "calibration_images"
    ):

        count = len([
            f
            for f in os.listdir(
                "calibration_images"
            )
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])

        st.write(
            f"Calibration images: **{count}**"
        )

    else:

        st.warning(
            "calibration_images folder not found."
        )


# ============================================================
# UPLOAD
# ============================================================

uploaded_files = st.file_uploader(

    "Upload thermal image(s)",

    type=[
        "jpg",
        "jpeg",
        "png"
    ],

    accept_multiple_files=True
)


if not uploaded_files:

    st.info(
        "Upload one or more thermal images."
    )

    st.markdown(
        "### Fault Classification"
    )

    st.write(
        "**ΔT < 5°C → 🟢 NO FAULT**"
    )

    st.write(
        "**ΔT ≥ 5°C → 🔴 FAULT**"
    )

    st.warning(
        "Important: ordinary colourized thermal "
        "JPEG images do not necessarily contain "
        "the original camera radiometric temperature "
        "matrix."
    )

    st.stop()


# ============================================================
# TRAIN CALIBRATION MODEL
# ============================================================

model = train_calibration_model()


# ============================================================
# PROCESS IMAGES
# ============================================================

results = []


for uploaded_file in uploaded_files:

    try:

        image = Image.open(
            io.BytesIO(
                uploaded_file.getvalue()
            )
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

    image_col, result_col = st.columns(
        [1.2, 1]
    )

    with image_col:

        st.image(
            image,
            caption="Thermal Image",
            width="stretch"
        )

    reference = find_oem_reference(
        uploaded_file.name
    )


    # ========================================================
    # EXACT OEM IMAGE
    # ========================================================

    if reference is not None:

        with result_col:

            st.success(
                "OEM reference found."
            )

            display_result(

                reference["min"],

                reference["max"],

                "Actual OEM / Department reading",

                reference
            )

        difference = (
            reference["max"]
            -
            reference["min"]
        )

        results.append({

            "Image":
                uploaded_file.name,

            "Source":
                "OEM Reference",

            "Minimum (°C)":
                reference["min"],

            "Maximum (°C)":
                reference["max"],

            "Difference (°C)":
                round(
                    difference,
                    1
                ),

            "Decision":
                get_decision(
                    difference
                )[0]
        })

        continue


    # ========================================================
    # NEW IMAGE
    # ========================================================

    features, region = extract_features(
        image
    )

    with result_col:

        if model is None:

            st.warning(
                "No calibration model available."
            )

            st.write(
                "Place the 12 OEM images inside "
                "`calibration_images/` to enable "
                "calibrated estimation."
            )

        else:

            minimum, maximum = (
                predict_temperature(
                    features,
                    model
                )
            )

            # Keep predictions inside the
            # observed OEM temperature range.
            observed_min = min(
                v["min"]
                for v in OEM_DATA.values()
            )

            observed_max = max(
                v["max"]
                for v in OEM_DATA.values()
            )

            minimum = float(
                np.clip(
                    minimum,
                    observed_min,
                    observed_max
                )
            )

            maximum = float(
                np.clip(
                    maximum,
                    observed_min,
                    observed_max
                )
            )

            display_result(

                minimum,

                maximum,

                f"Calibrated estimate "
                f"using {model['samples']} OEM images"
            )

            st.warning(
                "This is an empirical estimate. "
                "It is not a direct radiometric "
                "temperature measurement."
            )

            difference = (
                maximum -
                minimum
            )

            results.append({

                "Image":
                    uploaded_file.name,

                "Source":
                    "Calibrated Estimate",

                "Minimum (°C)":
                    round(
                        minimum,
                        1
                    ),

                "Maximum (°C)":
                    round(
                        maximum,
                        1
                    ),

                "Difference (°C)":
                    round(
                        difference,
                        1
                    ),

                "Decision":
                    get_decision(
                        difference
                    )[0]
            })


    with st.expander(
        "View image region used for analysis"
    ):

        st.image(
            region,
            caption=(
                "Region used for "
                "colour-feature extraction"
            ),
            width="stretch"
        )


# ============================================================
# SUMMARY
# ============================================================

if results:

    st.divider()

    st.header(
        "📊 Analysis Summary"
    )

    result_df = pd.DataFrame(
        results
    )

    st.dataframe(
        result_df,
        width="stretch",
        hide_index=True
    )

    csv = result_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        "⬇️ Download Analysis Report",

        data=csv,

        file_name=(
            "KRCL_Thermal_Analysis_Report.csv"
        ),

        mime="text/csv"
    )


# ============================================================
# OEM VALIDATION TABLE
# ============================================================

st.divider()

st.header(
    "🔬 OEM Validation Dataset"
)

validation = []

for image_id, values in OEM_DATA.items():

    difference = (
        values["max"]
        -
        values["min"]
    )

    status = get_decision(
        difference
    )[0]

    validation.append({

        "Image ID":
            image_id,

        "OEM Maximum (°C)":
            values["max"],

        "OEM Minimum (°C)":
            values["min"],

        "OEM Difference (°C)":
            round(
                difference,
                1
            ),

        "Decision":
            status
    })


validation_df = pd.DataFrame(
    validation
)

st.dataframe(
    validation_df,
    width="stretch",
    hide_index=True
)


st.info(
    "The OEM values above are treated as the "
    "reference measurements supplied for this project."
)


# ============================================================
# IMPORTANT ENGINEERING NOTE
# ============================================================

st.divider()

st.subheader(
    "⚠️ Engineering Note"
)

st.write(
    "For operational deployment, the preferred input "
    "is the original radiometric data exported by the "
    "thermal camera. A normal colourized JPEG does not "
    "necessarily contain the underlying temperature matrix."
)

st.write(
    "The current application therefore distinguishes "
    "between actual OEM reference values and empirical "
    "estimates for previously unseen images."
)

