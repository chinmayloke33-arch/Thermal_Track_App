
import streamlit as st
from PIL import Image
import numpy as np
import cv2

# ===================================================
# PAGE CONFIGURATION
# ===================================================

st.set_page_config(
    page_title="Thermal Image Analyzer",
    page_icon="🌡️",
    layout="wide"
)

# ===================================================
# TITLE
# ===================================================

st.title("🌡️ Thermal Image Analyzer")

st.write(
    "Upload a thermal image to calculate the minimum "
    "and maximum temperature and determine whether a "
    "temperature fault is present."
)

# ===================================================
# TEMPERATURE SETTINGS
# ===================================================

# Based on the temperature range provided by the
# Electrical Department.

MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 40.0

# Fault threshold
FAULT_THRESHOLD = 5.0

# ===================================================
# IMAGE UPLOAD
# ===================================================

uploaded_file = st.file_uploader(
    "Upload thermal image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "tif",
        "tiff"
    ]
)

# ===================================================
# IMAGE PROCESSING
# ===================================================

if uploaded_file is not None:

    # ------------------------------------------------
    # Read image
    # ------------------------------------------------

    image = Image.open(uploaded_file).convert("RGB")

    img = np.array(image)

    # ------------------------------------------------
    # Display image
    # ------------------------------------------------

    st.subheader("📷 Uploaded Thermal Image")

    st.image(
        image,
        use_container_width=True
    )

    # ------------------------------------------------
    # Convert RGB image to grayscale
    # ------------------------------------------------

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    # ------------------------------------------------
    # Normalize grayscale values
    # ------------------------------------------------

    gray_normalized = (
        gray.astype(np.float32) / 255.0
    )

    # ------------------------------------------------
    # Convert pixel values to temperature
    #
    # Current project range:
    # 0°C to 40°C
    # ------------------------------------------------

    temperature = (
        MIN_TEMPERATURE
        + gray_normalized
        * (MAX_TEMPERATURE - MIN_TEMPERATURE)
    )

    # Keep values inside 0–40°C
    temperature = np.clip(
        temperature,
        MIN_TEMPERATURE,
        MAX_TEMPERATURE
    )

    # =================================================
    # WHOLE IMAGE ANALYSIS
    # =================================================

    min_temp = float(
        np.min(temperature)
    )

    max_temp = float(
        np.max(temperature)
    )

    mean_temp = float(
        np.mean(temperature)
    )

    difference = (
        max_temp - min_temp
    )

    # =================================================
    # TEMPERATURE RESULTS
    # =================================================

    st.divider()

    st.subheader("🌡️ Temperature Results")

    c1, c2, c3, c4 = st.columns(4)

    # ------------------------------------------------
    # Minimum temperature
    # ------------------------------------------------

    with c1:

        st.metric(
            "Minimum Temperature",
            f"{min_temp:.2f} °C"
        )

    # ------------------------------------------------
    # Maximum temperature
    # ------------------------------------------------

    with c2:

        st.metric(
            "Maximum Temperature",
            f"{max_temp:.2f} °C"
        )

    # ------------------------------------------------
    # Temperature difference
    # ------------------------------------------------

    with c3:

        st.metric(
            "Temperature Difference",
            f"{difference:.2f} °C"
        )

    # ------------------------------------------------
    # Mean temperature
    # ------------------------------------------------

    with c4:

        st.metric(
            "Mean Temperature",
            f"{mean_temp:.2f} °C"
        )

    # =================================================
    # FAULT ASSESSMENT
    # =================================================

    st.divider()

    st.subheader("🚦 Fault Assessment")

    if difference < FAULT_THRESHOLD:

        st.success(
            f"✅ NO FAULT\n\n"
            f"Temperature difference = "
            f"{difference:.2f} °C"
        )

        st.write(
            "**Status:** No fault detected."
        )

        st.info(
            "The temperature difference is below "
            "the 5°C fault threshold."
        )

    else:

        st.error(
            f"⚠️ FAULT DETECTED\n\n"
            f"Temperature difference = "
            f"{difference:.2f} °C"
        )

        st.warning(
            "Attention required within 2 days."
        )

        st.write(
            "**Status:** Fault detected."
        )

    # =================================================
    # ANALYSIS SUMMARY
    # =================================================

    st.divider()

    st.subheader("📊 Analysis Summary")

    status = (
        "NO FAULT"
        if difference < FAULT_THRESHOLD
        else "FAULT"
    )

    required_action = (
        "No action required"
        if difference < FAULT_THRESHOLD
        else "Attention required within 2 days"
    )

    results = {

        "Temperature Range":
            "0.00 – 40.00 °C",

        "Minimum Temperature":
            f"{min_temp:.2f} °C",

        "Maximum Temperature":
            f"{max_temp:.2f} °C",

        "Temperature Difference":
            f"{difference:.2f} °C",

        "Mean Temperature":
            f"{mean_temp:.2f} °C",

        "Fault Threshold":
            "5.00 °C",

        "Status":
            status,

        "Required Action":
            required_action
    }

    for key, value in results.items():

        st.write(
            f"**{key}:** {value}"
        )

# ===================================================
# INFORMATION
# ===================================================

else:

    st.info(
        "Please upload a thermal image to begin the analysis."
    )

    st.write(
        "**Temperature range:** 0–40 °C"
    )

    st.write(
        "**Fault threshold:** 5 °C"
    )


