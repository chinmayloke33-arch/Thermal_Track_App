
import streamlit as st
from PIL import Image
import numpy as np
import cv2

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Thermal Image Analyzer",
    page_icon="🌡️",
    layout="wide"
)

# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("🌡️ Thermal Image Analyzer")

st.write(
    "Upload a thermal image to calculate the minimum temperature, "
    "maximum temperature, temperature difference, and fault status."
)

# ============================================================
# TEMPERATURE SETTINGS
# ============================================================

# Temporary calibration range based on the Electrical Department
# results provided for this project.

MIN_TEMPERATURE = 7.0
MAX_TEMPERATURE = 40.0

# Fault threshold
FAULT_THRESHOLD = 5.0

# Percentiles used to remove extreme image pixels
LOW_PERCENTILE = 2
HIGH_PERCENTILE = 98

# ============================================================
# IMAGE UPLOAD
# ============================================================

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

# ============================================================
# MAIN ANALYSIS
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = Image.open(uploaded_file).convert("RGB")

    img = np.array(image)

    # --------------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------------

    st.subheader("📷 Uploaded Thermal Image")

    st.image(
        image,
        use_container_width=True
    )

    # ========================================================
    # IMAGE PROCESSING
    # ========================================================

    # Convert RGB image to grayscale
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    # Convert pixels to floating point
    gray_float = gray.astype(np.float32)

    # --------------------------------------------------------
    # REMOVE EXTREME PIXELS
    # --------------------------------------------------------
    #
    # Instead of using the absolute darkest and brightest
    # pixels, use the 2nd and 98th percentile.
    #
    # This prevents a single black or white pixel, border,
    # text, or image artifact from becoming Tmin or Tmax.
    # --------------------------------------------------------

    low_pixel = np.percentile(
        gray_float,
        LOW_PERCENTILE
    )

    high_pixel = np.percentile(
        gray_float,
        HIGH_PERCENTILE
    )

    # Prevent division by zero
    if high_pixel <= low_pixel:

        st.error(
            "The uploaded image does not contain enough "
            "temperature variation for analysis."
        )

        st.stop()

    # ========================================================
    # CONVERT IMAGE VALUES TO TEMPERATURE
    # ========================================================

    temperature = (
        MIN_TEMPERATURE
        +
        (
            (gray_float - low_pixel)
            /
            (high_pixel - low_pixel)
        )
        *
        (MAX_TEMPERATURE - MIN_TEMPERATURE)
    )

    # Keep temperature inside the selected range
    temperature = np.clip(
        temperature,
        MIN_TEMPERATURE,
        MAX_TEMPERATURE
    )

    # ========================================================
    # TEMPERATURE CALCULATIONS
    # ========================================================

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

    # ========================================================
    # TEMPERATURE RESULTS
    # ========================================================

    st.divider()

    st.subheader("🌡️ Temperature Results")

    c1, c2, c3, c4 = st.columns(4)

    # --------------------------------------------------------
    # MINIMUM
    # --------------------------------------------------------

    with c1:

        st.metric(
            "Minimum Temperature",
            f"{min_temp:.2f} °C"
        )

    # --------------------------------------------------------
    # MAXIMUM
    # --------------------------------------------------------

    with c2:

        st.metric(
            "Maximum Temperature",
            f"{max_temp:.2f} °C"
        )

    # --------------------------------------------------------
    # DIFFERENCE
    # --------------------------------------------------------

    with c3:

        st.metric(
            "Temperature Difference",
            f"{difference:.2f} °C"
        )

    # --------------------------------------------------------
    # MEAN
    # --------------------------------------------------------

    with c4:

        st.metric(
            "Mean Temperature",
            f"{mean_temp:.2f} °C"
        )

    # ========================================================
    # FAULT ASSESSMENT
    # ========================================================

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
            "Temperature difference is below "
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

    # ========================================================
    # ANALYSIS SUMMARY
    # ========================================================

    st.divider()

    st.subheader("📊 Analysis Summary")

    if difference < FAULT_THRESHOLD:

        status = "NO FAULT"

        required_action = (
            "No action required"
        )

    else:

        status = "FAULT"

        required_action = (
            "Attention required within 2 days"
        )

    results = {

        "Temperature Range":
            "7.00 – 40.00 °C",

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

    # ========================================================
    # ANALYSIS INFORMATION
    # ========================================================

    st.divider()

    st.subheader("ℹ️ Analysis Information")

    st.write(
        f"Temperature calibration range: "
        f"{MIN_TEMPERATURE:.1f}–"
        f"{MAX_TEMPERATURE:.1f} °C"
    )

    st.write(
        f"Pixel analysis range: "
        f"{LOW_PERCENTILE}th–"
        f"{HIGH_PERCENTILE}th percentile"
    )

    st.write(
        "Extreme pixels are ignored to reduce the effect "
        "of image borders, text, noise, and isolated pixels."
    )

# ============================================================
# NO IMAGE
# ============================================================

else:

    st.info(
        "Please upload a thermal image to begin the analysis."
    )

    st.write(
        "**Current temperature range:** 7–40 °C"
    )

    st.write(
        "**Fault threshold:** 5 °C"
    )

    st.write(
        "**Fault action:** Attention required within 2 days."
    )


