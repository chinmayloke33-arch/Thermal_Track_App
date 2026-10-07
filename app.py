import streamlit as st
from PIL import Image
import numpy as np
from thermal_analyzer import analyze_thermal_image

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Thermal Image Analyzer",
    page_icon="🌡️",
    layout="wide"
)

# ============================================================
# APPLICATION TITLE & DESCRIPTION
# ============================================================

st.title("🌡️ Thermal Image Analyzer")

st.write(
    "Upload a thermal image to perform dynamic region-of-interest (ROI) "
    "temperature estimation, hotspot analysis, and fault diagnosis."
)

# ============================================================
# SIDEBAR: THERMAL CAMERA CALIBRATION SETTINGS
# ============================================================

st.sidebar.header("⚙️ Camera Calibration")

min_scale = st.sidebar.number_input(
    "Palette Min Temp (°C)",
    value=7.0,
    step=1.0,
    help="The lowest temperature represented on your thermal camera scale."
)

max_scale = st.sidebar.number_input(
    "Palette Max Temp (°C)",
    value=40.0,
    step=1.0,
    help="The highest temperature represented on your thermal camera scale."
)

fault_thresh = st.sidebar.number_input(
    "Fault Threshold (°C)",
    value=5.0,
    step=0.5,
    help="Temperature difference required to flag an operational fault."
)

# Validate temperature range inputs
if max_scale <= min_scale:
    st.sidebar.error("Maximum temperature must be greater than minimum temperature.")

# ============================================================
# FILE UPLOAD SECTION
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Thermal Image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

# ============================================================
# MAIN APPLICATION LOGIC
# ============================================================

if uploaded_file is not None:

    # 1. Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    col_img, col_info = st.columns([1, 1])

    with col_img:
        st.subheader("📷 Uploaded Thermal Image")
        st.image(image, use_container_width=True)

    # 2. Process image using thermal_analyzer module
    try:
        results = analyze_thermal_image(
            image=img_array,
            min_scale_temp=min_scale,
            max_scale_temp=max_scale,
            threshold=fault_thresh
        )
    except Exception as err:
        st.error(f"Error processing image: {str(err)}")
        st.stop()

    # 3. Display calculated temperature metrics
    st.divider()
    st.subheader("🌡️ Temperature Analysis")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            label="Minimum Temperature",
            value=f"{results['min_temperature']:.2f} °C"
        )

    with c2:
        st.metric(
            label="Maximum Temperature",
            value=f"{results['max_temperature']:.2f} °C"
        )

    with c3:
        st.metric(
            label="Temperature Difference",
            value=f"{results['temperature_difference']:.2f} °C"
        )

    with c4:
        st.metric(
            label="Mean Temperature",
            value=f"{results['mean_temperature']:.2f} °C"
        )

    # 4. Display fault evaluation
    st.divider()
    st.subheader("🚦 Fault Assessment")

    if results["status"] == "NO FAULT":
        st.success(
            f"✅ **NO FAULT DETECTED**\n\n"
            f"Calculated Temperature Difference: **{results['temperature_difference']:.2f} °C** "
            f"(Threshold: {results['threshold']:.2f} °C)"
        )
        st.info("Operating conditions are nominal. No immediate action required.")
    else:
        st.error(
            f"⚠️ **FAULT DETECTED**\n\n"
            f"Calculated Temperature Difference: **{results['temperature_difference']:.2f} °C** "
            f"(Exceeds Threshold of {results['threshold']:.2f} °C)"
        )
        st.warning(f"**Recommended Action:** {results['action']}")

    # 5. Display analysis summary table
    st.divider()
    st.subheader("📊 Executive Summary")

    summary = {
        "Calibration Scale Range": f"{min_scale:.2f} °C to {max_scale:.2f} °C",
        "Estimated Min Temperature": f"{results['min_temperature']:.2f} °C",
        "Estimated Max Temperature": f"{results['max_temperature']:.2f} °C",
        "Active Temperature Delta": f"{results['temperature_difference']:.2f} °C",
        "Average Region Temperature": f"{results['mean_temperature']:.2f} °C",
        "Configured Fault Threshold": f"{results['threshold']:.2f} °C",
        "Evaluation Status": results["status"],
        "Required Action": results["action"]
    }

    for key, val in summary.items():
        st.write(f"**{key}:** {val}")

else:
    # Instructions displayed before file upload
    st.info("Please upload a thermal image above to generate an analysis report.")
    st.write("---")
    st.write("### Instructions:")
    st.write("1. Set the calibration scale on the left sidebar to match your thermal camera's legend limits.")
    st.write("2. Upload a thermal inspection image (`.jpg`, `.png`, `.bmp`, or `.tiff`).")
    st.write("3. Review calculated temperature differences and fault flags.")
