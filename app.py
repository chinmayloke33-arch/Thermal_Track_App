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
    "Upload a thermal image to perform actual temperature calculations, "
    "spot-calibration mapping, and operational fault assessment."
)

# ============================================================
# SIDEBAR: ACTUAL TEMPERATURE CALIBRATION
# ============================================================

st.sidebar.header("⚙️ Actual Spot Calibration")
st.sidebar.write("Enter the known temperature bounds displayed on your camera overlay:")

known_cold = st.sidebar.number_input(
    "Spot Minimum Temp (°C)",
    value=7.0,
    step=0.1,
    format="%.2f",
    help="Cold spot or minimum scale temperature from camera legend."
)

known_hot = st.sidebar.number_input(
    "Spot Maximum Temp (°C)",
    value=40.0,
    step=0.1,
    format="%.2f",
    help="Hot spot or maximum scale temperature from camera legend."
)

fault_thresh = st.sidebar.number_input(
    "Fault Threshold (°C)",
    value=5.0,
    step=0.5,
    format="%.2f",
    help="Temperature difference required to trigger a fault."
)

# Validation check
if known_hot <= known_cold:
    st.sidebar.error("Maximum temperature must be strictly greater than minimum temperature.")

# ============================================================
# IMAGE UPLOAD SECTION
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Thermal Image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

# ============================================================
# MAIN ANALYSIS LOGIC
# ============================================================

if uploaded_file is not None:

    # 1. Load Image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # 2. Display Image
    col_img, col_info = st.columns([1, 1])

    with col_img:
        st.subheader("📷 Uploaded Thermal Image")
        st.image(image, use_container_width=True)

    # 3. Calculate Actual Temperatures
    try:
        results = analyze_thermal_image(
            image=img_array,
            known_cold_temp=known_cold,
            known_hot_temp=known_hot,
            threshold=fault_thresh
        )
    except Exception as err:
        st.error(f"Analysis error: {str(err)}")
        st.stop()

    # 4. Metric Displays
    st.divider()
    st.subheader("🌡️ Actual Temperature Results")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            label="Actual Minimum Temp",
            value=f"{results['min_temperature']:.2f} °C"
        )

    with c2:
        st.metric(
            label="Actual Maximum Temp",
            value=f"{results['max_temperature']:.2f} °C"
        )

    with c3:
        st.metric(
            label="Temperature Difference (ΔT)",
            value=f"{results['temperature_difference']:.2f} °C"
        )

    with c4:
        st.metric(
            label="Average Target Temp",
            value=f"{results['mean_temperature']:.2f} °C"
        )

    # 5. Fault Assessment Output
    st.divider()
    st.subheader("🚦 Fault Assessment")

    if results["status"] == "NO FAULT":
        st.success(
            f"✅ **NO FAULT DETECTED**\n\n"
            f"Calculated Temperature Difference = **{results['temperature_difference']:.2f} °C** "
            f"(Threshold: {results['threshold']:.2f} °C)"
        )
        st.info("System operating within acceptable thermal parameters.")
    else:
        st.error(
            f"⚠️ **FAULT DETECTED**\n\n"
            f"Calculated Temperature Difference = **{results['temperature_difference']:.2f} °C** "
            f"(Exceeds Threshold of {results['threshold']:.2f} °C)"
        )
        st.warning(f"**Action Required:** {results['action']}")

    # 6. Executive Summary Table
    st.divider()
    st.subheader("📊 Operational Summary")

    summary_data = {
        "Calibrated Cold Spot": f"{known_cold:.2f} °C",
        "Calibrated Hot Spot": f"{known_hot:.2f} °C",
        "Measured Minimum": f"{results['min_temperature']:.2f} °C",
        "Measured Maximum": f"{results['max_temperature']:.2f} °C",
        "Temperature Delta (ΔT)": f"{results['temperature_difference']:.2f} °C",
        "Average Temperature": f"{results['mean_temperature']:.2f} °C",
        "Configured Threshold": f"{results['threshold']:.2f} °C",
        "Diagnosis": results["status"],
        "Recommended Action": results["action"]
    }

    for key, val in summary_data.items():
        st.write(f"**{key}:** {val}")

else:
    st.info("Please upload a thermal image above to compute actual temperatures.")
    st.write("---")
    st.write("### How to get exact readings:")
    st.write("1. Check the temperature values displayed on your thermal camera screen legend.")
    st.write("2. Enter those numbers into the sidebar inputs (`Spot Minimum Temp` and `Spot Maximum Temp`).")
    st.write("3. Upload the image to view the exact calculated temperature delta across target equipment.")
