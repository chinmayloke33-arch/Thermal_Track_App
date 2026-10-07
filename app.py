import streamlit as st
from PIL import Image
import numpy as np
from streamlit_cropper import st_cropper
from thermal_analyzer import analyze_thermal_image

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Thermal Image Analyzer",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Thermal Image Analyzer")
st.write(
    "Upload a thermal image, crop directly around the active component "
    "(excluding background noise and scale legends), and analyze localized $\Delta T$."
)

# ============================================================
# SIDEBAR CALIBRATION
# ============================================================

st.sidebar.header("⚙️ OEM Scale Calibration")

known_cold = st.sidebar.number_input(
    "Scale Min Temp (°C)",
    value=7.0,
    step=0.1,
    format="%.2f",
    help="Minimum temperature printed on camera legend."
)

known_hot = st.sidebar.number_input(
    "Scale Max Temp (°C)",
    value=40.0,
    step=0.1,
    format="%.2f",
    help="Maximum temperature printed on camera legend."
)

fault_thresh = st.sidebar.number_input(
    "Fault Threshold (°C)",
    value=5.0,
    step=0.5,
    format="%.2f",
    help="Temperature difference required to trigger a fault."
)

if known_hot <= known_cold:
    st.sidebar.error("Maximum temperature must be greater than minimum temperature.")

# ============================================================
# FILE UPLOAD & CROPPING INTERFACE
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Thermal Image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    full_image_np = np.array(image)

    st.write("---")
    st.subheader("✂️ Interactive Component Selection")
    st.caption("Drag and adjust the red box to highlight the target component (e.g. connector, joint, rail element).")

    col_crop, col_preview = st.columns([1.2, 0.8])

    with col_crop:
        # Interactive cropping widget
        cropped_img = st_cropper(
            image,
            realtime_update=True,
            box_color="#FF0000",
            aspect_ratio=None,
            key="thermal_cropper"
        )

    with col_preview:
        st.subheader("Selected Region Preview")
        st.image(cropped_img, use_container_width=True)

    # Convert cropped selection to NumPy array
    cropped_roi_np = np.array(cropped_img)

    try:
        # Passes both full frame (for global scale) and cropped region (for local ΔT)
        results = analyze_thermal_image(
            full_image=full_image_np,
            cropped_roi=cropped_roi_np,
            known_cold_temp=known_cold,
            known_hot_temp=known_hot,
            threshold=fault_thresh
        )
    except Exception as err:
        st.error(f"Analysis error: {str(err)}")
        st.stop()

    # ============================================================
    # RESULTS DISPLAY
    # ============================================================

    st.divider()
    st.subheader("🌡️ Localized Temperature Results")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("ROI Minimum Temp", f"{results['min_temperature']:.2f} °C")
    with c2:
        st.metric("ROI Maximum Temp", f"{results['max_temperature']:.2f} °C")
    with c3:
        st.metric("Temperature Delta (ΔT)", f"{results['temperature_difference']:.2f} °C")
    with c4:
        st.metric("ROI Mean Temp", f"{results['mean_temperature']:.2f} °C")

    st.divider()
    st.subheader("🚦 Fault Diagnosis")

    if results["status"] == "NO FAULT":
        st.success(
            f"✅ **NO FAULT DETECTED**\n\n"
            f"Calculated $\Delta T$ = **{results['temperature_difference']:.2f} °C** "
            f"(Threshold: {results['threshold']:.2f} °C)"
        )
    else:
        st.error(
            f"⚠️ **FAULT DETECTED**\n\n"
            f"Calculated $\Delta T$ = **{results['temperature_difference']:.2f} °C** "
            f"(Exceeds Threshold of {results['threshold']:.2f} °C)"
        )
        st.warning(f"**Recommended Action:** {results['action']}")

else:
    st.info("Upload a thermal image above to begin interactive ROI analysis.")
