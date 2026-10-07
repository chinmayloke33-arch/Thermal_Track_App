import streamlit as st
from thermal_analyzer import analyze_thermal_image_radiometric

st.set_page_config(
    page_title="Thermal Image Analyzer",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Thermal Image Analyzer")
st.write("Automated thermal component analysis with dynamic scale syncing.")

# ============================================================
# BENCHMARK DATASET LOOKUP (Matches image table)
# ============================================================

BENCHMARKS = {
    "Custom / Manual Entry": {"min": 7.0, "max": 40.0},
    "20250315-133514-016 (ΔT: 1.9°C)": {"min": 37.6, "max": 39.5},
    "20251111-112915-003 (ΔT: 9.6°C)": {"min": 21.9, "max": 31.5},
    "20251111-114541-001 (ΔT: 28.4°C)": {"min": 7.2, "max": 35.6},
    "20251111-114605-002 (ΔT: 8.0°C)": {"min": 27.4, "max": 35.4},
    "20251113-074547-002 (ΔT: 3.9°C)": {"min": 13.7, "max": 17.6},
    "20251113-091808-012 (ΔT: 5.4°C)": {"min": 19.3, "max": 24.7},
    "20251113-092707-005 (ΔT: 3.9°C)": {"min": 20.5, "max": 24.4},
    "20251113-093327-012 (ΔT: 5.7°C)": {"min": 18.3, "max": 24.0},
    "20251113-113151-016 (ΔT: 9.8°C)": {"min": 24.7, "max": 34.5},
    "20251114-101128-007 (ΔT: 10.4°C)": {"min": 15.8, "max": 26.2},
    "20251114-114841-018 (ΔT: 21.4°C)": {"min": 13.6, "max": 35.0},
    "20251117-124611-004 (ΔT: 11.0°C)": {"min": 22.8, "max": 33.8},
}

st.sidebar.header("⚙️ OEM Scale Calibration")

# Callback to sync inputs when dropdown changes
def update_scale_values():
    selected = st.session_state["preset_select"]
    st.session_state["min_temp_input"] = BENCHMARKS[selected]["min"]
    st.session_state["max_temp_input"] = BENCHMARKS[selected]["max"]

# Initialize session state if first run
if "min_temp_input" not in st.session_state:
    st.session_state["min_temp_input"] = 7.0
if "max_temp_input" not in st.session_state:
    st.session_state["max_temp_input"] = 40.0

st.sidebar.selectbox(
    "Select Target Image Preset",
    list(BENCHMARKS.keys()),
    key="preset_select",
    on_change=update_scale_values
)

known_cold = st.sidebar.number_input(
    "Scale Min Temp (°C)",
    key="min_temp_input",
    step=0.1,
    format="%.1f"
)

known_hot = st.sidebar.number_input(
    "Scale Max Temp (°C)",
    key="max_temp_input",
    step=0.1,
    format="%.1f"
)

fault_thresh = st.sidebar.number_input(
    "Fault Threshold (°C)",
    value=5.0,
    step=0.5,
    format="%.1f"
)

# ============================================================
# FILE UPLOAD & PROCESSING
# ============================================================

uploaded_file = st.file_uploader("Upload Thermal Image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()

    st.divider()
    st.image(uploaded_file, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)

    try:
        results = analyze_thermal_image_radiometric(
            file_bytes=file_bytes,
            threshold=fault_thresh,
            known_cold_temp=known_cold,
            known_hot_temp=known_hot
        )

        st.divider()
        st.subheader("📊 Calibrated Temperature Results")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Baseline Temp (T_min)", f"{results['min_temperature']:.1f} °C")
        with c2:
            st.metric("Hotspot Temp (T_max)", f"{results['max_temperature']:.1f} °C")
        with c3:
            st.metric("Calculated Delta (ΔT)", f"{results['temperature_difference']:.1f} °C")
        with c4:
            st.metric("Mean Component Temp", f"{results['mean_temperature']:.1f} °C")

        st.divider()
        st.subheader("🚦 Fault Status")

        if results["status"] == "FAULT":
            st.error(
                f"⚠️ **FAULT DETECTED**\n\n"
                f"Calculated $\Delta T$ = **{results['temperature_difference']:.1f} °C** "
                f"(Exceeds threshold of {results['threshold']:.1f} °C)"
            )
        else:
            st.success(
                f"✅ **NO FAULT DETECTED**\n\n"
                f"Calculated $\Delta T$ = **{results['temperature_difference']:.1f} °C** "
                f"(Within normal limits < {results['threshold']:.1f} °C)"
            )

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

else:
    st.info("Upload a thermal image and select its preset from the sidebar.")
