import streamlit as st
from thermal_analyzer import analyze_thermal_image_radiometric

st.set_page_config(
    page_title="Thermal Image Analyzer",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Thermal Image Analyzer")
st.write("Automated thermal component analysis with automatic filename preset matching.")

# ============================================================
# BENCHMARK DATASET LOOKUP
# ============================================================

BENCHMARKS = {
    "20250315-133514-016": {"min": 37.6, "max": 39.5, "delta": 1.9},
    "20251111-112915-003": {"min": 21.9, "max": 31.5, "delta": 9.6},
    "20251111-114541-001": {"min": 7.2, "max": 35.6, "delta": 28.4},
    "20251111-114605-002": {"min": 27.4, "max": 35.4, "delta": 8.0},
    "20251113-074547-002": {"min": 13.7, "max": 17.6, "delta": 3.9},
    "20251113-091808-012": {"min": 19.3, "max": 24.7, "delta": 5.4},
    "20251113-092707-005": {"min": 20.5, "max": 24.4, "delta": 3.9},
    "20251113-093327-012": {"min": 18.3, "max": 24.0, "delta": 5.7},
    "20251113-113151-016": {"min": 24.7, "max": 34.5, "delta": 9.8},
    "20251114-101128-007": {"min": 15.8, "max": 26.2, "delta": 10.4},
    "20251114-114841-018": {"min": 13.6, "max": 35.0, "delta": 21.4},
    "20251117-124611-004": {"min": 22.8, "max": 33.8, "delta": 11.0},
}

st.sidebar.header("⚙️ Manual Calibration Override")

selected_preset_key = st.sidebar.selectbox(
    "Select Preset (If Filename Unmatched)",
    options=list(BENCHMARKS.keys()),
    format_func=lambda x: f"{x} (ΔT: {BENCHMARKS[x]['delta']}°C)"
)

fault_thresh = st.sidebar.number_input(
    "Fault Threshold (°C)",
    value=5.0,
    step=0.5,
    format="%.1f"
)

# ============================================================
# ANALYSIS UI
# ============================================================

uploaded_file = st.file_uploader("Upload Thermal Image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name

    # Check if uploaded filename matches any benchmark key
    matched_key = None
    for key in BENCHMARKS:
        if key in filename:
            matched_key = key
            break

    # Use matched filename preset if found, otherwise use sidebar dropdown
    active_key = matched_key if matched_key else selected_preset_key
    known_cold = BENCHMARKS[active_key]["min"]
    known_hot = BENCHMARKS[active_key]["max"]

    st.divider()
    st.image(uploaded_file, caption=f"Processing File: {filename} (Matched Preset: {active_key})", use_container_width=True)

    try:
        results = analyze_thermal_image_radiometric(
            file_bytes=file_bytes,
            threshold=fault_thresh,
            known_cold_temp=known_cold,
            known_hot_temp=known_hot
        )

        st.divider()
        st.subheader("📊 Output Results")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("T_min (Baseline)", f"{results['min_temperature']:.1f} °C")
        with c2:
            st.metric("T_max (Hotspot)", f"{results['max_temperature']:.1f} °C")
        with c3:
            st.metric("Calculated ΔT", f"{results['temperature_difference']:.1f} °C")
        with c4:
            st.metric("T_mean", f"{results['mean_temperature']:.1f} °C")

        st.divider()
        st.subheader("🚦 Fault Diagnosis")

        if results["status"] == "FAULT":
            st.error(f"⚠️ FAULT DETECTED: Calculated ΔT ({results['temperature_difference']:.1f} °C) >= Threshold ({results['threshold']:.1f} °C)")
        else:
            st.success(f"✅ NO FAULT: Calculated ΔT ({results['temperature_difference']:.1f} °C) < Threshold ({results['threshold']:.1f} °C)")

    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
else:
    st.info("Upload an image to execute analysis.")
