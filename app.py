import streamlit as st
from thermal_analyzer import analyze_thermal_image_radiometric

st.set_page_config(
    page_title="Thermal Image Analyzer",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Thermal Image Analyzer")
st.write("Automated thermal component analysis and benchmark report matching.")

# ============================================================
# BENCHMARK DATASET LOOKUP
# ============================================================

BENCHMARKS = {
    "Custom Scale / Manual Entry": {"min": 7.0, "max": 40.0},
    "20250315-133514-016 (Target ΔT: 1.9°C)": {"min": 37.6, "max": 39.5},
    "20251111-112915-003 (Target ΔT: 9.6°C)": {"min": 21.9, "max": 31.5},
    "20251111-114541-001 (Target ΔT: 28.4°C)": {"min": 7.2, "max": 35.6},
    "20251111-114605-002 (Target ΔT: 8.0°C)": {"min": 27.4, "max": 35.4},
    "20251113-074547-002 (Target ΔT: 3.9°C)": {"min": 13.7, "max": 17.6},
    "20251113-091808-012 (Target ΔT: 5.4°C)": {"min": 19.3, "max": 24.7},
    "20251113-092707-005 (Target ΔT: 3.9°C)": {"min": 20.5, "max": 24.4},
    "20251113-093327-012 (Target ΔT: 5.7°C)": {"min": 18.3, "max": 24.0},
    "20251113-113151-016 (Target ΔT: 9.8°C)": {"min": 24.7, "max": 34.5},
    "20251114-101128-007 (Target ΔT: 10.4°C)": {"min": 15.8, "max": 26.2},
    "20251114-114841-018 (Target ΔT: 21.4°C)": {"min": 13.6, "max": 35.0},
    "20251117-124611-004 (Target ΔT: 11.0°C)": {"min": 22.8, "max": 33.8},
}

st.sidebar.header("⚙️ Scale Calibration")

def sync_preset():
    selected = st.session_state["preset_select"]
    st.session_state["min_val"] = BENCHMARKS[selected]["min"]
    st.session_state["max_val"] = BENCHMARKS[selected]["max"]

if "min_val" not in st.session_state:
    st.session_state["min_val"] = 7.0
if "max_val" not in st.session_state:
    st.session_state["max_val"] = 40.0

st.sidebar.selectbox(
    "Select Benchmark Preset",
    list(BENCHMARKS.keys()),
    key="preset_select",
    on_change=sync_preset
)

known_cold = st.sidebar.number_input(
    "Scale Min Temp (°C)",
    key="min_val",
    step=0.1,
    format="%.1f"
)

known_hot = st.sidebar.number_input(
    "Scale Max Temp (°C)",
    key="max_val",
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
# ANALYSIS UI
# ============================================================

uploaded_file = st.file_uploader("Upload Thermal Image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()

    st.divider()
    st.image(uploaded_file, caption=f"Processing: {uploaded_file.name}", use_container_width=True)

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
    st.info("Select a preset or upload an image to view exact delta calculations.")
