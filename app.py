import streamlit as st
from thermal_analyzer import analyze_thermal_image_radiometric

st.set_page_config(page_title="Radiometric Thermal Analyzer", layout="wide")
st.title("🌡️ Radiometric Thermal Image Analyzer")

st.write(
    "Upload raw **Radiometric JPGs (.jpg / .rjpg)** exported directly from the camera. "
    "Calculations use embedded sensor metadata for exact OEM temperature values."
)

st.sidebar.header("⚙️ Configuration")
fault_thresh = st.sidebar.number_input("Fault Threshold (°C)", value=5.0, step=0.5)

uploaded_file = st.file_uploader("Upload Radiometric Image", type=["jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    
    st.image(uploaded_file, caption="Uploaded Thermal Target", use_container_width=True)

    try:
        results = analyze_thermal_image_radiometric(
            file_bytes=file_bytes,
            threshold=fault_thresh
        )

        st.divider()
        st.subheader("📊 True Sensor Temperature Results")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Frame Min Temp", f"{results['min_temperature']} °C")
        with c2:
            st.metric("Peak Hotspot Temp", f"{results['max_temperature']} °C")
        with c3:
            st.metric("Sensor Delta (ΔT)", f"{results['temperature_difference']} °C")
        with c4:
            st.metric("Average Temp", f"{results['mean_temperature']} °C")

        st.divider()
        if results["status"] == "FAULT":
            st.error(f"⚠️ FAULT DETECTED: ΔT ({results['temperature_difference']} °C) >= {fault_thresh} °C")
        else:
            st.success(f"✅ NO FAULT: ΔT ({results['temperature_difference']} °C) < {fault_thresh} °C")

    except Exception as e:
        st.error(
            f"Unable to extract radiometric data: {str(e)}. "
            "Ensure the uploaded file is a raw camera file containing embedded thermal metadata."
        )
