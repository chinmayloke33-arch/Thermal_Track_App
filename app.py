import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.set_page_config(
    page_title="Thermal Image Analyzer",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Thermal Image Analyzer")
st.write("Upload a thermal image to calculate the minimum and maximum temperature.")

# ---------------------------------------------------
# Upload image
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload thermal image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    st.subheader("Uploaded Thermal Image")
    st.image(image, use_container_width=True)

    st.divider()

    # ---------------------------------------------------
    # Temperature calibration
    # ---------------------------------------------------

    st.subheader("Temperature Scale")

    st.info(
        "Enter the temperature corresponding to the top and bottom "
        "of the thermal image's color scale."
    )

    col1, col2 = st.columns(2)

    with col1:
        max_scale_temp = st.number_input(
            "Top / maximum scale temperature (°C)",
            value=25.0,
            step=0.1
        )

    with col2:
        min_scale_temp = st.number_input(
            "Bottom / minimum scale temperature (°C)",
            value=21.0,
            step=0.1
        )

    if max_scale_temp <= min_scale_temp:
        st.error("Maximum scale temperature must be greater than minimum scale temperature.")
        st.stop()

    # ---------------------------------------------------
    # Convert image to temperature
    # ---------------------------------------------------

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Normalize grayscale intensity
    gray_normalized = gray.astype(np.float32) / 255.0

    # Convert intensity to temperature
    temperature = (
        min_scale_temp
        + gray_normalized *
        (max_scale_temp - min_scale_temp)
    )

    # ---------------------------------------------------
    # Whole image analysis
    # ---------------------------------------------------

    min_temp = float(np.min(temperature))
    max_temp = float(np.max(temperature))
    difference = max_temp - min_temp
    mean_temp = float(np.mean(temperature))

    st.divider()

    st.subheader("🌡️ Temperature Results")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Minimum Temperature",
            f"{min_temp:.2f} °C"
        )

    with c2:
        st.metric(
            "Maximum Temperature",
            f"{max_temp:.2f} °C"
        )

    with c3:
        st.metric(
            "Temperature Difference",
            f"{difference:.2f} °C"
        )

    with c4:
        st.metric(
            "Mean Temperature",
            f"{mean_temp:.2f} °C"
        )

    # ---------------------------------------------------
    # Fault decision
    # ---------------------------------------------------

    st.divider()

    if difference < 5:
        st.success(
            f"✅ NO FAULT\n\n"
            f"Temperature difference is {difference:.2f} °C, "
            f"which is less than 5 °C."
        )

        st.write("**Status:** No fault detected.")

    else:
        st.error(
            f"⚠️ FAULT DETECTED\n\n"
            f"Temperature difference is {difference:.2f} °C, "
            f"which is equal to or greater than 5 °C."
        )

        st.warning(
            "Attention required within 2 days."
        )

    # ---------------------------------------------------
    # Temperature information
    # ---------------------------------------------------

    st.divider()

    st.subheader("📊 Analysis Summary")

    results = {
        "Minimum Temperature": f"{min_temp:.2f} °C",
        "Maximum Temperature": f"{max_temp:.2f} °C",
        "Temperature Difference": f"{difference:.2f} °C",
        "Mean Temperature": f"{mean_temp:.2f} °C",
        "Threshold": "5.00 °C",
        "Status": "NO FAULT" if difference < 5 else "FAULT",
        "Required Action": (
            "None"
            if difference < 5
            else "Attention required within 2 days"
        )
    }

    for key, value in results.items():
        st.write(f"**{key}:** {value}")
