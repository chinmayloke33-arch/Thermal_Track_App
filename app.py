import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.set_page_config(
    page_title="Thermal Track Analyzer",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Thermal Track Analyzer")

uploaded_file = st.file_uploader(
    "Upload thermal image",
    type=["jpg", "jpeg", "png"]
)

st.sidebar.header("Temperature Settings")

# Use the temperature range actually visible on the thermal image.
# These can be changed for different cameras.
UPPER_TEMP = st.sidebar.number_input(
    "Upper scale temperature (°C)",
    value=40.0,
    step=1.0
)

LOWER_TEMP = st.sidebar.number_input(
    "Lower scale temperature (°C)",
    value=0.0,
    step=1.0
)

FAULT_LIMIT = 5.0


# =========================================================
# THERMAL COLOR INTENSITY
# =========================================================

def thermal_intensity(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    # Thermal images:
    # dark purple = colder
    # red/orange = warmer
    # yellow/white = hottest

    intensity = (
        0.55 * value +
        0.45 * saturation
    )

    return intensity


# =========================================================
# REMOVE UNWANTED AREAS
# =========================================================

def create_valid_mask(image):

    h, w = image.shape[:2]

    mask = np.ones((h, w), dtype=np.uint8)

    # Remove right-side temperature scale
    mask[:, int(w * 0.89):] = 0

    # Remove top information
    mask[:int(h * 0.13), :] = 0

    # Remove bottom timestamp
    mask[int(h * 0.90):, :] = 0

    return mask


# =========================================================
# DETECT RAIL
# =========================================================

def detect_rail(image):

    intensity = thermal_intensity(image)

    valid = create_valid_mask(image)

    values = intensity[valid > 0]

    if len(values) == 0:
        return None

    # Only keep the hottest part of the thermal image.
    # This prevents the purple background from becoming
    # the minimum temperature.
    threshold = np.percentile(values, 88)

    mask = (
        (intensity >= threshold) &
        (valid > 0)
    ).astype(np.uint8) * 255

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Connect broken rail lines
    kernel2 = np.ones((7, 7), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel2
    )

    # Keep only reasonably large regions
    number, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    final_mask = np.zeros_like(mask)

    for i in range(1, number):

        area = stats[i, cv2.CC_STAT_AREA]

        if area > 50:
            final_mask[labels == i] = 255

    return final_mask


# =========================================================
# CONVERT THERMAL INTENSITY TO TEMPERATURE
# =========================================================

def calculate_temperature(image, mask):

    intensity = thermal_intensity(image)

    pixels = intensity[mask > 0]

    if len(pixels) < 20:
        return None

    # Remove extreme noise
    low = np.percentile(pixels, 5)
    high = np.percentile(pixels, 95)

    pixels = pixels[
        (pixels >= low) &
        (pixels <= high)
    ]

    if len(pixels) == 0:
        return None

    # Normalize ONLY inside detected thermal rail region
    normalized = (
        pixels - low
    ) / max(
        high - low,
        1
    )

    temperatures = (
        LOWER_TEMP +
        normalized *
        (UPPER_TEMP - LOWER_TEMP)
    )

    minimum = float(np.percentile(temperatures, 5))
    maximum = float(np.percentile(temperatures, 95))
    mean = float(np.mean(temperatures))

    difference = maximum - minimum

    return minimum, maximum, mean, difference


# =========================================================
# SHOW DETECTED AREA
# =========================================================

def show_detection(image, mask):

    output = image.copy()

    overlay = output.copy()

    overlay[mask > 0] = [255, 255, 255]

    output = cv2.addWeighted(
        output,
        0.70,
        overlay,
        0.30,
        0
    )

    return output


# =========================================================
# MAIN
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image = np.array(image)

    st.subheader("Uploaded Thermal Image")

    st.image(
        image,
        use_container_width=True
    )

    # -----------------------------------------------------
    # Detect rail
    # -----------------------------------------------------

    with st.spinner("Analyzing thermal rail..."):

        rail_mask = detect_rail(image)

    if rail_mask is None:

        st.error(
            "Unable to detect the thermal rail."
        )

        st.stop()

    # -----------------------------------------------------
    # Display detection
    # -----------------------------------------------------

    st.subheader("Detected Rail Region")

    detected = show_detection(
        image,
        rail_mask
    )

    st.image(
        detected,
        use_container_width=True
    )

    # -----------------------------------------------------
    # Calculate temperatures
    # -----------------------------------------------------

    result = calculate_temperature(
        image,
        rail_mask
    )

    if result is None:

        st.error(
            "Not enough thermal information was detected."
        )

        st.stop()

    minimum, maximum, mean, difference = result

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    st.subheader("Temperature Analysis")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Minimum",
            f"{minimum:.1f} °C"
        )

    with c2:
        st.metric(
            "Maximum",
            f"{maximum:.1f} °C"
        )

    with c3:
        st.metric(
            "Mean",
            f"{mean:.1f} °C"
        )

    with c4:
        st.metric(
            "Difference",
            f"{difference:.1f} °C"
        )

    # -----------------------------------------------------
    # Fault decision
    # -----------------------------------------------------

    st.subheader("Track Condition")

    if difference >= FAULT_LIMIT:

        st.error("⚠️ FAULT DETECTED")

        st.write(
            f"Temperature difference = "
            f"{difference:.1f} °C"
        )

        st.warning(
            "Attention required within 2 days."
        )

    else:

        st.success("✅ NO FAULT")

        st.write(
            f"Temperature difference = "
            f"{difference:.1f} °C"
        )

        st.info(
            "Temperature difference is below 5 °C."
        )
