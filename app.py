
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

st.title("🌡️ Thermal Image Analyzer")

st.write(
    "Thermal image analysis for railway track inspection. "
    "The system extracts the thermal region and calculates "
    "minimum temperature, maximum temperature and temperature difference."
)

# ============================================================
# SETTINGS
# ============================================================

FAULT_THRESHOLD = 5.0

# Default scale values.
# These are ONLY defaults and are NOT used to create
# an artificial 7–40°C temperature range.

DEFAULT_SCALE_MIN = 20.0
DEFAULT_SCALE_MAX = 40.0

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Analysis Settings")

st.sidebar.write(
    "Enter the temperature values shown at the top "
    "and bottom of the thermal colour scale."
)

scale_max = st.sidebar.number_input(
    "Thermal scale MAX (°C)",
    min_value=-50.0,
    max_value=150.0,
    value=DEFAULT_SCALE_MAX,
    step=0.1
)

scale_min = st.sidebar.number_input(
    "Thermal scale MIN (°C)",
    min_value=-50.0,
    max_value=150.0,
    value=DEFAULT_SCALE_MIN,
    step=0.1
)

if scale_max <= scale_min:
    st.sidebar.error(
        "Scale MAX must be greater than Scale MIN."
    )
    st.stop()

st.sidebar.divider()

# Percentage of hottest pixels used for maximum
hot_percentile = st.sidebar.slider(
    "Hot pixel percentile",
    min_value=90,
    max_value=100,
    value=98,
    step=1
)

# Percentage of coldest pixels used for minimum
cold_percentile = st.sidebar.slider(
    "Cold pixel percentile",
    min_value=0,
    max_value=10,
    value=2,
    step=1
)

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
# FUNCTION:
# CONVERT THERMAL IMAGE TO INTENSITY
# ============================================================

def calculate_thermal_intensity(image_rgb):

    image = image_rgb.astype(np.float32)

    # Convert RGB to HSV
    hsv = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2HSV
    )

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # --------------------------------------------------------
    # Thermal images contain very bright yellow/white
    # regions for hot surfaces.
    #
    # We combine brightness and saturation rather than using
    # grayscale.
    # --------------------------------------------------------

    brightness = v / 255.0
    saturation = s / 255.0

    # Red/yellow/white thermal colours generally have
    # strong thermal intensity.
    thermal_intensity = (
        0.65 * brightness +
        0.35 * saturation
    )

    return thermal_intensity


# ============================================================
# FUNCTION:
# REMOVE UI / TEXT / TEMPERATURE SCALE
# ============================================================

def create_analysis_mask(image_rgb):

    height, width, _ = image_rgb.shape

    hsv = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2HSV
    )

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # --------------------------------------------------------
    # Basic thermal brightness mask
    # --------------------------------------------------------

    mask = (
        (v > 90) &
        (s > 70)
    )

    # --------------------------------------------------------
    # Ignore the right-side temperature colour bar.
    # The colour scale occupies approximately the last 8–10%
    # of the image.
    # --------------------------------------------------------

    right_limit = int(width * 0.90)

    mask[:, right_limit:] = False

    # --------------------------------------------------------
    # Ignore the upper information area.
    # --------------------------------------------------------

    top_limit = int(height * 0.08)

    mask[:top_limit, :] = False

    # --------------------------------------------------------
    # Ignore bottom timestamp area.
    # --------------------------------------------------------

    bottom_limit = int(height * 0.92)

    mask[bottom_limit:, :] = False

    # --------------------------------------------------------
    # Remove small isolated objects.
    # --------------------------------------------------------

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask.astype(bool)


# ============================================================
# FUNCTION:
# CONVERT THERMAL INTENSITY TO TEMPERATURE
# ============================================================

def intensity_to_temperature(
    intensity,
    scale_min,
    scale_max
):

    # Normalize image intensity.
    #
    # IMPORTANT:
    # This is calibrated using the thermal scale supplied
    # by the user, not an arbitrary 7–40°C range.

    intensity = np.clip(
        intensity,
        0.0,
        1.0
    )

    temperature = (
        scale_min
        +
        intensity *
        (scale_max - scale_min)
    )

    return temperature


# ============================================================
# MAIN PROGRAM
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")

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
    # CREATE THERMAL MASK
    # ========================================================

    mask = create_analysis_mask(img)

    number_of_pixels = int(
        np.sum(mask)
    )

    if number_of_pixels < 50:

        st.error(
            "Not enough thermal pixels were detected. "
            "Try adjusting the image or analysis settings."
        )

        st.stop()

    # ========================================================
    # CALCULATE THERMAL INTENSITY
    # ========================================================

    intensity = calculate_thermal_intensity(
        img
    )

    selected_intensity = intensity[
        mask
    ]

    # ========================================================
    # REMOVE EXTREME NOISE
    # ========================================================

    low_intensity = np.percentile(
        selected_intensity,
        cold_percentile
    )

    high_intensity = np.percentile(
        selected_intensity,
        hot_percentile
    )

    selected_intensity = selected_intensity[
        (selected_intensity >= low_intensity) &
        (selected_intensity <= high_intensity)
    ]

    if len(selected_intensity) < 20:

        st.error(
            "Not enough valid thermal pixels remain "
            "after filtering."
        )

        st.stop()

    # ========================================================
    # CONVERT TO TEMPERATURE
    # ========================================================

    temperatures = intensity_to_temperature(
        selected_intensity,
        scale_min,
        scale_max
    )

    # ========================================================
    # CALCULATE RESULTS
    # ========================================================

    min_temp = float(
        np.min(temperatures)
    )

    max_temp = float(
        np.max(temperatures)
    )

    mean_temp = float(
        np.mean(temperatures)
    )

    difference = abs(
        max_temp - min_temp
    )

    # ========================================================
    # RESULTS
    # ========================================================

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
            "**Status:** NO FAULT"
        )

        st.info(
            "Temperature difference is below "
            "the 5°C threshold."
        )

        status = "NO FAULT"

        required_action = (
            "No action required"
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
            "**Status:** FAULT"
        )

        status = "FAULT"

        required_action = (
            "Attention required within 2 days"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    st.divider()

    st.subheader("📊 Analysis Summary")

    results = {

        "Minimum Temperature":
            f"{min_temp:.2f} °C",

        "Maximum Temperature":
            f"{max_temp:.2f} °C",

        "Temperature Difference":
            f"{difference:.2f} °C",

        "Mean Temperature":
            f"{mean_temp:.2f} °C",

        "Fault Threshold":
            f"{FAULT_THRESHOLD:.2f} °C",

        "Status":
            status,

        "Required Action":
            required_action,

        "Thermal Pixels Analysed":
            f"{number_of_pixels:,}"
    }

    for key, value in results.items():

        st.write(
            f"**{key}:** {value}"
        )

    # ========================================================
    # SHOW ANALYSIS MASK
    # ========================================================

    st.divider()

    st.subheader("🔍 Thermal Region Detected")

    display_mask = (
        mask.astype(np.uint8) * 255
    )

    st.image(
        display_mask,
        caption="Pixels used for thermal analysis",
        use_container_width=True
    )

    # ========================================================
    # INFORMATION
    # ========================================================

    st.divider()

    st.subheader("ℹ️ Analysis Information")

    st.write(
        f"**Thermal scale:** "
        f"{scale_min:.1f} – {scale_max:.1f} °C"
    )

    st.write(
        f"**Fault threshold:** "
        f"{FAULT_THRESHOLD:.1f} °C"
    )

    st.write(
        f"**Hot percentile:** "
        f"{hot_percentile}th"
    )

    st.write(
        f"**Cold percentile:** "
        f"{cold_percentile}th"
    )

    st.warning(
        "This version is a thermal-colour prototype. "
        "It does not read the camera's hidden radiometric "
        "temperature data. The thermal scale values shown "
        "on the image should be entered in the sidebar."
    )

else:

    st.info(
        "Please upload a thermal image to begin."
    )

    st.write(
        "**Fault threshold:** 5 °C"
    )

    st.write(
        "**Fault action:** Attention required within 2 days."
    )
