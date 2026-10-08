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
st.write("Upload a thermal image to analyze the track temperature.")

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("Settings")

scale_max = st.sidebar.number_input(
    "Upper temperature (°C)",
    value=35.0,
    step=1.0
)

scale_min = st.sidebar.number_input(
    "Lower temperature (°C)",
    value=-26.0,
    step=1.0
)

fault_limit = st.sidebar.number_input(
    "Fault threshold (°C)",
    value=5.0,
    step=0.5
)

# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------

def get_colorbar(image):
    h, w = image.shape[:2]

    x1 = int(w * 0.91)
    x2 = int(w * 0.985)

    y1 = int(h * 0.15)
    y2 = int(h * 0.85)

    return image[y1:y2, x1:x2]


def create_temperature_lookup(image, max_temp, min_temp):
    colorbar = get_colorbar(image)

    if colorbar.size == 0:
        return None

    h, w = colorbar.shape[:2]

    # Use several columns instead of only one
    x1 = max(0, int(w * 0.25))
    x2 = min(w, int(w * 0.75))

    colors = []
    temperatures = []

    for y in range(h):

        row = colorbar[y, x1:x2]

        if len(row) == 0:
            continue

        color = np.mean(row, axis=0)

        colors.append(color)

        temperature = max_temp - (
            y / max(1, h - 1)
        ) * (max_temp - min_temp)

        temperatures.append(temperature)

    if len(colors) == 0:
        return None

    return (
        np.array(colors, dtype=np.float32),
        np.array(temperatures, dtype=np.float32)
    )


def detect_rail(image):
    h, w = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    saturation = hsv[:, :, 1]
    brightness = hsv[:, :, 2]

    # Bright thermal regions
    mask = (
        (brightness > 130) &
        (saturation > 45)
    ).astype(np.uint8) * 255

    # Remove temperature scale
    mask[:, int(w * 0.88):] = 0

    # Remove text/header
    mask[:int(h * 0.12), :] = 0

    # Remove timestamp/footer
    mask[int(h * 0.90):, :] = 0

    # Morphological filtering
    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((7, 7), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel1
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel2
    )

    # Keep only reasonably large connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    clean = np.zeros_like(mask)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area > 100:
            clean[labels == i] = 255

    # Slight dilation
    clean = cv2.dilate(
        clean,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    return clean


def pixel_temperature(pixel, lookup):
    colors, temperatures = lookup

    pixel = pixel.astype(np.float32)

    distances = np.linalg.norm(
        colors - pixel,
        axis=1
    )

    index = np.argmin(distances)

    return float(temperatures[index])


def calculate_temperature(image, mask, lookup):

    pixels = image[mask > 0]

    if len(pixels) == 0:
        return None

    # Sample maximum 8000 pixels
    if len(pixels) > 8000:
        indexes = np.linspace(
            0,
            len(pixels) - 1,
            8000
        ).astype(int)

        pixels = pixels[indexes]

    temperatures = []

    for pixel in pixels:

        temp = pixel_temperature(
            pixel,
            lookup
        )

        temperatures.append(temp)

    temperatures = np.array(
        temperatures,
        dtype=np.float32
    )

    if len(temperatures) < 10:
        return None

    # Remove extreme noise
    p2 = np.percentile(
        temperatures,
        2
    )

    p98 = np.percentile(
        temperatures,
        98
    )

    temperatures = temperatures[
        (temperatures >= p2) &
        (temperatures <= p98)
    ]

    if len(temperatures) == 0:
        return None

    minimum = float(np.min(temperatures))
    maximum = float(np.max(temperatures))
    mean = float(np.mean(temperatures))

    difference = maximum - minimum

    return minimum, maximum, mean, difference


def show_detection(image, mask):

    result = image.copy()

    overlay = image.copy()

    overlay[mask > 0] = [255, 255, 255]

    result = cv2.addWeighted(
        result,
        0.65,
        overlay,
        0.35,
        0
    )

    return result


# ---------------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload thermal image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image = np.array(image)

    st.subheader("Thermal Image")

    st.image(
        image,
        use_container_width=True
    )

    # -----------------------------------------------------
    # CREATE TEMPERATURE SCALE
    # -----------------------------------------------------

    lookup = create_temperature_lookup(
        image,
        scale_max,
        scale_min
    )

    if lookup is None:

        st.error(
            "Could not read the thermal temperature scale."
        )

        st.stop()

    # -----------------------------------------------------
    # DETECT RAIL
    # -----------------------------------------------------

    with st.spinner("Analyzing thermal track..."):

        rail_mask = detect_rail(image)

    detected = show_detection(
        image,
        rail_mask
    )

    st.subheader("Detected Thermal Track")

    st.image(
        detected,
        use_container_width=True
    )

    # -----------------------------------------------------
    # CALCULATE TEMPERATURE
    # -----------------------------------------------------

    result = calculate_temperature(
        image,
        rail_mask,
        lookup
    )

    if result is None:

        st.error(
            "No suitable thermal track region was detected."
        )

        st.stop()

    minimum, maximum, mean, difference = result

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    st.subheader("Temperature Results")

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
    # FAULT DECISION
    # -----------------------------------------------------

    st.subheader("Track Condition")

    if difference >= fault_limit:

        st.error(
            "⚠️ FAULT DETECTED"
        )

        st.write(
            f"Temperature difference: "
            f"{difference:.1f} °C"
        )

        st.warning(
            "Attention required within 2 days."
        )

    else:

        st.success(
            "✅ NO FAULT"
        )

        st.write(
            f"Temperature difference: "
            f"{difference:.1f} °C"
        )

        st.info(
            "Temperature difference is below the fault threshold."
        )

else:

    st.info(
        "Upload a thermal image to start the analysis."
    )
