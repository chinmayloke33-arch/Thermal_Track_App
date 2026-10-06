import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import (
    streamlit_image_coordinates
)

from thermal_analyzer import (
    get_temperature_from_colorbar,
    estimate_temperature_map,
    analyze_track,
    create_result_image
)


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Thermal Track Inspection",
    page_icon="🌡️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌡️ Thermal Track Inspection System")

st.write(
    "Upload a thermal image of the railway track "
    "and analyze the temperature variation."
)

st.info(
    "The system uses the temperature scale visible "
    "on the thermal image to estimate temperatures."
)


# ============================================================
# SESSION STATE
# ============================================================

if "stage" not in st.session_state:
    st.session_state.stage = "upload"

if "colorbar_points" not in st.session_state:
    st.session_state.colorbar_points = []

if "track_points" not in st.session_state:
    st.session_state.track_points = []

if "last_click" not in st.session_state:
    st.session_state.last_click = None

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "result_image" not in st.session_state:
    st.session_state.result_image = None


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


if uploaded_file is None:

    st.warning(
        "Please upload a thermal image to begin."
    )

    st.stop()


# ============================================================
# READ IMAGE
# ============================================================

pil_image = Image.open(
    uploaded_file
).convert("RGB")

rgb_image = np.array(
    pil_image
)

# OpenCV uses BGR
image = cv2.cvtColor(
    rgb_image,
    cv2.COLOR_RGB2BGR
)


# ============================================================
# RESIZE IMAGE FOR DISPLAY
# ============================================================

original_height, original_width = (
    image.shape[:2]
)

max_display_width = 1000

if original_width > max_display_width:

    display_width = max_display_width

    scale = (
        display_width
        / original_width
    )

    display_height = int(
        original_height * scale
    )

    display_image = cv2.resize(
        image,
        (
            display_width,
            display_height
        ),
        interpolation=cv2.INTER_AREA
    )

else:

    display_image = image.copy()

    display_width = original_width
    display_height = original_height

    scale = 1.0


display_rgb = cv2.cvtColor(
    display_image,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# RESET BUTTON
# ============================================================

if st.sidebar.button(
    "🔄 Start Again"
):

    st.session_state.stage = "upload"

    st.session_state.colorbar_points = []

    st.session_state.track_points = []

    st.session_state.last_click = None

    st.session_state.analysis = None

    st.session_state.result_image = None

    st.rerun()


# ============================================================
# STEP 1
# ============================================================

st.header("Step 1 — Select Temperature Scale")

st.write(
    "Click the TOP of the vertical temperature "
    "color bar, then click the BOTTOM."
)

st.caption(
    "Usually the temperature scale is located "
    "on the right side of the thermal image."
)


if st.session_state.stage == "upload":

    st.session_state.stage = "colorbar"


# Display image
click = streamlit_image_coordinates(
    display_rgb,
    key="temperature_scale_image"
)


# Process color-bar clicks
if (
    click is not None
    and st.session_state.stage == "colorbar"
):

    current_click = (
        click["x"],
        click["y"]
    )

    if (
        current_click
        != st.session_state.last_click
    ):

        st.session_state.last_click = (
            current_click
        )

        original_x = int(
            click["x"] / scale
        )

        original_y = int(
            click["y"] / scale
        )

        st.session_state.colorbar_points.append(
            (
                original_x,
                original_y
            )
        )

        st.rerun()


# Show selected points
if len(
    st.session_state.colorbar_points
) >= 1:

    st.success(
        "TOP of color bar selected."
    )


if len(
    st.session_state.colorbar_points
) >= 2:

    st.success(
        "BOTTOM of color bar selected."
    )


# ============================================================
# TEMPERATURE VALUES
# ============================================================

if len(
    st.session_state.colorbar_points
) >= 2:

    st.header(
        "Temperature Scale Values"
    )

    col1, col2 = st.columns(2)

    with col1:

        top_temperature = st.number_input(
            "Temperature at TOP (°C)",
            value=33.0,
            step=0.1
        )

    with col2:

        bottom_temperature = st.number_input(
            "Temperature at BOTTOM (°C)",
            value=-19.0,
            step=0.1
        )

    if st.button(
        "Continue to Track Selection",
        type="primary"
    ):

        st.session_state.stage = "track"

        st.session_state.track_points = []

        st.session_state.last_click = None

        st.rerun()


# ============================================================
# STEP 2 — TRACK REGION
# ============================================================

if st.session_state.stage == "track":

    st.header(
        "Step 2 — Select Track / Rail Region"
    )

    st.write(
        "Click points around the track or rail "
        "to create a polygon."
    )

    st.info(
        "Select ONLY the region of the track/rail "
        "that you want to analyze."
    )

    # Show image again
    click = streamlit_image_coordinates(
        display_rgb,
        key="track_selection_image"
    )

    if click is not None:

        current_click = (
            click["x"],
            click["y"]
        )

        if (
            current_click
            != st.session_state.last_click
        ):

            st.session_state.last_click = (
                current_click
            )

            original_x = int(
                click["x"] / scale
            )

            original_y = int(
                click["y"] / scale
            )

            st.session_state.track_points.append(
                (
                    original_x,
                    original_y
                )
            )

            st.rerun()


    # Number of selected points
    st.write(
        f"Selected points: "
        f"**{len(st.session_state.track_points)}**"
    )


    # Display selected coordinates
    if len(
        st.session_state.track_points
    ) > 0:

        st.write(
            st.session_state.track_points
        )


    col1, col2 = st.columns(2)


    with col1:

        if st.button(
            "↩️ Undo Last Point"
        ):

            if len(
                st.session_state.track_points
            ) > 0:

                st.session_state.track_points.pop()

                st.session_state.last_click = None

                st.rerun()


    with col2:

        if st.button(
            "✅ Finish Track Selection",
            type="primary"
        ):

            if len(
                st.session_state.track_points
            ) < 3:

                st.error(
                    "Please select at least "
                    "3 points."
                )

            else:

                st.session_state.stage = "ready"

                st.rerun()


# ============================================================
# STEP 3 — ANALYSIS
# ============================================================

if st.session_state.stage == "ready":

    st.header(
        "Step 3 — Analyze Image"
    )

    st.success(
        f"Track region selected using "
        f"{len(st.session_state.track_points)} points."
    )

    st.write(
        "The system will now convert the thermal "
        "colors into estimated temperatures."
    )


    if st.button(
        "🔍 Analyze Thermal Image",
        type="primary"
    ):

        try:

            # ------------------------------------------------
            # Create color-temperature calibration
            # ------------------------------------------------

            top_point = (
                st.session_state.colorbar_points[0]
            )

            bottom_point = (
                st.session_state.colorbar_points[1]
            )


            palette, temperatures = (
                get_temperature_from_colorbar(
                    image,
                    top_point,
                    bottom_point,
                    top_temperature,
                    bottom_temperature
                )
            )


            # ------------------------------------------------
            # Convert image to temperature map
            # ------------------------------------------------

            with st.spinner(
                "Converting image colors to temperature..."
            ):

                temperature_map = (
                    estimate_temperature_map(
                        image,
                        palette,
                        temperatures
                    )
                )


            # ------------------------------------------------
            # Analyze selected track
            # ------------------------------------------------

            analysis = analyze_track(
                image,
                st.session_state.track_points,
                temperature_map
            )


            # ------------------------------------------------
            # Create annotated image
            # ------------------------------------------------

            result = create_result_image(
                image,
                analysis
            )


            st.session_state.analysis = analysis

            st.session_state.result_image = result

            st.session_state.stage = "result"

            st.rerun()


        except Exception as error:

            st.error(
                f"Analysis failed: {error}"
            )


# ============================================================
# RESULTS
# ============================================================

if (
    st.session_state.stage == "result"
    and st.session_state.analysis is not None
):

    analysis = st.session_state.analysis

    st.header(
        "📊 Thermal Analysis Result"
    )


    # --------------------------------------------------------
    # Main measurements
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Minimum Temperature",
            f"{analysis['min_temp']:.2f} °C"
        )

    with col2:

        st.metric(
            "Maximum Temperature",
            f"{analysis['max_temp']:.2f} °C"
        )

    with col3:

        st.metric(
            "Temperature Difference",
            f"{analysis['difference']:.2f} °C"
        )


    st.divider()


    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    difference = analysis["difference"]


    if difference < 5.0:

        st.success(
            "✅ NO FAULT DETECTED"
        )

        st.write(
            f"The temperature difference is "
            f"**{difference:.2f} °C**, which is "
            f"less than 5 °C."
        )

        st.write(
            "**No immediate attention is required.**"
        )

    else:

        st.error(
            "⚠️ FAULT DETECTED"
        )

        st.write(
            f"The temperature difference is "
            f"**{difference:.2f} °C**, which is "
            f"5 °C or greater."
        )

        st.warning(
            "⚠️ Attention is required within 2 days."
        )


    # --------------------------------------------------------
    # Detailed values
    # --------------------------------------------------------

    st.subheader(
        "Additional Temperature Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.write(
            f"**Average:** "
            f"{analysis['mean_temp']:.2f} °C"
        )

    with col2:

        st.write(
            f"**Median:** "
            f"{analysis['median_temp']:.2f} °C"
        )

    with col3:

        st.write(
            f"**5th–95th percentile:** "
            f"{analysis['p5']:.2f}–"
            f"{analysis['p95']:.2f} °C"
        )


    # --------------------------------------------------------
    # Annotated image
    # --------------------------------------------------------

    st.subheader(
        "Analyzed Thermal Image"
    )

    result_rgb = cv2.cvtColor(
        st.session_state.result_image,
        cv2.COLOR_BGR2RGB
    )

    st.image(
        result_rgb,
        caption="Thermal Analysis Result",
        width="stretch"
    )


    # --------------------------------------------------------
    # Download result
    # --------------------------------------------------------

    success, encoded_image = cv2.imencode(
        ".jpg",
        st.session_state.result_image
    )

    if success:

        st.download_button(
            "⬇️ Download Analysis Result",
            data=encoded_image.tobytes(),
            file_name="thermal_analysis_result.jpg",
            mime="image/jpeg"
        )