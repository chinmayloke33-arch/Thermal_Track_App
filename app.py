```python
import streamlit as st
from PIL import Image
import numpy as np

from thermal_analyzer import analyze_thermal_image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Thermal Track Inspection",
    page_icon="🌡️",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌡️ Thermal Track Inspection")

st.write(
    "Upload a thermal image of the railway track. "
    "The system automatically reads the P1 and P2 "
    "temperature readings and checks for a fault."
)

st.divider()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Thermal Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Uploaded Thermal Image")

    st.image(
        image,
        caption=uploaded_file.name,
        use_container_width=True
    )

    st.divider()


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    if st.button(
        "🔍 Analyze Image",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Automatically reading P1 and P2 temperatures..."
        ):

            result = analyze_thermal_image(
                np.array(image)
            )


        # ====================================================
        # ANALYSIS FAILED
        # ====================================================

        if not result["success"]:

            st.error(
                "❌ Analysis could not be completed."
            )

            st.warning(
                result["message"]
            )

            st.info(
                "Make sure the P1 and P2 temperature "
                "readings are clearly visible in the image."
            )

            # OCR debugging information
            if result.get("ocr_text"):

                with st.expander(
                    "Show OCR information"
                ):

                    st.code(
                        result["ocr_text"]
                    )


        # ====================================================
        # ANALYSIS SUCCESSFUL
        # ====================================================

        else:

            st.success(
                "✅ Analysis completed successfully."
            )

            st.divider()


            # =================================================
            # P1 / P2 / DIFFERENCE
            # =================================================

            st.subheader(
                "Temperature Readings"
            )

            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "P1 Temperature",
                    f'{result["p1"]:.1f} °C'
                )


            with col2:

                st.metric(
                    "P2 Temperature",
                    f'{result["p2"]:.1f} °C'
                )


            with col3:

                st.metric(
                    "Difference",
                    f'{result["difference"]:.1f} °C'
                )


            st.divider()


            # =================================================
            # MINIMUM / MAXIMUM
            # =================================================

            st.subheader(
                "Temperature Summary"
            )

            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Minimum Temperature",
                    f'{result["minimum"]:.1f} °C'
                )


            with col2:

                st.metric(
                    "Maximum Temperature",
                    f'{result["maximum"]:.1f} °C'
                )


            st.divider()


            # =================================================
            # FAULT DECISION
            # =================================================

            difference = result["difference"]


            if difference < 5.0:

                st.success(
                    "✅ NO FAULT DETECTED"
                )

                st.write(
                    f"The temperature difference is "
                    f"**{difference:.1f} °C**, which is "
                    f"less than the **5 °C threshold**."
                )

                st.info(
                    "No immediate attention is required."
                )


            else:

                st.error(
                    "⚠️ FAULT DETECTED"
                )

                st.write(
                    f"The temperature difference is "
                    f"**{difference:.1f} °C**, which is "
                    f"greater than or equal to the "
                    f"**5 °C threshold**."
                )

                st.warning(
                    "Attention is required within 2 days."
                )


            st.divider()


            # =================================================
            # DETAILED RESULT
            # =================================================

            st.subheader(
                "Analysis Details"
            )

            st.write(
                f'**P1:** {result["p1"]:.1f} °C'
            )

            st.write(
                f'**P2:** {result["p2"]:.1f} °C'
            )

            st.write(
                f'**Minimum:** {result["minimum"]:.1f} °C'
            )

            st.write(
                f'**Maximum:** {result["maximum"]:.1f} °C'
            )

            st.write(
                f'**Difference:** {result["difference"]:.1f} °C'
            )


            # =================================================
            # TEMPERATURE SCALE
            # =================================================

            if result.get("temperature_scale"):

                top, bottom = (
                    result["temperature_scale"]
                )

                st.caption(
                    f"Detected temperature scale: "
                    f"{top:.1f} °C to {bottom:.1f} °C"
                )


            # =================================================
            # OCR DEBUGGING
            # =================================================

            if result.get("ocr_text"):

                with st.expander(
                    "OCR information"
                ):

                    st.code(
                        result["ocr_text"]
                    )


else:

    st.info(
        "Please upload a thermal image to begin."
    )
```
