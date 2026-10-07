import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from thermal_analyzer import analyze_image, create_result_image


st.set_page_config(
    page_title="Thermal Track Inspection",
    page_icon="🌡️",
    layout="wide",
)

st.title("🌡️ Thermal Track Inspection")
st.write(
    "Upload a thermal image. The application automatically detects "
    "the P1/P2 measurement points and the temperature scale."
)

uploaded = st.file_uploader(
    "Upload thermal image",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    pil_image = Image.open(uploaded).convert("RGB")
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    st.subheader("Uploaded Image")
    st.image(pil_image, use_container_width=True)

    if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
        with st.spinner("Automatically detecting P1, P2 and temperature scale..."):
            try:
                result = analyze_image(image)

                st.success("Analysis completed successfully.")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "P1 Temperature",
                    f"{result['P1_temperature']:.2f} °C",
                )

                c2.metric(
                    "P2 Temperature",
                    f"{result['P2_temperature']:.2f} °C",
                )

                c3.metric(
                    "Temperature Difference",
                    f"{result['difference']:.2f} °C",
                )

                st.write(
                    f"**Temperature scale detected:** "
                    f"{result['scale_bottom']:.1f} °C to "
                    f"{result['scale_top']:.1f} °C"
                )

                if result["difference"] < 5:
                    st.success(
                        f"✅ {result['status']} — {result['action']}"
                    )
                else:
                    st.error(
                        f"⚠️ {result['status']} — {result['action']}"
                    )

                st.subheader("Detected Points")
                annotated = create_result_image(image, result)
                annotated_rgb = cv2.cvtColor(
                    annotated, cv2.COLOR_BGR2RGB
                )
                st.image(
                    annotated_rgb,
                    caption="Automatically detected P1 and P2",
                    use_container_width=True,
                )

                ok, encoded = cv2.imencode(".jpg", annotated)
                if ok:
                    st.download_button(
                        "⬇️ Download Result Image",
                        data=encoded.tobytes(),
                        file_name="thermal_track_result.jpg",
                        mime="image/jpeg",
                    )

            except Exception as e:
                st.error(
                    "Automatic analysis could not be completed for this image."
                )
                st.warning(str(e))
                st.info(
                    "If this image format differs from the camera format used "
                    "for development, send the image to the developer so the "
                    "automatic detector can be adjusted."
                )
else:
    st.info("Upload a thermal image to begin.")
