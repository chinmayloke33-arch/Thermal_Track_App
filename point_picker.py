"""
point_picker.py - let the user click the Max point and the Min point(s)
on the thermal image, then show the result.

pip install streamlit-image-coordinates   (add to requirements.txt)

Usage in Appcure.py, after the image is uploaded:

    from point_picker import render_point_picker
    result = render_point_picker(image, file_key=image_id)
    if result:
        maximum_temperature = result["max"]
        minimum_temperature = result["min"]
        difference = result["delta"]
"""

import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from thermal_core import analyze_thermal_image, point_temperature

DISPLAY_WIDTH = 700
RADIUS = 6  # pixels around the click that are examined


def render_point_picker(pil_image, file_key):
    ss = st.session_state
    k = f"pp_{file_key}"
    ss.setdefault(k, {"max": None, "minA": None, "minB": None, "last": None})
    pts = ss[k]

    # Temperature map is computed once per image
    if f"{k}_res" not in ss:
        with st.spinner("Reading colour scale..."):
            ss[f"{k}_res"] = analyze_thermal_image(pil_image)
    res = ss[f"{k}_res"]
    tmap = res.get("_tmap")

    if tmap is None:
        st.error(
            "Could not read the colour scale from this image, so point "
            "temperatures are unavailable. " + " ".join(res["warnings"])
        )
        return None
    for w in res["warnings"]:
        st.warning(w)

    st.caption(
        f"Scale read: {res['min']} to {res['max']} °C "
        f"({res['source']})"
    )

    target = st.radio(
        "Click on the image to set:",
        ["Maximum point", "Minimum point A", "Minimum point B (optional)"],
        horizontal=True,
        key=f"{k}_target",
    )
    key_of = {
        "Maximum point": "max",
        "Minimum point A": "minA",
        "Minimum point B (optional)": "minB",
    }[target]

    # Draw markers on a copy, at display size
    rgb = np.array(pil_image.convert("RGB"))
    H, W = rgb.shape[:2]
    scale = DISPLAY_WIDTH / W
    disp = cv2.resize(rgb, (DISPLAY_WIDTH, int(H * scale)))
    colours = {"max": (255, 40, 40), "minA": (40, 160, 255),
               "minB": (40, 255, 160)}
    labels = {"max": "MAX", "minA": "MIN A", "minB": "MIN B"}
    for name, p in pts.items():
        if name == "last" or p is None:
            continue
        cx, cy = int(p[0] * scale), int(p[1] * scale)
        cv2.drawMarker(disp, (cx, cy), colours[name],
                       cv2.MARKER_CROSS, 22, 2)
        cv2.putText(disp, labels[name], (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colours[name], 2)

    click = streamlit_image_coordinates(disp, key=f"{k}_img")

    # Process a NEW click only (the component repeats the last value)
    if click and click != pts["last"]:
        pts["last"] = click
        # convert displayed pixels -> original pixels
        sx = W / click["width"]
        pts[key_of] = (int(click["x"] * sx), int(click["y"] * sx))
        st.rerun()

    col1, col2 = st.columns(2)
    if col2.button("Clear points", key=f"{k}_clear"):
        ss[k] = {"max": None, "minA": None, "minB": None, "last": None}
        st.rerun()

    if pts["max"] is None or pts["minA"] is None:
        st.info("Select a Maximum point and at least Minimum point A.")
        return None

    t_max = point_temperature(tmap, *pts["max"], RADIUS, "max")
    t_a = point_temperature(tmap, *pts["minA"], RADIUS, "median")
    t_b = (point_temperature(tmap, *pts["minB"], RADIUS, "median")
           if pts["minB"] else None)

    if t_max is None or t_a is None or (pts["minB"] and t_b is None):
        st.error(
            "A selected point is on text or a spot that does not match "
            "the colour scale. Click again nearby."
        )
        return None

    t_min = max(v for v in (t_a, t_b) if v is not None)  # higher of A / B
    delta = round(t_max - t_min, 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Maximum °C", f"{t_max:.1f}")
    c2.metric("Minimum °C", f"{t_min:.1f}")
    c3.metric("Difference °C", f"{delta:.1f}")
    if t_b is not None:
        st.caption(f"Min A = {t_a:.1f} °C, Min B = {t_b:.1f} °C "
                   "(the higher one is used)")

    return {"max": round(t_max, 1), "min": round(t_min, 1), "delta": delta,
            "source": "User-selected points (colour-scale mapped)"}
