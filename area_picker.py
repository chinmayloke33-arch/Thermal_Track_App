"""
area_picker.py - draw rectangles (click and drag) on the thermal image
instead of clicking single points.

  Maximum area   -> hottest value inside the rectangle
  Minimum area A -> typical (median) value inside the rectangle
  Minimum area B -> optional second reference, the higher one is used

Requires:  streamlit-image-coordinates >= 0.2.0   (click_and_drag support)
           e.g. in requirements.txt:  streamlit-image-coordinates>=0.2.0

Usage in Appcure.py (same return value as render_point_picker):

    from area_picker import render_area_picker
    result = render_area_picker(image, file_key=file_key)
"""

import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from thermal_core import analyze_thermal_image, analyze_with_manual_bar

DISPLAY_WIDTH = 700
MAX_PERCENTILE = 99.5   # use 100 for the absolute hottest pixel
MIN_BOX = 8             # a plain click becomes a box of +/- this many px
MIN_VALID_PIXELS = 10   # fewer valid pixels in the box -> ask to redraw


def area_temperature(tmap, box, mode="max"):
    """Temperature inside box=(x1, y1, x2, y2) in tmap pixel coordinates.

    mode="max":    MAX_PERCENTILE of the valid pixels (hottest in the area)
    mode="median": median of the valid pixels (reference area)
    Returns None when the box holds too few palette-matched pixels
    (for example it only covers text).
    """
    h, w = tmap.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0, int(x1)), min(w, int(x2))))
    y1, y2 = sorted((max(0, int(y1)), min(h, int(y2))))
    win = tmap[y1:y2, x1:x2]
    v = win[~np.isnan(win)]
    if v.size < MIN_VALID_PIXELS:
        return None
    if mode == "max":
        return float(np.percentile(v, MAX_PERCENTILE))
    return float(np.median(v))


def _empty():
    return {"max": None, "minA": None, "minB": None,
            "barTop": None, "barBot": None, "last": None}


def _to_box(click, W, H):
    """Convert a component result (display pixels) to an original-pixel box."""
    s = W / click["width"]
    if "x1" in click:
        x1, y1, x2, y2 = (click["x1"], click["y1"],
                          click["x2"], click["y2"])
    else:  # plain click from an older component version
        x1 = x2 = click["x"]
        y1 = y2 = click["y"]
    x1, x2 = sorted((x1 * s, x2 * s))
    y1, y2 = sorted((y1 * s, y2 * s))
    if (x2 - x1) < 4 and (y2 - y1) < 4:  # click, not a drag
        x1, x2 = x1 - MIN_BOX, x2 + MIN_BOX
        y1, y2 = y1 - MIN_BOX, y2 + MIN_BOX
    return (max(0, int(x1)), max(0, int(y1)),
            min(W, int(x2)), min(H, int(y2)))


def render_area_picker(pil_image, file_key):
    ss = st.session_state
    k = f"ap_{file_key}"
    ss.setdefault(k, _empty())
    pts = ss[k]

    # Temperature map is computed once per image
    if f"{k}_res" not in ss:
        with st.spinner("Reading colour scale..."):
            ss[f"{k}_res"] = analyze_thermal_image(pil_image)
    res = ss[f"{k}_res"]
    tmap = res.get("_tmap")

    manual = tmap is None
    if manual:
        st.warning(
            "The colour bar was not detected automatically. Enter the "
            "scale limits shown on the image, then click the middle of "
            "the colour bar at its TOP end and BOTTOM end."
        )
        m1, m2 = st.columns(2)
        txt_max = m1.text_input("Scale top (max) °C", key=f"{k}_smax")
        txt_min = m2.text_input("Scale bottom (min) °C", key=f"{k}_smin")
        try:
            sc_max, sc_min = float(txt_max), float(txt_min)
        except ValueError:
            sc_max = sc_min = None
    else:
        for w_ in res["warnings"]:
            st.warning(w_)
        st.caption(
            f"Scale read: {res['min']} to {res['max']} °C "
            f"({res['source']})"
        )

    target = st.radio(
        "Draw on the image (click and drag) to set:",
        ["Maximum area", "Minimum area A", "Minimum area B (optional)"]
        + (["Scale bar TOP", "Scale bar BOTTOM"] if manual else []),
        horizontal=True,
        key=f"{k}_target",
    )
    key_of = {
        "Maximum area": "max",
        "Minimum area A": "minA",
        "Minimum area B (optional)": "minB",
        "Scale bar TOP": "barTop",
        "Scale bar BOTTOM": "barBot",
    }[target]

    rgb = np.array(pil_image.convert("RGB"))
    H, W = rgb.shape[:2]
    scale = DISPLAY_WIDTH / W
    disp = cv2.resize(rgb, (DISPLAY_WIDTH, int(H * scale)))
    colours = {"max": (255, 40, 40), "minA": (40, 160, 255),
               "minB": (40, 255, 160),
               "barTop": (255, 220, 0), "barBot": (255, 220, 0)}
    labels = {"max": "MAX", "minA": "MIN A", "minB": "MIN B",
              "barTop": "BAR TOP", "barBot": "BAR BOTTOM"}

    for name, p in pts.items():
        if name == "last" or p is None:
            continue
        if name in ("barTop", "barBot"):  # still single points
            cx, cy = int(p[0] * scale), int(p[1] * scale)
            cv2.drawMarker(disp, (cx, cy), colours[name],
                           cv2.MARKER_CROSS, 22, 2)
            cv2.putText(disp, labels[name], (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colours[name], 2)
        else:
            x1, y1, x2, y2 = [int(v * scale) for v in p]
            cv2.rectangle(disp, (x1, y1), (x2, y2), colours[name], 2)
            cv2.putText(disp, labels[name], (x1 + 4, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colours[name], 2)

    click = streamlit_image_coordinates(
        disp, key=f"{k}_img", click_and_drag=True
    )

    # Process a NEW selection only (the component repeats the last value)
    if click and click != pts["last"]:
        pts["last"] = click
        if key_of in ("barTop", "barBot"):
            s = W / click["width"]
            cx = click.get("x1", click.get("x"))
            cy = click.get("y1", click.get("y"))
            pts[key_of] = (int(cx * s), int(cy * s))
        else:
            pts[key_of] = _to_box(click, W, H)
        st.rerun()

    col1, col2 = st.columns(2)
    if col2.button("Clear areas", key=f"{k}_clear"):
        ss[k] = _empty()
        st.rerun()

    if manual:
        if sc_max is None or pts["barTop"] is None or pts["barBot"] is None:
            st.info("Enter scale limits and click the bar TOP and BOTTOM.")
            return None
        ck = (pts["barTop"], pts["barBot"], sc_max, sc_min)
        if ss.get(f"{k}_mkey") != ck:
            ss[f"{k}_mkey"] = ck
            ss[f"{k}_mres"] = analyze_with_manual_bar(
                pil_image, pts["barTop"], pts["barBot"], sc_max, sc_min
            )
        mres = ss.get(f"{k}_mres")
        if not mres:
            st.error("Bar points or scale limits are not valid.")
            return None
        tmap = mres["_tmap"]

    if pts["max"] is None or pts["minA"] is None:
        st.info("Draw a Maximum area and at least Minimum area A.")
        return None

    t_max = area_temperature(tmap, pts["max"], "max")
    t_a = area_temperature(tmap, pts["minA"], "median")
    t_b = (area_temperature(tmap, pts["minB"], "median")
           if pts["minB"] else None)

    if t_max is None or t_a is None or (pts["minB"] and t_b is None):
        st.error(
            "A selected area is mostly text or does not match the colour "
            "scale. Draw it again over the image."
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
            "source": "User-selected areas (colour-scale mapped)"}
