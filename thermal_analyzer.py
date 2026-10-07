
import re
import cv2
import numpy as np
import pytesseract


def _ocr_number(crop, whitelist="0123456789.-"):
    """Read a number from a small image crop."""
    if crop is None or crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # The camera places digits inside a light gray box; removing the border
    # greatly improves OCR accuracy.
    hh, ww = gray.shape[:2]
    if ww >= 20 and hh >= 10:
        gray = gray[max(0, int(hh * 0.12)):max(1, int(hh * 0.88)),
                    max(0, int(ww * 0.25)):min(ww, int(ww * 0.75))]
    scale = 10
    up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    candidates = [up]
    for threshold in (140, 160, 180, 200):
        candidates.append(cv2.threshold(up, threshold, 255, cv2.THRESH_BINARY)[1])

    values = []
    for img in candidates:
        for psm in (7, 8, 10):
            text = pytesseract.image_to_string(
                img,
                config=f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
            ).strip()
            m = re.search(r"-?\d+(?:\.\d+)?", text)
            if m:
                try:
                    values.append(float(m.group()))
                except ValueError:
                    pass

    if not values:
        return None

    # Most common OCR value; this is more stable than trusting one OCR pass.
    counts = {}
    for v in values:
        key = round(v, 3)
        counts[key] = counts.get(key, 0) + 1

    return max(counts, key=counts.get)


def _find_temperature_boxes(image):
    """
    Find the two light rectangular boxes containing the temperature
    scale numbers, normally on the right side of the thermal image.
    """
    h, w = image.shape[:2]
    x0 = int(w * 0.82)
    roi = image[:, x0:]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bright = cv2.inRange(gray, 160, 255)

    contours, _ = cv2.findContours(
        bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh

        if 25 <= bw <= 100 and 15 <= bh <= 50 and 400 <= area <= 5000:
            # Convert ROI coordinates back to image coordinates.
            boxes.append((x + x0, y, bw, bh))

    # Remove near-duplicates.
    unique = []
    for b in sorted(boxes, key=lambda z: z[1]):
        if not any(abs(b[0] - q[0]) < 5 and abs(b[1] - q[1]) < 8 for q in unique):
            unique.append(b)

    return sorted(unique, key=lambda z: z[1])


def detect_temperature_scale(image):
    """Automatically detect the top/bottom temperature labels and color bar."""
    boxes = _find_temperature_boxes(image)

    if len(boxes) < 2:
        raise ValueError(
            "Could not automatically find the two temperature-scale labels."
        )

    # Normally the first and last boxes are the upper and lower scale values.
    top_box = boxes[0]
    bottom_box = boxes[-1]

    def crop_box(b, pad=3):
        x, y, w, h = b
        return image[
            max(0, y - pad):min(image.shape[0], y + h + pad),
            max(0, x - pad):min(image.shape[1], x + w + pad)
        ]

    top_temp = _ocr_number(crop_box(top_box))
    bottom_temp = _ocr_number(crop_box(bottom_box))

    if top_temp is None or bottom_temp is None:
        raise ValueError(
            "Temperature labels were found, but OCR could not read both values."
        )

    # Color-bar x-position is near the center of the temperature boxes.
    x_center = int(round((top_box[0] + top_box[2] / 2 +
                          bottom_box[0] + bottom_box[2] / 2) / 2))

    # The actual color bar lies between the two number boxes.
    y_start = top_box[1] + top_box[3] + 7
    y_end = bottom_box[1] - 12

    if y_end <= y_start + 20:
        raise ValueError("Could not determine the vertical temperature color bar.")

    return {
        "top_temperature": float(top_temp),
        "bottom_temperature": float(bottom_temp),
        "x_center": x_center,
        "y_start": y_start,
        "y_end": y_end,
        "top_box": top_box,
        "bottom_box": bottom_box,
    }


def _vertical_white_segments(image):
    """Find narrow vertical white/gray marker segments."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    white = cv2.inRange(gray, 200, 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    vertical = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = image.shape[:2]
    segments = []

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)

        # Marker segments are small vertical bars.
        if (
            0.25 * w < x < 0.90 * w
            and 0.15 * h < y < 0.70 * h
            and 1 <= bw <= 6
            and 9 <= bh <= 35
            and bw * bh >= 12
        ):
            segments.append((x, y, bw, bh))

    return segments


def detect_measurement_points(image):
    """
    Detect the camera's P1/P2 crosshair markers.

    A marker consists of two aligned vertical white segments separated
    by a small gap. The center between those segments is the measurement
    point.
    """
    segments = _vertical_white_segments(image)

    pairs = []
    for a in segments:
        ax, ay, aw, ah = a
        acx = ax + aw / 2
        acy = ay + ah / 2

        for b in segments:
            if b is a:
                continue

            bx, by, bw, bh = b
            bcx = bx + bw / 2
            bcy = by + bh / 2

            # Same vertical line, separated by the horizontal marker.
            if abs(acx - bcx) <= 4 and 12 <= abs(acy - bcy) <= 55:
                upper, lower = (a, b) if ay < by else (b, a)

                # Avoid treating two parts of the same line as a duplicate.
                center_x = (upper[0] + upper[2] / 2 +
                            lower[0] + lower[2] / 2) / 2
                center_y = (
                    upper[1] + upper[3] / 2 +
                    lower[1] + lower[3] / 2
                ) / 2

                score = abs(acx - bcx) + abs((lower[1] - (upper[1] + upper[3])) - 18)

                pairs.append({
                    "x": float(center_x),
                    "y": float(center_y),
                    "score": float(score),
                })

    # Deduplicate centers.
    unique = []
    for p in sorted(pairs, key=lambda q: q["score"]):
        if not any(
            abs(p["x"] - q["x"]) < 8 and abs(p["y"] - q["y"]) < 8
            for q in unique
        ):
            unique.append(p)

    if len(unique) < 2:
        raise ValueError(
            "Could not automatically detect both P1 and P2 measurement points."
        )

    # For this camera layout, the point numbered 1 is normally the right-hand
    # marker and point numbered 2 is the left-hand marker. We also try OCR
    # locally to confirm the point numbers.
    unique = sorted(unique, key=lambda p: p["x"])

    labelled = []
    for p in unique[:6]:
        x, y = int(round(p["x"])), int(round(p["y"]))
        h, w = image.shape[:2]

        x1, x2 = max(0, x - 30), min(w, x + 30)
        y1, y2 = max(0, y - 8), min(h, y + 42)

        crop = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        up = cv2.resize(gray, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)

        number = None
        for threshold in (140, 160, 180, 200):
            binary = cv2.threshold(up, threshold, 255, cv2.THRESH_BINARY)[1]
            text = pytesseract.image_to_string(
                binary,
                config="--psm 10 -c tessedit_char_whitelist=12"
            )
            if "1" in text:
                number = 1
                break
            if "2" in text:
                number = 2
                break

        labelled.append({**p, "label": number})

    p1 = next((p for p in labelled if p["label"] == 1), None)
    p2 = next((p for p in labelled if p["label"] == 2), None)

    if p1 is None or p2 is None:
        # Fallback for the common camera layout:
        # left marker = P2, right marker = P1.
        p2, p1 = labelled[0], labelled[1]

    return {
        "P1": (float(p1["x"]), float(p1["y"])),
        "P2": (float(p2["x"]), float(p2["y"])),
    }


def _build_color_palette(image, scale):
    """Build a vertical BGR palette from the detected color bar."""
    x = int(scale["x_center"])
    y1 = int(scale["y_start"])
    y2 = int(scale["y_end"])

    x1 = max(0, x - 7)
    x2 = min(image.shape[1], x + 8)

    palette = []
    ys = []

    for y in range(y1, y2 + 1):
        row = image[y, x1:x2]

        if row.size == 0:
            continue

        # Median is less affected by the black border and compression noise.
        color = np.median(row, axis=0)
        palette.append(color)
        ys.append(y)

    return np.asarray(palette, dtype=np.float32), np.asarray(ys)


def _point_temperature(image, point, palette, ys, scale):
    """
    Estimate temperature around a point by matching nearby thermal colors
    to the detected color-bar palette.
    """
    x, y = int(round(point[0])), int(round(point[1]))
    radius = 9

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    samples = []

    for yy in range(max(0, y - radius), min(image.shape[0], y + radius + 1)):
        for xx in range(max(0, x - radius), min(image.shape[1], x + radius + 1)):
            h, s, v = hsv[yy, xx]

            # Ignore the white/gray camera crosshair overlay.
            if s >= 60 and v >= 25:
                samples.append(image[yy, xx].astype(np.float32))

    if len(samples) < 10:
        raise ValueError(
            f"Not enough thermal-color pixels were found around point ({x}, {y})."
        )

    samples = np.asarray(samples)

    # Convert each sample to the nearest color-bar position.
    temperatures = []

    for color in samples:
        distances = np.linalg.norm(palette - color, axis=1)
        idx = int(np.argmin(distances))
        bar_y = ys[idx]

        frac = (bar_y - scale["y_start"]) / max(
            1, scale["y_end"] - scale["y_start"]
        )

        temp = (
            scale["top_temperature"]
            + frac * (scale["bottom_temperature"] - scale["top_temperature"])
        )

        temperatures.append(temp)

    # The point marker sits on a hot rail while nearby background is often
    # much colder.  Taking the upper quartile suppresses the background and
    # the white/black marker overlay while retaining the rail temperature.
    return float(np.percentile(temperatures, 75))


def analyze_image(image):
    """Complete automatic analysis."""
    if image is None:
        raise ValueError("No image was provided.")

    scale = detect_temperature_scale(image)
    points = detect_measurement_points(image)

    palette, ys = _build_color_palette(image, scale)

    p1_temp = _point_temperature(
        image, points["P1"], palette, ys, scale
    )
    p2_temp = _point_temperature(
        image, points["P2"], palette, ys, scale
    )

    difference = abs(p1_temp - p2_temp)

    if difference < 5:
        status = "NO FAULT DETECTED"
        action = "No immediate attention required."
    else:
        status = "FAULT DETECTED"
        action = "Attention is required within 2 days."

    return {
        "P1_temperature": p1_temp,
        "P2_temperature": p2_temp,
        "difference": difference,
        "status": status,
        "action": action,
        "scale_top": scale["top_temperature"],
        "scale_bottom": scale["bottom_temperature"],
        "P1_point": points["P1"],
        "P2_point": points["P2"],
    }


def create_result_image(image, result):
    """Draw detected P1/P2 and analysis information on the image."""
    output = image.copy()

    for label in ("P1", "P2"):
        x, y = result[f"{label}_point"]
        x, y = int(round(x)), int(round(y))

        cv2.circle(output, (x, y), 8, (255, 255, 255), 2)
        cv2.putText(
            output,
            label,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    text_lines = [
        f"P1: {result['P1_temperature']:.2f} C",
        f"P2: {result['P2_temperature']:.2f} C",
        f"Difference: {result['difference']:.2f} C",
        result["status"],
    ]

    y = 30
    for line in text_lines:
        cv2.putText(
            output,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28

    return output
