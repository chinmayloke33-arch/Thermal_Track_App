
import re
import cv2
import numpy as np
import pytesseract


def _numeric_tokens(text):
    """Return plausible numeric tokens from OCR text."""
    if not text:
        return []

    # OCR often turns 25 into 2S, 21 into 2I, etc.
    cleaned = (
        text.upper()
        .replace("O", "0")
        .replace("I", "1")
        .replace("L", "1")
        .replace("S", "5")
        .replace("B", "8")
    )

    return [
        float(x)
        for x in re.findall(r"(?<!\d)\d{1,3}(?:\.\d+)?(?!\d)", cleaned)
    ]


def _prepare_ocr_images(crop):
    """Create several high-quality OCR versions of a crop."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Upscale substantially because the camera labels are small.
    up = cv2.resize(gray, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)

    # Light blur removes JPEG noise while preserving digits.
    blur = cv2.GaussianBlur(up, (3, 3), 0)

    images = [up]

    # Otsu and adaptive threshold variants.
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    images.append(otsu)
    images.append(cv2.bitwise_not(otsu))

    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 7
    )
    images.append(adaptive)
    images.append(cv2.bitwise_not(adaptive))

    # Contrast enhancement.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    contrast = cv2.resize(
        contrast, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC
    )
    images.append(contrast)

    return images


def _ocr_crop(crop):
    """OCR a crop using several preprocessing and Tesseract modes."""
    values = []

    for img in _prepare_ocr_images(crop):
        for psm in (6, 7, 8, 10, 11, 13):
            config = (
                f"--oem 3 --psm {psm} "
                "-c tessedit_char_whitelist=0123456789"
            )

            text = pytesseract.image_to_string(img, config=config)
            values.extend(_numeric_tokens(text))

    # Keep temperatures in a realistic camera-display range.
    values = [v for v in values if -100 <= v <= 200]

    if not values:
        return None

    # Most frequent value wins.
    counts = {}
    for value in values:
        key = round(value, 2)
        counts[key] = counts.get(key, 0) + 1

    return max(counts, key=counts.get)


def _ocr_right_scale(image):
    """
    Read the temperature numbers directly from the right-hand scale area.

    This is the primary OCR method. It does not depend on finding the
    rectangular border of the number box perfectly.
    """
    h, w = image.shape[:2]

    # In the supplied camera format the scale is on the right.
    x0 = int(w * 0.84)
    x1 = w
    roi = image[0:h, x0:x1]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)

    all_candidates = []

    for img in [
        up,
        cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.adaptiveThreshold(
            up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 41, 9
        ),
    ]:
        for psm in (6, 11, 12):
            data = pytesseract.image_to_data(
                img,
                config=(
                    f"--oem 3 --psm {psm} "
                    "-c tessedit_char_whitelist=0123456789"
                ),
                output_type=pytesseract.Output.DICT,
            )

            for i, raw in enumerate(data["text"]):
                vals = _numeric_tokens(raw)
                if not vals:
                    continue

                try:
                    conf = float(data["conf"][i])
                except Exception:
                    conf = 0

                # Coordinates are in the upscaled ROI.
                cx = (data["left"][i] + data["width"][i] / 2) / 6
                cy = (data["top"][i] + data["height"][i] / 2) / 6

                for value in vals:
                    if -100 <= value <= 200 and data["width"][i] > 5:
                        all_candidates.append({
                            "value": value,
                            "x": cx + x0,
                            "y": cy,
                            "confidence": conf,
                        })

    if not all_candidates:
        return None

    # The top scale number and bottom scale number are the candidates
    # nearest the upper/lower parts of the right-side scale.
    # Group near-identical OCR readings.
    groups = {}
    for c in all_candidates:
        key = round(c["value"], 1)
        groups.setdefault(key, []).append(c)

    representatives = []
    for value, items in groups.items():
        best = max(items, key=lambda z: z["confidence"])
        representatives.append({
            "value": value,
            "y": float(np.median([z["y"] for z in items])),
            "confidence": max(z["confidence"] for z in items),
        })

    # Need two different values. Prefer candidates that are vertically far apart.
    best_pair = None
    best_score = -1

    for i in range(len(representatives)):
        for j in range(i + 1, len(representatives)):
            a = representatives[i]
            b = representatives[j]

            dy = abs(a["y"] - b["y"])
            if dy < 80:
                continue

            score = dy + 0.2 * (a["confidence"] + b["confidence"])

            if score > best_score:
                best_score = score
                best_pair = (a, b)

    if best_pair is None:
        return None

    a, b = sorted(best_pair, key=lambda z: z["y"])

    return {
        "top_temperature": float(a["value"]),
        "bottom_temperature": float(b["value"]),
        "top_y": a["y"],
        "bottom_y": b["y"],
    }


def _find_temperature_boxes(image):
    h, w = image.shape[:2]
    x0 = int(w * 0.80)
    roi = image[:, x0:]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bright = cv2.inRange(gray, 150, 255)

    contours, _ = cv2.findContours(
        bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh

        if 20 <= bw <= 120 and 12 <= bh <= 70 and 250 <= area <= 8000:
            boxes.append((x + x0, y, bw, bh))

    unique = []
    for b in sorted(boxes, key=lambda z: z[1]):
        if not any(abs(b[0] - q[0]) < 8 and abs(b[1] - q[1]) < 10 for q in unique):
            unique.append(b)

    return sorted(unique, key=lambda z: z[1])


def detect_temperature_scale(image):
    """
    Automatically detect temperature endpoints.

    Primary method: OCR the entire right-side scale.
    Fallback: locate the two light boxes and OCR enlarged crops.
    """
    ocr_result = _ocr_right_scale(image)

    boxes = _find_temperature_boxes(image)

    if ocr_result is not None:
        top_temp = ocr_result["top_temperature"]
        bottom_temp = ocr_result["bottom_temperature"]

        # The color bar is normally between the two labels.
        if boxes:
            x_center = int(np.mean([
                b[0] + b[2] / 2 for b in boxes
            ]))
            top_y = int(ocr_result["top_y"])
            bottom_y = int(ocr_result["bottom_y"])
        else:
            h, w = image.shape[:2]
            x_center = int(w * 0.94)
            top_y = int(image.shape[0] * 0.18)
            bottom_y = int(image.shape[0] * 0.86)

        y_start = top_y + 25
        y_end = bottom_y - 25

        if y_end > y_start + 20:
            return {
                "top_temperature": top_temp,
                "bottom_temperature": bottom_temp,
                "x_center": x_center,
                "y_start": y_start,
                "y_end": y_end,
            }

    # Fallback: enlarged OCR of detected boxes.
    if len(boxes) >= 2:
        top_box = boxes[0]
        bottom_box = boxes[-1]

        def enlarged_crop(box):
            x, y, bw, bh = box
            pad_x = max(20, bw)
            pad_y = max(12, bh)
            return image[
                max(0, y - pad_y):min(image.shape[0], y + bh + pad_y),
                max(0, x - pad_x):min(image.shape[1], x + bw + pad_x)
            ]

        top_temp = _ocr_crop(enlarged_crop(top_box))
        bottom_temp = _ocr_crop(enlarged_crop(bottom_box))

        if top_temp is not None and bottom_temp is not None:
            x_center = int(
                (top_box[0] + top_box[2] / 2 +
                 bottom_box[0] + bottom_box[2] / 2) / 2
            )

            y_start = top_box[1] + top_box[3] + 4
            y_end = bottom_box[1] - 4

            if y_end > y_start + 20:
                return {
                    "top_temperature": float(top_temp),
                    "bottom_temperature": float(bottom_temp),
                    "x_center": x_center,
                    "y_start": y_start,
                    "y_end": y_end,
                }

    raise ValueError(
        "Temperature labels were found, but OCR could not reliably read "
        "both values. Please use a clearer image or send this image to "
        "the developer for calibration."
    )


def _vertical_white_segments(image):
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

        if (
            0.18 * w < x < 0.90 * w
            and 0.15 * h < y < 0.70 * h
            and 1 <= bw <= 7
            and 8 <= bh <= 40
            and bw * bh >= 12
        ):
            segments.append((x, y, bw, bh))

    return segments


def detect_measurement_points(image):
    segments = _vertical_white_segments(image)

    pairs = []
    for i, a in enumerate(segments):
        ax, ay, aw, ah = a
        acx = ax + aw / 2
        acy = ay + ah / 2

        for j, b in enumerate(segments):
            if i == j:
                continue

            bx, by, bw, bh = b
            bcx = bx + bw / 2
            bcy = by + bh / 2

            if abs(acx - bcx) <= 5 and 10 <= abs(acy - bcy) <= 60:
                upper, lower = (a, b) if ay < by else (b, a)

                center_x = (
                    upper[0] + upper[2] / 2 +
                    lower[0] + lower[2] / 2
                ) / 2
                center_y = (
                    upper[1] + upper[3] / 2 +
                    lower[1] + lower[3] / 2
                ) / 2

                pairs.append({
                    "x": float(center_x),
                    "y": float(center_y),
                    "score": abs(acx - bcx),
                })

    unique = []
    for p in pairs:
        if not any(
            abs(p["x"] - q["x"]) < 8 and abs(p["y"] - q["y"]) < 8
            for q in unique
        ):
            unique.append(p)

    if len(unique) < 2:
        raise ValueError(
            "Could not automatically detect both P1 and P2 measurement points."
        )

    # The supplied camera layout has P2 on the left and P1 on the right.
    unique = sorted(unique, key=lambda p: p["x"])
    p2 = unique[0]
    p1 = unique[1]

    return {
        "P1": (p1["x"], p1["y"]),
        "P2": (p2["x"], p2["y"]),
    }


def _build_color_palette(image, scale):
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

        color = np.median(row, axis=0)
        palette.append(color)
        ys.append(y)

    return np.asarray(palette, dtype=np.float32), np.asarray(ys)


def _point_temperature(image, point, palette, ys, scale):
    x, y = int(round(point[0])), int(round(point[1]))
    radius = 9

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    samples = []

    for yy in range(max(0, y - radius), min(image.shape[0], y + radius + 1)):
        for xx in range(max(0, x - radius), min(image.shape[1], x + radius + 1)):
            h, s, v = hsv[yy, xx]

            # Ignore the white/gray marker overlay.
            if s >= 60 and v >= 25:
                samples.append(image[yy, xx].astype(np.float32))

    if len(samples) < 10:
        raise ValueError(
            f"Not enough thermal-color pixels were found around ({x}, {y})."
        )

    samples = np.asarray(samples)
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

    return float(np.median(temperatures))


def analyze_image(image):
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

    lines = [
        f"P1: {result['P1_temperature']:.2f} C",
        f"P2: {result['P2_temperature']:.2f} C",
        f"Difference: {result['difference']:.2f} C",
        result["status"],
    ]

    y = 30
    for line in lines:
        cv2.putText(
            output, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 255, 255), 2, cv2.LINE_AA
        )
        y += 28

    return output

