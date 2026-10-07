import re
from typing import Optional, Tuple

import cv2
import numpy as np
import pytesseract


# ============================================================
# THERMAL TRACK ANALYZER
# ============================================================
#
# Workflow:
#     Upload image
#          ↓
#     Detect P1/P2 labels
#          ↓
#     Read their temperatures
#          ↓
#     Calculate |P1 - P2|
#          ↓
#     Difference < 5 °C
#          → NO FAULT
#
#     Difference >= 5 °C
#          → FAULT
#          → ATTENTION WITHIN 2 DAYS
#
# ============================================================


NUMBER = r"[-+]?\d+(?:[.,]\d+)?"


# ============================================================
# BASIC NUMBER CLEANING
# ============================================================

def _clean_number(text: str) -> Optional[float]:
    """
    Convert OCR text such as:

        25.4
        25,4
        25.4*
        25.4°C

    into a Python float.
    """

    if not text:
        return None

    text = text.replace(",", ".")
    text = text.replace("O", "0")
    text = text.replace("o", "0")
    text = text.replace("I", "1")
    text = text.replace("l", "1")
    text = text.replace("|", "1")
    text = text.replace("S", "5")
    text = text.replace("s", "5")
    text = text.replace("B", "8")

    match = re.search(NUMBER, text)

    if not match:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


# ============================================================
# OCR IMAGE PREPROCESSING
# ============================================================

def _ocr_variants(image: np.ndarray):
    """
    Create several versions of the image for OCR.

    Different thermal cameras can produce different text
    contrast, so several preprocessing methods are attempted.
    """

    variants = []

    # --------------------------------------------------------
    # Original RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    variants.append(rgb)

    # --------------------------------------------------------
    # Enlarged RGB
    # --------------------------------------------------------

    enlarged = cv2.resize(
        rgb,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC,
    )

    variants.append(enlarged)

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(enlarged, cv2.COLOR_RGB2GRAY)

    variants.append(gray)

    # --------------------------------------------------------
    # CLAHE contrast enhancement
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(gray)

    variants.append(enhanced)

    # --------------------------------------------------------
    # Threshold versions
    # --------------------------------------------------------

    for threshold in (120, 150, 180, 200):

        binary = cv2.threshold(
            gray,
            threshold,
            255,
            cv2.THRESH_BINARY,
        )[1]

        variants.append(binary)

        inverted = cv2.threshold(
            gray,
            threshold,
            255,
            cv2.THRESH_BINARY_INV,
        )[1]

        variants.append(inverted)

    return variants


# ============================================================
# FIND P1 / P2 FROM OCR TEXT
# ============================================================

def _find_p1_p2_from_text(
    text: str,
) -> Tuple[Optional[float], Optional[float]]:

    if not text:
        return None, None

    # Normalize OCR mistakes.
    normalized = text

    normalized = normalized.replace("°", " ")
    normalized = normalized.replace("º", " ")
    normalized = normalized.replace("*", " ")
    normalized = normalized.replace("|", " ")
    normalized = normalized.replace("\t", " ")

    normalized = re.sub(
        r"[ ]+",
        " ",
        normalized,
    )

    p1 = None
    p2 = None

    # --------------------------------------------------------
    # Method 1:
    #
    # 25.4 P1
    # 25.4°C P1
    # P1 25.4
    # --------------------------------------------------------

    for line in normalized.splitlines():

        line = line.strip()

        if not line:
            continue

        # --------------------------------------------
        # Temperature followed by P1
        # --------------------------------------------

        match = re.search(
            rf"({NUMBER})\s*(?:C|°C)?\s*P\s*1\b",
            line,
            flags=re.IGNORECASE,
        )

        if match and p1 is None:
            p1 = _clean_number(match.group(1))

        # --------------------------------------------
        # Temperature followed by P2
        # --------------------------------------------

        match = re.search(
            rf"({NUMBER})\s*(?:C|°C)?\s*P\s*2\b",
            line,
            flags=re.IGNORECASE,
        )

        if match and p2 is None:
            p2 = _clean_number(match.group(1))

        # --------------------------------------------
        # P1 followed by temperature
        # --------------------------------------------

        match = re.search(
            rf"P\s*1\s*[:=]?\s*({NUMBER})",
            line,
            flags=re.IGNORECASE,
        )

        if match and p1 is None:
            p1 = _clean_number(match.group(1))

        # --------------------------------------------
        # P2 followed by temperature
        # --------------------------------------------

        match = re.search(
            rf"P\s*2\s*[:=]?\s*({NUMBER})",
            line,
            flags=re.IGNORECASE,
        )

        if match and p2 is None:
            p2 = _clean_number(match.group(1))

    # ========================================================
    # Method 2:
    # Search complete OCR text.
    # ========================================================

    compact = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    # P1
    if p1 is None:

        match = re.search(
            rf"({NUMBER})\s*(?:C)?\s*P\s*1\b",
            compact,
            flags=re.IGNORECASE,
        )

        if match:
            p1 = _clean_number(match.group(1))

    # P2
    if p2 is None:

        match = re.search(
            rf"({NUMBER})\s*(?:C)?\s*P\s*2\b",
            compact,
            flags=re.IGNORECASE,
        )

        if match:
            p2 = _clean_number(match.group(1))

    # P1 → temperature
    if p1 is None:

        match = re.search(
            rf"P\s*1\s*[:=]?\s*({NUMBER})",
            compact,
            flags=re.IGNORECASE,
        )

        if match:
            p1 = _clean_number(match.group(1))

    # P2 → temperature
    if p2 is None:

        match = re.search(
            rf"P\s*2\s*[:=]?\s*({NUMBER})",
            compact,
            flags=re.IGNORECASE,
        )

        if match:
            p2 = _clean_number(match.group(1))

    return p1, p2


# ============================================================
# OCR P1 / P2
# ============================================================

def read_p1_p2(image: np.ndarray):
    """
    Automatically detect P1 and P2 temperature readings.

    Returns:

        p1
        p2
        OCR text
    """

    all_text = []

    variants = _ocr_variants(image)

    # ========================================================
    # Main OCR attempts
    # ========================================================

    for variant in variants:

        for psm in (6, 11, 12, 7):

            try:

                text = pytesseract.image_to_string(
                    variant,
                    config=f"--psm {psm}",
                )

            except Exception:
                continue

            all_text.append(text)

            p1, p2 = _find_p1_p2_from_text(text)

            if p1 is not None and p2 is not None:

                return (
                    p1,
                    p2,
                    "\n".join(all_text),
                )

    # ========================================================
    # Numeric-focused OCR fallback
    # ========================================================

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.resize(
        gray,
        None,
        fx=5,
        fy=5,
        interpolation=cv2.INTER_CUBIC,
    )

    for threshold in (
        100,
        120,
        140,
        160,
        180,
        200,
        220,
    ):

        binary = cv2.threshold(
            gray,
            threshold,
            255,
            cv2.THRESH_BINARY,
        )[1]

        for psm in (6, 7, 11, 12):

            try:

                text = pytesseract.image_to_string(
                    binary,
                    config=(
                        f"--psm {psm} "
                        "-c tessedit_char_whitelist="
                        "0123456789Pp.-°C*"
                    ),
                )

            except Exception:
                continue

            all_text.append(text)

            p1, p2 = _find_p1_p2_from_text(text)

            if p1 is not None and p2 is not None:

                return (
                    p1,
                    p2,
                    "\n".join(all_text),
                )

    # ========================================================
    # Could not find both readings
    # ========================================================

    return (
        None,
        None,
        "\n".join(all_text),
    )


# ============================================================
# TEMPERATURE SCALE DETECTION
# ============================================================

def _read_scale_boxes(image: np.ndarray):
    """
    Attempt to find the small temperature boxes displayed
    near the thermal color scale.

    This information is supplementary.

    P1/P2 remain the primary measurements.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    # Light, relatively low-saturation pixels.
    mask = (
        (gray > 175)
        & (hsv[:, :, 1] < 110)
    ).astype(np.uint8) * 255

    # Clean small gaps.
    kernel = np.ones(
        (3, 3),
        np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []

    image_height, image_width = gray.shape

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        # Reasonable text box size.
        if not (
            15 <= w <= 120
            and 10 <= h <= 60
        ):
            continue

        ratio = w / float(h)

        if not (
            0.8 <= ratio <= 5.0
        ):
            continue

        # Ignore very low part of image.
        if y > image_height * 0.90:
            continue

        crop = image[
            max(0, y - 2):
            min(image_height, y + h + 2),
            max(0, x - 2):
            min(image_width, x + w + 2),
        ]

        if crop.size == 0:
            continue

        crop = cv2.resize(
            crop,
            None,
            fx=8,
            fy=8,
            interpolation=cv2.INTER_CUBIC,
        )

        crop_gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY,
        )

        # Try normal OCR.
        text = pytesseract.image_to_string(
            crop_gray,
            config=(
                "--psm 7 "
                "-c tessedit_char_whitelist=0123456789.-"
            ),
        ).strip()

        number = _clean_number(text)

        if number is not None:

            boxes.append(
                (
                    x,
                    y,
                    w,
                    h,
                    number,
                )
            )

    return boxes


def read_temperature_scale(
    image: np.ndarray,
):
    """
    Attempt to identify the top and bottom values
    of the displayed temperature scale.

    Returns:

        (top_temperature, bottom_temperature)

    or:

        None
    """

    boxes = _read_scale_boxes(
        image
    )

    if len(boxes) < 2:
        return None

    best_pair = None
    best_score = float("inf")

    for i in range(len(boxes)):

        for j in range(i + 1, len(boxes)):

            a = boxes[i]
            b = boxes[j]

            ax = a[0] + a[2] / 2
            bx = b[0] + b[2] / 2

            x_difference = abs(
                ax - bx
            )

            # They should be approximately vertically aligned.
            if x_difference > max(
                a[2],
                b[2],
            ) * 2.0:
                continue

            # Determine top and bottom.
            if a[1] < b[1]:

                top = a
                bottom = b

            else:

                top = b
                bottom = a

            vertical_gap = (
                bottom[1]
                - (top[1] + top[3])
            )

            if vertical_gap < 20:
                continue

            # Prefer reasonably aligned boxes.
            score = (
                x_difference
                + abs(vertical_gap - 200) * 0.01
            )

            if score < best_score:

                best_score = score

                best_pair = (
                    top[4],
                    bottom[4],
                )

    return best_pair


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_thermal_image(
    image_rgb: np.ndarray,
):
    """
    Main function called by Streamlit.

    Input:
        RGB image as NumPy array.

    Output:
        Dictionary containing:

        p1
        p2
        minimum
        maximum
        difference
        status
        action
    """

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if image_rgb is None:

        return {
            "success": False,
            "message": "No image was provided.",
        }

    if len(image_rgb.shape) != 3:

        return {
            "success": False,
            "message": "Invalid image format.",
        }

    # --------------------------------------------------------
    # RGB → BGR for OpenCV
    # --------------------------------------------------------

    try:

        image_bgr = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR,
        )

    except Exception as error:

        return {
            "success": False,
            "message": (
                f"Could not process image: {error}"
            ),
        }

    # ========================================================
    # STEP 1 — READ P1/P2
    # ========================================================

    p1, p2, ocr_text = read_p1_p2(
        image_bgr
    )

    # --------------------------------------------------------
    # P1/P2 could not be detected
    # --------------------------------------------------------

    if p1 is None or p2 is None:

        return {
            "success": False,

            "message": (
                "The app could not reliably read "
                "both P1 and P2 temperature values "
                "from this image."
            ),

            "ocr_text": ocr_text,
        }

    # ========================================================
    # STEP 2 — CALCULATE RESULTS
    # ========================================================

    minimum = min(
        p1,
        p2,
    )

    maximum = max(
        p1,
        p2,
    )

    difference = abs(
        p1 - p2
    )

    # ========================================================
    # STEP 3 — APPLY 5 °C RULE
    # ========================================================

    if difference < 5.0:

        status = "NO FAULT DETECTED"

        action = (
            "No immediate attention is required."
        )

    else:

        status = "FAULT DETECTED"

        action = (
            "Attention is required within 2 days."
        )

    # ========================================================
    # STEP 4 — OPTIONAL TEMPERATURE SCALE
    # ========================================================

    temperature_scale = (
        read_temperature_scale(
            image_bgr
        )
    )

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "success": True,

        # P1/P2
        "p1": float(p1),
        "p2": float(p2),

        # Min / max
        "minimum": float(minimum),
        "maximum": float(maximum),

        # Difference
        "difference": float(
            difference
        ),

        # Decision
        "status": status,
        "action": action,

        # Optional scale information
        "temperature_scale": (
            temperature_scale
        ),

        # OCR debugging information
        "ocr_text": ocr_text,
    }
