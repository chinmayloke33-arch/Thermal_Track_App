"""
thermal_core.py - improved temperature extraction for Appcure.py

Drop-in replacement for extract_scale_temperatures_ocr() and
predict_temperature(). Key changes:

1. Finds the colour bar in the image instead of guessing crop areas.
2. OCRs only the top and bottom labels of that bar (digits only).
3. Votes across several preprocessing variants; rejects implausible reads.
4. Maps every pixel to a temperature using the bar's own palette, so the
   app can report the true hottest spot, not just the scale limits.
5. Removes the hard-coded 7.2 / 39.5 fallback and returns a confidence
   flag so unreliable results are never silently saved as fact.
"""

import re
from collections import Counter

import cv2
import numpy as np
import pytesseract

NUM_RE = re.compile(r"(?<!\d)(-?\d{1,3}(?:\.\d{1,2})?)(?!\d)")
MIN_PLAUSIBLE, MAX_PLAUSIBLE = -80.0, 150.0


# ------------------------------------------------------------
# 1. Locate the colour bar
# ------------------------------------------------------------
def _find_colorbar_gradient(rgb):
    """
    Return (x1, y1, x2, y2) of the vertical colour bar, or None.

    Bar columns have (a) a large colour change top-to-bottom and
    (b) almost identical colour to the columns a few pixels to the
    right (horizontal uniformity). Flat background has no change; scene
    detail and text are not horizontally uniform. The bar must also be
    tall (>= 20% of image height). Searched in the right 35% of the
    image, where scales normally sit.
    """
    h, w = rgb.shape[:2]
    x_start = int(w * 0.65)
    lab = cv2.cvtColor(rgb[:, x_start:], cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (0, 0), 1.5)  # suppress JPEG noise
    wp = lab.shape[1]

    span = np.linalg.norm(lab.max(axis=0) - lab.min(axis=0), axis=1)  # (w',)
    hu = np.full(wp, 99.0, dtype=np.float32)
    shift = 2
    hu[:wp - shift] = np.linalg.norm(
        lab[:, :wp - shift] - lab[:, shift:], axis=2
    ).mean(axis=0)

    vstep_all = np.linalg.norm(np.diff(lab, axis=0), axis=2)   # (h-1, w')
    active = (vstep_all > 0.5).sum(axis=0)                     # rows changing

    ok = (span > 45) & (hu < 6) & (active > h * 0.12)
    cols = np.where(ok)[0]
    if cols.size < 4:
        return None

    runs = np.split(cols, np.where(np.diff(cols) > 2)[0] + 1)
    run = max(runs, key=len)
    if len(run) < 4 or len(run) > w * 0.12:
        return None

    x1, x2 = int(run[0] + x_start), int(run[-1] + x_start)
    mid = lab[:, run[len(run) // 2]]
    vstep = np.linalg.norm(np.diff(mid, axis=0), axis=1)
    rows = np.where(vstep > 0.5)[0]
    if rows.size < h * 0.12:
        return None
    seg = np.split(rows, np.where(np.diff(rows) > 4)[0] + 1)
    seg = max(seg, key=len)
    if len(seg) < h * 0.12:
        return None
    # trim blur-spread at the ends so palette rows are pure bar colour
    y1, y2 = int(seg[0]) + 4, int(seg[-1]) - 3
    return x1, y1, x2, y2


def find_colorbar(rgb):
    """
    Edge method first (works when the bar sits against a smooth
    background gradient or at the image border), then the gradient
    method.
    """
    bar = _find_colorbar_edges(rgb)
    if bar is not None:
        return refine_bar_ends(rgb, bar)
    return _find_colorbar_gradient(rgb)


def _find_colorbar_edges(rgb):
    """
    The bar has a sharp vertical left edge that runs for the bar's whole
    height; smooth scene gradients and thin wires do not. Find that edge,
    then take the bar from just inside it to the next strong edge or the
    image border. Returns (x1, y1, x2, y2) or None.
    """
    h, w = rgb.shape[:2]
    x_start = int(w * 0.65)
    lab = cv2.cvtColor(rgb[:, x_start:], cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (0, 0), 0.8)
    wp = lab.shape[1]

    d = np.linalg.norm(lab[:, :wp - 2] - lab[:, 2:], axis=2)   # (h, wp-2)
    strong = d > 20
    edge_len = strong.sum(axis=0)
    need = int(h * 0.12)
    cand = np.where(edge_len >= need)[0]
    if cand.size == 0:
        return None

    # Edge groups = left / right borders of the bar. If the bar sits
    # inside the image there are two groups close together; if it touches
    # the image border only its left edge exists.
    groups = np.split(cand, np.where(np.diff(cand) > 3)[0] + 1)
    best = max(int(edge_len[g].max()) for g in groups)
    # keep only edges nearly as long as the strongest (drops weak texture)
    groups = [g for g in groups if edge_len[g].max() >= 0.5 * best]

    def peak(g):
        return int(g[np.argmax(edge_len[g])])

    right_edge = None
    if len(groups) >= 2:
        gl, gr = peak(groups[-2]), peak(groups[-1])
        if 4 <= gr - gl <= w * 0.12:
            left, right_edge = gl, gr
    if right_edge is None:
        left = peak(groups[-1])

    x1 = x_start + left + 3
    x2 = (x_start + right_edge - 1) if right_edge is not None else (w - 3)
    if x2 - x1 < 3 or (x2 - x1) > w * 0.12:
        return None

    # Vertical extent: rows where the left edge is strong AND the bar
    # columns are horizontally uniform (excludes label text that overlaps
    # the bar's columns). Gaps up to 60 rows are bridged because the cold
    # end of the bar can look like the background for a while.
    cx1, cx2 = x1 - x_start, x2 - x_start
    block = lab[:, cx1:cx2 + 1]
    row_dev = np.linalg.norm(block - block.mean(axis=1, keepdims=True), axis=2)
    row_dev = row_dev.mean(axis=1)
    ok_rows = strong[:, left] & (row_dev < 8)
    rows = np.where(ok_rows)[0]
    if rows.size == 0:
        return None
    seg = np.split(rows, np.where(np.diff(rows) > 60)[0] + 1)
    seg = max(seg, key=len)
    if (seg[-1] - seg[0]) < need:
        return None
    y1, y2 = int(seg[0]) + 2, int(seg[-1]) - 1
    return x1, y1, x2, y2


def refine_bar_ends(rgb, bar):
    """
    The bar's colours change smoothly row to row, then jump where the bar
    ends. Use that jump to correct y1 / y2 (the edge method can stop short
    where the cold end of the bar resembles the background).
    """
    x1, y1, x2, y2 = bar
    h = rgb.shape[0]
    c = (x1 + x2) // 2
    col = cv2.cvtColor(
        rgb[:, max(0, c - 1):c + 2].astype(np.uint8), cv2.COLOR_RGB2LAB
    ).astype(np.float32).mean(axis=1)                    # (h, 3)
    step = np.linalg.norm(np.diff(col, axis=0), axis=1)   # len h-1
    inner = step[y1 + 8:max(y1 + 9, y2 - 8)]
    if inner.size == 0:
        return bar
    thr = max(6.0, 4.0 * float(np.median(inner)))

    new_y2 = y2
    for y in range(max(y1 + 20, y2 - 10), min(h - 2, y2 + 90)):
        if step[y] > thr:
            new_y2 = y
            break
    new_y1 = y1
    for y in range(y1 + 10, max(0, y1 - 45), -1):
        if step[y] > thr:
            new_y1 = y + 1
            break
    if new_y2 - new_y1 < 0.12 * h:
        return bar
    return x1, new_y1, x2, new_y2


def read_palette(rgb, bar):
    """
    Top-to-bottom list of RGB colours along the bar (hot -> cold).
    Rows at either end that jump abruptly from their neighbour (bar
    border or background bleeding in) are dropped.
    """
    x1, y1, x2, y2 = bar
    inner = rgb[y1:y2 + 1, x1 + 2:max(x1 + 3, x2 - 1)]
    pal = np.median(inner, axis=1)  # (rows, 3)

    lab = cv2.cvtColor(
        pal.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB
    ).reshape(-1, 3).astype(np.float32)
    jump = np.linalg.norm(np.diff(lab, axis=0), axis=1)  # len n-1

    lo, hi = 0, len(pal)
    k = max(3, int(len(pal) * 0.08))
    head = jump[:k]
    if head.size and head.max() > 25:
        lo = int(head.argmax()) + 1
    tail = jump[-k:]
    if tail.size and tail.max() > 25:
        hi = len(jump) - k + int(tail.argmax()) + 1
    if hi - lo < 10:
        lo, hi = 0, len(pal)
    return pal[lo:hi]


# ------------------------------------------------------------
# 2. OCR the bar labels
# ------------------------------------------------------------
_TOKEN = re.compile(r"-?\d+\.?\d*")


def _normalise_token(tok):
    """'25.5.' -> -?25.5 ; '227' -> 22.7 (labels always have 1 decimal)."""
    tok = tok.rstrip(".")
    neg = tok.startswith("-")
    digits = tok.lstrip("-")
    if not digits:
        return None
    if "." not in digits:
        if len(digits) < 2:
            return None
        digits = digits[:-1] + "." + digits[-1]
    try:
        v = float(digits)
    except ValueError:
        return None
    v = -v if neg else v
    return round(v, 1) if MIN_PLAUSIBLE <= v <= MAX_PLAUSIBLE else None


def _ocr_numbers(gray):
    """OCR one label crop (white text on dark). Returns candidate values."""
    out = []
    for scale in (3, 4, 6):
        g = cv2.resize(gray, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_CUBIC)
        for mode in ("otsu", "fixed"):
            if mode == "otsu":
                _, b = cv2.threshold(
                    g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, b = cv2.threshold(g, 150, 255, cv2.THRESH_BINARY)
            b = 255 - b                        # black text on white
            b = cv2.copyMakeBorder(b, 20, 20, 20, 20,
                                   cv2.BORDER_CONSTANT, value=255)
            txt = pytesseract.image_to_string(
                b, config="--psm 7 -c tessedit_char_whitelist=0123456789.-")
            for tok in _TOKEN.findall(txt):
                v = _normalise_token(tok)
                if v is not None:
                    out.append(v)
    return out


def _vote(values):
    if not values:
        return None, 0.0
    counts = Counter(values)
    val = max(counts, key=lambda v: (counts[v], len(str(v))))
    return val, counts[val] / len(values)


def read_scale_limits(rgb, bar):
    """
    OCR the labels above the top end and below the bottom end of the bar
    (the usual place on these cameras), then fall back to windows beside
    the bar. Returns (t_min, t_max, confidence 0-1) or (None, None, 0).
    """
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = bar
    xa = max(0, x1 - 45)

    def gray(ya, yb, xs, xe):
        ya, yb = max(0, ya), min(h, yb)
        if yb - ya < 6 or xe - xs < 10:
            return None
        return cv2.cvtColor(rgb[ya:yb, xs:xe], cv2.COLOR_RGB2GRAY)

    windows = [
        (gray(y1 - 30, y1 - 1, xa, w), gray(y2 + 2, y2 + 30, xa, w)),
        (gray(y1 - 34, y1 + 14, max(0, x1 - 70), min(w, x2 + 1)),
         gray(y2 - 14, y2 + 34, max(0, x1 - 70), min(w, x2 + 1))),
        # labels beside the bar (some cameras / exports)
        (gray(y1 - 14, y1 + 20, x2 + 3, min(w, x2 + 3 + 90)),
         gray(y2 - 20, y2 + 14, x2 + 3, min(w, x2 + 3 + 90))),
    ]
    for top_img, bot_img in windows:
        if top_img is None or bot_img is None:
            continue
        t_max, c_max = _vote(_ocr_numbers(top_img))
        t_min, c_min = _vote(_ocr_numbers(bot_img))
        if t_max is None or t_min is None or t_max <= t_min:
            continue
        if (t_max - t_min) > 200:
            continue
        return t_min, t_max, float(min(c_max, c_min))
    return None, None, 0.0


# ------------------------------------------------------------
# 3. Pixel -> temperature using the bar's own palette
# ------------------------------------------------------------
def pixel_temperature_map(rgb, palette, t_min, t_max, bar, thermal_box=None):
    """
    Return a float32 temperature image the same size as `rgb`
    (NaN where a pixel does not match the palette, e.g. text overlays).
    Uses CIELAB distance, which matches perceived colour far better
    than RGB or HSV.
    """
    n = len(palette)
    temps = np.linspace(t_max, t_min, n).astype(np.float32)  # top = hot

    pal_lab = cv2.cvtColor(
        palette.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB
    ).reshape(-1, 3).astype(np.float32)

    # Subsample palette for speed; interpolate afterwards
    step = max(1, n // 128)
    pal_s, temps_s = pal_lab[::step], temps[::step]

    img_lab = cv2.cvtColor(cv2.medianBlur(rgb, 3), cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = img_lab.shape[:2]
    flat = img_lab.reshape(-1, 3)

    # Chunked nearest-neighbour search
    idx = np.empty(len(flat), dtype=np.int32)
    dist = np.empty(len(flat), dtype=np.float32)
    chunk = 200_000
    for i in range(0, len(flat), chunk):
        d = ((flat[i:i + chunk, None, :] - pal_s[None, :, :]) ** 2).sum(-1)
        idx[i:i + chunk] = d.argmin(1)
        dist[i:i + chunk] = np.sqrt(d.min(1))

    tmap = temps_s[idx].reshape(h, w)
    tmap[dist.reshape(h, w) > 18.0] = np.nan  # no good palette match

    # Mask out the colour bar itself and anything outside the thermal box
    x1, y1, x2, y2 = bar
    mask = np.ones((h, w), bool)
    if thermal_box:
        bx1, by1, bx2, by2 = thermal_box
        mask[:] = False
        mask[by1:by2, bx1:bx2] = True
    mask[y1 - 4:y2 + 4, x1 - 4:x2 + 4] = False
    tmap[~mask] = np.nan
    return tmap


def region_stats(tmap, roi=None):
    """
    Robust min / max / mean of a temperature map.
    Uses the 0.2 and 99.8 percentiles (and ignores NaN) so one noisy
    pixel cannot create a fake hotspot.
    roi = (x1, y1, x2, y2) optional rectangle.
    """
    t = tmap if roi is None else tmap[roi[1]:roi[3], roi[0]:roi[2]]
    v = t[~np.isnan(t)]
    if v.size < 50:
        return None
    return {
        "min": float(np.percentile(v, 0.2)),
        "max": float(np.percentile(v, 99.8)),
        "mean": float(v.mean()),
        "pixels": int(v.size),
    }


# ------------------------------------------------------------
# 4. Public entry point
# ------------------------------------------------------------
def analyze_thermal_image(pil_image):
    """
    Returns a dict:
      min, max      - scale limits read from the colour bar (OCR)
      hot_max       - hottest real pixel found via palette mapping
      cold_min      - coldest real pixel found via palette mapping
      confidence    - 0-1
      source        - text label for the UI / database
      warnings      - list of strings
    """
    rgb = np.array(pil_image.convert("RGB"))
    warnings = []

    bar = find_colorbar(rgb)
    if bar is None:
        return {
            "min": None, "max": None, "hot_max": None, "cold_min": None,
            "confidence": 0.0,
            "source": "No colour bar found",
            "warnings": ["Colour bar not detected - enter values manually."],
        }

    t_min, t_max, conf = read_scale_limits(rgb, bar)
    if t_min is None:
        return {
            "min": None, "max": None, "hot_max": None, "cold_min": None,
            "confidence": 0.0,
            "source": "Scale OCR failed",
            "warnings": ["Could not read scale labels - enter manually."],
        }

    palette = read_palette(rgb, bar)
    h, w = rgb.shape[:2]
    thermal_box = (0, 0, bar[0] - 6, h)  # everything left of the bar

    tmap = pixel_temperature_map(
        rgb, palette, t_min, t_max, bar, thermal_box
    )
    stats = region_stats(tmap)

    if conf < 0.5:
        warnings.append(
            f"OCR agreement only {conf:.0%} - verify scale values."
        )

    return {
        "min": t_min,
        "max": t_max,
        "hot_max": round(stats["max"], 1) if stats else None,
        "cold_min": round(stats["min"], 1) if stats else None,
        "confidence": conf,
        "source": (
            "Colour-bar OCR (verified)" if conf >= 0.5
            else "Colour-bar OCR (low confidence)"
        ),
        "warnings": warnings,
        "_tmap": tmap,  # drop before saving; useful for ROI stats / heatmap
    }


# ------------------------------------------------------------
# 5. Area-based rule: max = Area01, min = higher of Area02 / Area03
# ------------------------------------------------------------
def detect_area_boxes(rgb):
    """
    Find the cyan rectangles IR Reporter draws for Area01/02/03.
    Returns a list of (x, y, w, h), sorted top-to-bottom, left-to-right.
    The caller must still decide which box is which (see assign_areas).
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # cyan: hue ~80-100 (OpenCV 0-179), bright and saturated
    mask = cv2.inRange(hsv, (80, 90, 150), (100, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = rgb.shape[:2]
    boxes = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 12 or bh < 12:
            continue
        if bw > w * 0.4 or bh > h * 0.4:
            continue
        fill = cv2.contourArea(c) / float(bw * bh)
        if fill < 0.5:  # thin outline of a rectangle is fine, text is not
            pass
        boxes.append((x, y, bw, bh))
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def area_delta(tmap, area1, area2, area3):
    """
    tmap: temperature map from pixel_temperature_map().
    area1/2/3: (x, y, w, h) boxes.
    max_t = hottest pixel in Area01
    ref_t = HIGHER of the hottest pixel in Area02 and Area03
    delta = max_t - ref_t
    """
    def box_max(b):
        x, y, bw, bh = b
        s = region_stats(tmap, (x, y, x + bw, y + bh))
        return None if s is None else s["max"]

    m1, m2, m3 = box_max(area1), box_max(area2), box_max(area3)
    refs = [v for v in (m2, m3) if v is not None]
    if m1 is None or not refs:
        return None
    ref = max(refs)
    return {
        "area1_max": round(m1, 1),
        "area2_max": None if m2 is None else round(m2, 1),
        "area3_max": None if m3 is None else round(m3, 1),
        "max": round(m1, 1),
        "min": round(ref, 1),
        "delta": round(m1 - ref, 1),
    }


def delta_from_values(area1, area2, area3):
    """Same rule when the three values are typed in / read from the report."""
    ref = max(area2, area3)
    return {"max": area1, "min": ref, "delta": round(area1 - ref, 1)}


# ------------------------------------------------------------
# 6. Temperature at a user-clicked point
# ------------------------------------------------------------
def point_temperature(tmap, x, y, radius=6, mode="max"):
    """
    Temperature around a clicked pixel (x, y) in tmap coordinates.
    mode="max": hottest pixel within `radius` (click near the hot spot).
    mode="median": median within `radius` (click on a reference area).
    Returns None if no valid palette-matched pixels are in the window.
    """
    h, w = tmap.shape[:2]
    x0, x1 = max(0, int(x) - radius), min(w, int(x) + radius + 1)
    y0, y1 = max(0, int(y) - radius), min(h, int(y) + radius + 1)
    win = tmap[y0:y1, x0:x1]
    v = win[~np.isnan(win)]
    if v.size == 0:
        return None
    return float(v.max() if mode == "max" else np.median(v))


# ------------------------------------------------------------
# 7. Manual scale (fallback when the colour bar is not detected)
# ------------------------------------------------------------
def analyze_with_manual_bar(pil_image, bar_top_xy, bar_bottom_xy, t_max, t_min):
    """
    The user clicked the middle of the colour bar at its top and bottom
    ends and typed the scale max / min. Builds the same result dict as
    analyze_thermal_image().
    """
    rgb = np.array(pil_image.convert("RGB"))
    (xt, yt), (xb, yb) = bar_top_xy, bar_bottom_xy
    if yb <= yt or t_max <= t_min:
        return None
    xc = int(round((xt + xb) / 2))
    bar = (max(0, xc - 3), int(yt), xc + 3, int(yb))
    palette = read_palette(rgb, bar)
    if len(palette) < 10:
        return None
    tmap = pixel_temperature_map(rgb, palette, t_min, t_max, bar, None)
    stats = region_stats(tmap)
    return {
        "min": t_min, "max": t_max,
        "hot_max": None if not stats else round(stats["max"], 1),
        "cold_min": None if not stats else round(stats["min"], 1),
        "confidence": 1.0,
        "source": "Manual scale (user-entered limits)",
        "warnings": [],
        "_tmap": tmap,
    }
