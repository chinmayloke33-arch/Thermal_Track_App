import cv2
import numpy as np


def analyze_thermal_image(
    image: np.ndarray,
    known_cold_temp: float = 7.0,
    known_hot_temp: float = 40.0,
    threshold: float = 5.0
) -> dict:
    """
    Measures micro-localized Delta T between peak hotspot cluster 
    and immediate adjacent conductor baseline.
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2. Crop inner 60% ROI to focus strictly on target equipment
    h, w = gray.shape
    roi = gray[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]

    # 3. Apply Gaussian blur to eliminate minor image noise
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)

    # 4. Extract peak hotspot (99.5th percentile)
    hotspot_val = np.percentile(blurred, 99.5)

    # 5. Extract immediate conductor baseline (93rd to 96th percentile region)
    # Comparing peak hotspot directly to adjacent carrying conductor
    baseline_pixels = blurred[(blurred >= np.percentile(blurred, 93)) & 
                              (blurred <= np.percentile(blurred, 96))]

    if len(baseline_pixels) == 0:
        baseline_val = np.percentile(blurred, 90)
    else:
        baseline_val = np.mean(baseline_pixels)

    # 6. Calculate temperatures in Celsius
    scale_range = known_hot_temp - known_cold_temp
    temp_per_pixel = scale_range / 255.0

    max_temp = float(known_cold_temp + (hotspot_val * temp_per_pixel))
    min_temp = float(known_cold_temp + (baseline_val * temp_per_pixel))
    mean_temp = float(known_cold_temp + (np.mean(blurred) * temp_per_pixel))

    # Calculate tight micro-localized Delta T
    temp_diff = float(max_temp - min_temp)

    # 7. Fault evaluation
    if temp_diff < threshold:
        status = "NO FAULT"
        action = "Operating within normal thermal limits."
    else:
        status = "FAULT"
        action = "Attention required within 2 days."

    return {
        "min_temperature": round(min_temp, 2),
        "max_temperature": round(max_temp, 2),
        "temperature_difference": round(temp_diff, 2),
        "mean_temperature": round(mean_temp, 2),
        "threshold": threshold,
        "status": status,
        "action": action
    }
