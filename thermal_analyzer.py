import cv2
import numpy as np


def analyze_thermal_image(
    image: np.ndarray,
    known_cold_temp: float = 7.0,
    known_hot_temp: float = 40.0,
    threshold: float = 5.0
) -> dict:
    """
    Computes local min/max temperatures and delta T for a user-cropped ROI,
    eliminating background scale artifacts and legend noise.
    """
    # 1. Convert ROI to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2. Extract intensity bounds within selected component
    min_pixel = float(np.min(gray))
    max_pixel = float(np.max(gray))
    mean_pixel = float(np.mean(gray))

    if max_pixel <= min_pixel:
        # Fallback if cropped area is completely uniform
        temp_diff = 0.0
        min_temp = known_cold_temp
        max_temp = known_cold_temp
        mean_temp = known_cold_temp
    else:
        # Scale slope per pixel intensity count across 0-255 spectrum
        scale_range = known_hot_temp - known_cold_temp
        temp_per_pixel = scale_range / 255.0

        min_temp = known_cold_temp + (min_pixel * temp_per_pixel)
        max_temp = known_cold_temp + (max_pixel * temp_per_pixel)
        mean_temp = known_cold_temp + (mean_pixel * temp_per_pixel)
        temp_diff = max_temp - min_temp

    # 3. Fault assessment
    if temp_diff < threshold:
        status = "NO FAULT"
        action = "Operating within normal thermal parameters."
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
