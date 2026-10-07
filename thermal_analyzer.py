import cv2
import numpy as np


def analyze_thermal_image(
    image: np.ndarray,
    known_cold_temp: float = 7.0,
    known_hot_temp: float = 40.0,
    threshold: float = 5.0
) -> dict:
    """
    Maps image pixel intensities linearly based on two verified spot temperature measurements.
    """
    # Convert RGB to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Filter out extreme outer margin text and legends
    h, w = gray.shape
    roi = gray[int(h * 0.05):int(h * 0.95), int(w * 0.05):int(w * 0.95)]

    min_pixel = float(np.min(roi))
    max_pixel = float(np.max(roi))

    if max_pixel <= min_pixel:
        raise ValueError("Image lacks sufficient thermal variation for analysis.")

    # Calculate scale factor per pixel unit
    temp_per_unit = (known_hot_temp - known_cold_temp) / (max_pixel - min_pixel)

    # Convert entire ROI array to actual Celsius
    actual_temperatures = known_cold_temp + (roi - min_pixel) * temp_per_unit

    min_temp = float(np.min(actual_temperatures))
    max_temp = float(np.max(actual_temperatures))
    mean_temp = float(np.mean(actual_temperatures))
    temp_diff = max_temp - min_temp

    status = "FAULT" if temp_diff >= threshold else "NO FAULT"
    action = "Attention required within 2 days." if status == "FAULT" else "Normal operation."

    return {
        "min_temperature": round(min_temp, 2),
        "max_temperature": round(max_temp, 2),
        "temperature_difference": round(temp_diff, 2),
        "mean_temperature": round(mean_temp, 2),
        "threshold": threshold,
        "status": status,
        "action": action
    }
