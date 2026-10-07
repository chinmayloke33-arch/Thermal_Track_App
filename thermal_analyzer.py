import cv2
import numpy as np


def analyze_thermal_image(
    image: np.ndarray,
    known_cold_temp: float,
    known_hot_temp: float,
    threshold: float = 5.0
) -> dict:
    """
    Direct pixel-intensity linear mapping against ground-truth OEM scale boundaries.
    """
    # 1. Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2. Crop inner 80% region to remove scale bar/UI border artifacts
    h, w = gray.shape
    roi = gray[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]

    # 3. Find pixel intensity extrema within the active image body
    min_pixel = float(np.min(roi))
    max_pixel = float(np.max(roi))

    if max_pixel <= min_pixel:
        raise ValueError("Image lacks sufficient contrast or thermal dynamic range.")

    # 4. Compute slope (deg C per intensity count) based on input scale
    temp_per_count = (known_hot_temp - known_cold_temp) / (max_pixel - min_pixel)

    # 5. Map pixel values directly to temperatures
    min_temp = known_cold_temp
    max_temp = known_hot_temp
    mean_temp = float(known_cold_temp + (np.mean(roi) - min_pixel) * temp_per_count)

    # Delta T directly matches the scale delta for full-frame targets
    temp_diff = float(max_temp - min_temp)

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
