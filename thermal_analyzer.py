import cv2
import numpy as np


def analyze_thermal_image(
    image: np.ndarray,
    known_cold_temp: float = 7.0,
    known_hot_temp: float = 40.0,
    threshold: float = 5.0
) -> dict:
    """
    Analyzes actual thermal variation by ignoring scale bars and cropping to 
    the active thermal target area.
    """
    # 1. Convert RGB to grayscale intensity map
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2. Crop out thermal camera side legends/scale bars (crop 15% inner border)
    h, w = gray.shape
    roi = gray[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]

    # 3. Filter out cold background noise (ignore pixels below 10th percentile)
    valid_pixels = roi[roi > np.percentile(roi, 10)]

    if len(valid_pixels) == 0:
        valid_pixels = roi.flatten()

    # 4. Use 5th percentile as ambient equipment reference and 95th as hotspot
    pixel_ambient = np.percentile(valid_pixels, 5)
    pixel_hotspot = np.percentile(valid_pixels, 95)
    pixel_mean = np.mean(valid_pixels)

    # Scale factor per pixel intensity unit based on full scale bounds
    scale_range = known_hot_temp - known_cold_temp
    temp_per_pixel = scale_range / 255.0

    # 5. Convert pixel intensities to actual temperatures
    min_temp = float(known_cold_temp + (pixel_ambient * temp_per_pixel))
    max_temp = float(known_cold_temp + (pixel_hotspot * temp_per_pixel))
    mean_temp = float(known_cold_temp + (pixel_mean * temp_per_pixel))

    # Calculate actual hotspot vs ambient temperature delta
    temp_diff = float(max_temp - min_temp)

    # 6. Fault decision
    if temp_diff < threshold:
        status = "NO FAULT"
        action = "No immediate action required."
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
