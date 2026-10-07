import cv2
import numpy as np


def analyze_thermal_image(
    image: np.ndarray,
    known_cold_temp: float = 7.0,
    known_hot_temp: float = 40.0,
    threshold: float = 5.0
) -> dict:
    """
    Measures localized hotspot delta relative to adjacent operational baselines
    to deliver accurate physical inspection ΔT values.
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2. Crop inner 70% ROI to remove scale bars, legends, and outer background
    h, w = gray.shape
    roi = gray[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]

    # 3. Apply Gaussian blur to reduce single-pixel noise and artifacts
    blurred_roi = cv2.GaussianBlur(roi, (5, 5), 0)

    # 4. Extract localized hotspot cluster (top 2% highest intensity pixels)
    hotspot_threshold = np.percentile(blurred_roi, 98)
    hotspot_pixels = blurred_roi[blurred_roi >= hotspot_threshold]

    # 5. Extract adjacent operating component baseline (80th to 90th percentile region)
    # This represents healthy carrying equipment rather than cold background air
    baseline_lower = np.percentile(blurred_roi, 80)
    baseline_upper = np.percentile(blurred_roi, 90)
    baseline_pixels = blurred_roi[(blurred_roi >= baseline_lower) & (blurred_roi <= baseline_upper)]

    if len(baseline_pixels) == 0:
        baseline_pixels = blurred_roi

    # 6. Calculate pixel intensities
    pixel_hotspot = np.mean(hotspot_pixels)
    pixel_baseline = np.mean(baseline_pixels)
    pixel_mean = np.mean(blurred_roi)

    # Convert scale bounds
    scale_range = known_hot_temp - known_cold_temp
    temp_per_pixel = scale_range / 255.0

    max_temp = float(known_cold_temp + (pixel_hotspot * temp_per_pixel))
    min_temp = float(known_cold_temp + (pixel_baseline * temp_per_pixel))
    mean_temp = float(known_cold_temp + (pixel_mean * temp_per_pixel))

    # Calculate targeted localized Delta T
    temp_diff = float(max_temp - min_temp)

    # 7. Fault classification based on target threshold
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
