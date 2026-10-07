import cv2
import numpy as np


def analyze_thermal_image(
    image: np.ndarray,
    known_cold_temp: float = 7.0,
    known_hot_temp: float = 40.0,
    threshold: float = 5.0
) -> dict:
    """
    Isolates the active carrying component to calculate localized hotspot Delta T,
    preventing cold background air from inflating results to 25-30°C.
    """
    # 1. Convert to grayscale intensity map
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2. Crop inner 70% ROI to remove outer camera legends, text, and side margins
    h, w = gray.shape
    roi = gray[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]

    # 3. Apply mild Gaussian blur to suppress camera noise
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)

    # 4. Strictly isolate active warm equipment (ignore lowest 85% of pixels representing background air/casing)
    equipment_cutoff = np.percentile(blurred, 85)
    equipment_pixels = blurred[blurred >= equipment_cutoff]

    if len(equipment_pixels) < 10:
        equipment_pixels = blurred.flatten()

    # 5. Measure peak hotspot (top 0.5% intensity) vs healthy carrying conductor baseline (bottom of equipment region)
    hotspot_pixel = np.percentile(equipment_pixels, 99.5)
    baseline_pixel = np.percentile(equipment_pixels, 15)  # Baseline within the active conductor
    mean_pixel = np.mean(equipment_pixels)

    # 6. Map intensity values to temperature range
    scale_range = known_hot_temp - known_cold_temp
    temp_per_pixel = scale_range / 255.0

    max_temp = float(known_cold_temp + (hotspot_pixel * temp_per_pixel))
    min_temp = float(known_cold_temp + (baseline_pixel * temp_per_pixel))
    mean_temp = float(known_cold_temp + (mean_pixel * temp_per_pixel))

    # Calculate micro-localized Delta T
    temp_diff = float(max_temp - min_temp)

    # 7. Fault evaluation
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
